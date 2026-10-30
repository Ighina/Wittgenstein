#!/usr/bin/env python3
"""
evaluate_math_oracle.py — score orchestrate-math under ORACLE LOCALISATION.

Why this script exists
----------------------
Our headline SPOT math numbers come from the needle-in-a-haystack protocol:
the system reads the whole paper, extracts every self-contained equation,
theorem and proof (18 per paper on average), verifies each one, and is scored
on whether the errors it reports match the single annotated error — so every
other flagged statement costs precision.

A paper claiming ~89% on "the Math/CS Equation and Proof subset" of SPOT cannot
be measured that way: at 37 annotated equation/proof errors, 89% recall with a
meaningful precision would be far above anything SPOT itself reports. The
likely protocol is ORACLE LOCALISATION — the model is asked only about the
statement the ground truth points at, so (a) it never has to find the needle
and (b) nothing it says about the other statements is ever counted.

This script reconstructs that protocol for our system WITHOUT re-running it.
orchestrate-math already verifies every extracted statement in isolation and
records a per-statement verdict in `results[]`, so the verdict on the
annotated statement is exactly the answer our verifier would give if handed
that statement alone. The only thing needed is to find which extracted
statement is the annotated one — done by an LLM judge that is shown the
statements WITHOUT our verdicts, so localisation cannot be biased by them.

Metrics reported
----------------
coverage            fraction of papers where the annotated statement was among
                    the extracted ones (covered or partial). An extractor miss
                    is invisible under oracle localisation, since the oracle
                    protocol hands the statement over for free.
detect_raw          papers where a located statement carries
                    error_detected = true (any confidence) / all papers
detect_conf         same, restricted to confidence >= 0.7 (the threshold above
                    which orchestrate-math promotes a verdict to findings[])
detect_raw_cov      detect_raw over covered papers only — the most lenient
                    reading, and the closest analogue of a reported accuracy
                    on oracle-supplied statements
detect_same_defect  flagged AND a second judge call confirms our stated reason
                    describes the annotated defect rather than some other
                    problem in the same statement
fp_rate             fraction of the OTHER extracted statements (the ones with
                    no annotated error) that we also flag. Oracle localisation
                    never looks at these, which is precisely what it hides.

Usage
-----
  python scripts/evaluate_math_oracle.py --workspace-dir data/spot-workspace \
      --tag opus5 --judge-model claude-sonnet-5
  ... --subset math-cs          # Mathematics + Computer Science papers only
  ... --no-defect-check         # skip judge call B (localisation only)
"""

import argparse
import csv
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_verifier import (  # noqa: E402
    filter_ground_truth,
    load_anthropic_client,
    load_json,
    prompt_fingerprint,
    save_json,
    _client_matches,
    _model_tag,
)
from llm_providers import complete  # noqa: E402

PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "task_specific_output" / "math-oracle-spot"

MATH_TASK = "math"
CONF_THRESHOLD = 0.7          # orchestrate-math's findings[] promotion cut-off
SNIPPET_CHARS = 1200          # per-statement text shown to the judge
MAX_TOKENS = 4000
MAX_RETRIES = 3
PROMPT_VERSION = "1"

MATH_CS = {"Mathematics", "Computer Science"}


# ---------------------------------------------------------------------------
# Report reading
# ---------------------------------------------------------------------------


def report_name(tag: Optional[str]) -> str:
    return f"paper_verified_math-{tag}.json" if tag else "paper_verified_math.json"


def detector_report_name(tag: str) -> str:
    return f"math-detector-{tag}.json"


def report_from_detector(detector: Dict[str, Any]) -> Dict[str, Any]:
    """Shape a math-detector-<tag>.json like an orchestrator report.

    Scoring the detector ALONE is the measurement that isolates v1's binding
    constraint. v1's coverage — the annotated statement extracted and
    self-contained — was 59.5% on SPOT's equation/proof papers, and no verifier
    can check a statement it never receives. Running the same localisation
    judge over the detector's candidates says whether detect-first lifts that
    ceiling, with no verifier stage involved and nothing else changed.

    Every candidate is by definition a suspected error, so `error_detected` is
    true for all of them. That makes `detect_raw` degenerate (it equals
    coverage) and `fp_rate_clean` meaningless here — a detector has no
    unflagged statements to be wrong about. COVERAGE is the number to read.
    """
    results: List[Dict[str, Any]] = []
    for c in detector.get("candidates") or []:
        if not isinstance(c, dict) or not c.get("candidate_id"):
            continue
        results.append(
            {
                "snippet_id": str(c["candidate_id"]),
                "snippet_type": str(c.get("check_kind") or "candidate").upper(),
                "location": c.get("error_location") or "",
                "content": c.get("claim_content") or c.get("exact_quote") or "",
                "exact_quote": c.get("exact_quote") or "",
                "suspected_defect": c.get("suspected_defect") or "",
                "status": "ERROR_DETECTED",
                "error_detected": True,
                "confidence": c.get("confidence") or 0.0,
            }
        )
    return {
        "skill": "math-detector",
        "paper_name": detector.get("paper_id"),
        "results": results,
        # findings[] mirrors results[] so judge call B can read a reason: the
        # detector's `suspected_defect` IS its evidence.
        "findings": [
            {
                "snippet_id": r["snippet_id"],
                "evidence": r["suspected_defect"],
                "exact_quote": r["exact_quote"],
                "confidence": r["confidence"],
            }
            for r in results
        ],
    }


def as_float(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def snippet_index(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """{snippet_id: result} for every statement the skill checked."""
    out: Dict[str, Dict[str, Any]] = {}
    for r in report.get("results") or []:
        if isinstance(r, dict) and r.get("snippet_id"):
            out[str(r["snippet_id"])] = r
    return out


def finding_index(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for f in report.get("findings") or []:
        if isinstance(f, dict) and f.get("snippet_id"):
            out[str(f["snippet_id"])] = f
    return out


def truncate(text: Any, limit: int = SNIPPET_CHARS) -> str:
    s = str(text or "").strip()
    return s if len(s) <= limit else s[:limit] + " …[truncated]"


# ---------------------------------------------------------------------------
# Judge call A — localisation (verdict-blind)
# ---------------------------------------------------------------------------


def build_locate_prompt(
    title: str, gt: Dict[str, Any], snippets: List[Dict[str, Any]]
) -> str:
    lines = [
        "You are localising a KNOWN, published error inside a scientific paper.",
        "",
        f"PAPER: {title}",
        "",
        "THE ANNOTATED ERROR (ground truth, from the SPOT benchmark):",
        f"  Annotated location: {gt.get('location') or 'unspecified'}",
        f"  Description: {truncate(gt.get('error_explanation'), 3000)}",
    ]
    target = str(gt.get("target_find") or "").strip()
    if target:
        lines.append(f"  Erroneous text as annotated: {truncate(target, 1500)}")
    lines += [
        "",
        "A verification system extracted the following self-contained "
        "mathematical statements from this paper. Each is shown with the "
        "section its extractor recorded and its text. You are NOT told what "
        "the system concluded about any of them.",
        "",
    ]
    for i, s in enumerate(snippets, 1):
        lines += [
            f"[{i}] id={s['snippet_id']}  type={s.get('snippet_type') or '?'}  "
            f"location={truncate(s.get('location'), 160)}",
            f"    {truncate(s.get('content'))}",
            "",
        ]
    lines += [
        "TASK",
        "Decide which of these extracted statements, if any, CONTAIN the "
        "annotated error — meaning a careful reviewer handed that statement "
        "alone, with no other part of the paper, and asked 'is this correct?' "
        "could discover the annotated defect from the statement's own text.",
        "",
        "Rules:",
        "- Judge CONTAINMENT of the defect, not topical similarity. A statement "
        "that merely mentions the same objects, or cites the broken result "
        "without restating it, does NOT contain the error.",
        "- A defect may live in more than one statement (e.g. a wrong lemma and "
        "the proof that uses it). List every statement that contains it.",
        "- coverage = \"covered\": at least one listed statement contains the "
        "defect and carries enough of it to be caught on its own.",
        "- coverage = \"partial\": the defect is only partly visible in the "
        "listed statements — e.g. the annotation is about a dependency between "
        "two results and only one side was extracted, or catching it needs a "
        "figure, table or numeric table that is not among the statements. "
        "Still list the statements that carry the visible part.",
        "- coverage = \"absent\": no extracted statement contains the defect. "
        "Use an empty located_ids.",
        "",
        "Return ONLY a JSON object, no prose:",
        '{"coverage": "covered" | "partial" | "absent", '
        '"located_ids": ["<snippet id>", ...], '
        '"rationale": "<at most 60 words>"}',
    ]
    return "\n".join(lines)


def validate_locate(obj: Any, valid_ids: set) -> Optional[str]:
    if not isinstance(obj, dict):
        return "not a JSON object"
    cov = obj.get("coverage")
    if cov not in ("covered", "partial", "absent"):
        return f"coverage must be covered/partial/absent, got {cov!r}"
    ids = obj.get("located_ids")
    if not isinstance(ids, list) or any(not isinstance(x, str) for x in ids):
        return "located_ids must be a list of strings"
    unknown = [x for x in ids if x not in valid_ids]
    if unknown:
        return f"located_ids contains unknown snippet ids: {unknown[:3]}"
    if cov == "absent" and ids:
        return "coverage=absent but located_ids is non-empty"
    if cov in ("covered", "partial") and not ids:
        return f"coverage={cov} but located_ids is empty"
    return None


# ---------------------------------------------------------------------------
# Judge call B — does our stated reason describe the annotated defect?
# ---------------------------------------------------------------------------


def build_defect_prompt(
    title: str, gt: Dict[str, Any], flagged: List[Dict[str, Any]]
) -> str:
    lines = [
        "A verification system flagged a mathematical statement in a paper as "
        "erroneous. Decide whether the defect it describes is THE SAME defect "
        "as the one recorded in the benchmark annotation.",
        "",
        f"PAPER: {title}",
        "",
        "ANNOTATED DEFECT (ground truth):",
        f"  Annotated location: {gt.get('location') or 'unspecified'}",
        f"  Description: {truncate(gt.get('error_explanation'), 3000)}",
        "",
        "WHAT THE SYSTEM SAID about the statement(s) at that location:",
        "",
    ]
    for s in flagged:
        lines += [
            f"  statement id={s['snippet_id']}  type={s.get('snippet_type') or '?'}  "
            f"confidence={s.get('confidence')}  status={s.get('status')}",
            f"  its reason: {truncate(s.get('reason'), 3000) or '(none recorded)'}",
            "",
        ]
    lines += [
        "TASK",
        "Answer true only if the system's reason identifies the same underlying "
        "mathematical defect as the annotation — the same broken step, identity, "
        "sign, bound, hypothesis or dependency. Answer false if it flags the "
        "statement for an unrelated reason, or merely expresses vague doubt "
        "about it, or restates the statement without naming a defect.",
        "Partial credit counts as true when the system names the annotated "
        "defect and adds further complaints alongside it.",
        "",
        "Return ONLY a JSON object, no prose:",
        '{"same_defect": true | false, "rationale": "<at most 50 words>"}',
    ]
    return "\n".join(lines)


def validate_defect(obj: Any) -> Optional[str]:
    if not isinstance(obj, dict):
        return "not a JSON object"
    if not isinstance(obj.get("same_defect"), bool):
        return "same_defect must be a boolean"
    return None


# ---------------------------------------------------------------------------
# Judge plumbing: parse, cache, call
# ---------------------------------------------------------------------------


def parse_json(raw: str) -> Any:
    """Pull the first JSON object out of a model reply, fences and all."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text[3:]
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def cached_call(
    client: Any,
    judge_model: str,
    prompt: str,
    validator,
    cache_path: Path,
    cache_key: str,
    overwrite: bool,
) -> Optional[Dict[str, Any]]:
    """One judge call, memoised on (cache_key, prompt fingerprint).

    The fingerprint IS part of the key, so a changed rubric, changed ground
    truth or a re-run of the skill with different snippets all miss the cache
    instead of silently returning a stale verdict.
    """
    fp = prompt_fingerprint(prompt)
    cache: Dict[str, Any] = {}
    if cache_path.exists():
        try:
            cache = load_json(cache_path) or {}
        except Exception:
            cache = {}
    entry = cache.get(cache_key)
    if (
        not overwrite
        and isinstance(entry, dict)
        and entry.get("prompt_fingerprint") == fp
        and entry.get("prompt_version") == PROMPT_VERSION
        and isinstance(entry.get("result"), dict)
    ):
        return entry["result"]

    last_err = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = complete(
                model=judge_model,
                prompt=prompt,
                max_tokens=MAX_TOKENS,
                temperature=0.0 if attempt == 1 else 0.3,
                client=client if _client_matches(client, judge_model) else None,
            )
        except Exception as e:
            last_err = f"API error: {type(e).__name__}: {e}"
            print(f"  [JUDGE] {cache_key} attempt {attempt}: {last_err}",
                  file=sys.stderr)
            continue
        parsed = parse_json(response.text)
        err = validator(parsed)
        if err is None:
            cache[cache_key] = {
                "judge_model": judge_model,
                "prompt_version": PROMPT_VERSION,
                "prompt_fingerprint": fp,
                "result": parsed,
            }
            save_json(cache_path, cache)
            return parsed
        last_err = err
        print(
            f"  [JUDGE] {cache_key} attempt {attempt}: {err} "
            f"(raw: {(response.text or '')[:160]!r})",
            file=sys.stderr,
        )
    print(f"  [JUDGE FAILURE] {cache_key}: {last_err}", file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# Per-paper scoring
# ---------------------------------------------------------------------------


def score_paper(
    paper_dir: Path,
    tag: Optional[str],
    client: Any,
    judge_model: str,
    overwrite: bool,
    defect_check: bool,
    from_detector: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    paper_id = paper_dir.name
    gt_all = load_json(paper_dir / "ground_truth.json")
    gt_math = filter_ground_truth(gt_all, MATH_TASK)
    if not gt_math:
        return None
    # SPOT carries exactly one annotated error per paper.
    gt = gt_math[0]

    if from_detector:
        report = report_from_detector(
            load_json(paper_dir / detector_report_name(from_detector))
        )
    else:
        report = load_json(paper_dir / report_name(tag))
    snippets = list(snippet_index(report).values())
    findings = finding_index(report)

    meta: Dict[str, Any] = {}
    meta_path = paper_dir / "metadata.json"
    if meta_path.exists():
        meta = load_json(meta_path) or {}
    title = meta.get("title") or report.get("paper_name") or paper_id
    category = meta.get("paper_category") or ""

    row: Dict[str, Any] = {
        "paper_id": paper_id,
        "paper_category": category,
        "severity": gt.get("_spot_severity") or "",
        "gt_location": gt.get("location") or "",
        "n_snippets": len(snippets),
        "n_flagged_total": sum(1 for s in snippets if s.get("error_detected")),
        "coverage": "absent",
        "located_ids": "",
        "n_located": 0,
        "flag_raw": False,
        "flag_conf": False,
        "best_confidence": 0.0,
        "in_findings": False,
        "located_statuses": "",
        "abstained": False,
        "same_defect": "",
        "locate_rationale": "",
        "defect_rationale": "",
        "judge_failed": False,
    }

    if not snippets:
        row["locate_rationale"] = "the skill extracted no statements from this paper"
        return row

    cache_key_tag = f"det-{from_detector}" if from_detector else (tag or "untagged")
    cache_path = paper_dir / (
        f"judge-mathoracle-{cache_key_tag}-{_model_tag(judge_model)}.json"
    )
    valid_ids = {str(s["snippet_id"]) for s in snippets}

    located = cached_call(
        client,
        judge_model,
        build_locate_prompt(title, gt, snippets),
        lambda o: validate_locate(o, valid_ids),
        cache_path,
        "locate",
        overwrite,
    )
    if located is None:
        row["judge_failed"] = True
        return row

    ids = [i for i in located["located_ids"] if i in valid_ids]
    by_id = snippet_index(report)
    row["coverage"] = located["coverage"]
    row["located_ids"] = ";".join(ids)
    row["n_located"] = len(ids)
    row["locate_rationale"] = str(located.get("rationale") or "")[:400]

    row["located_statuses"] = ";".join(
        str(by_id[sid].get("status") or "?") for sid in ids
    )
    # UNVERIFIABLE on every located statement is an ABSTENTION, not a clean
    # bill of health: a forced-choice oracle protocol has no such answer, so it
    # is reported apart from the misses rather than folded into them.
    row["abstained"] = bool(ids) and all(
        str(by_id[sid].get("status")) == "UNVERIFIABLE" for sid in ids
    )

    flagged = []
    for sid in ids:
        s = by_id[sid]
        if not s.get("error_detected"):
            continue
        conf = as_float(s.get("confidence"))
        f = findings.get(sid) or {}
        # Each skill records its reason in a different place, and a result that
        # was not promoted to findings[] has no `evidence` to read. Gather from
        # every field any of them uses, so judge call B is never handed an
        # empty reason for a flagged statement:
        #   orchestrate-math (v1): reasoning, lean_result, fallback_result
        #   v2:                    reconciliation.note, baseline_result
        #   v3:                    suspected_defect, adjudication.note
        recon = s.get("reconciliation") or {}
        adj = s.get("adjudication") or {}
        reason = " ".join(
            str(x)
            for x in (
                s.get("reasoning"),
                f.get("evidence"),
                f.get("exact_quote"),
                s.get("suspected_defect"),
                adj.get("note") if isinstance(adj, dict) else None,
                recon.get("note") if isinstance(recon, dict) else None,
                s.get("lean_result"),
                s.get("fallback_result"),
            )
            if x and str(x) != "None"
        )
        flagged.append(
            {
                "snippet_id": sid,
                "snippet_type": s.get("snippet_type"),
                "confidence": conf,
                "status": s.get("status"),
                "reason": reason,
            }
        )
        row["flag_raw"] = True
        row["best_confidence"] = max(row["best_confidence"], conf)
        if conf >= CONF_THRESHOLD:
            row["flag_conf"] = True
        if sid in findings:
            row["in_findings"] = True

    if flagged and defect_check:
        verdict = cached_call(
            client,
            judge_model,
            build_defect_prompt(title, gt, flagged),
            validate_defect,
            cache_path,
            "defect",
            overwrite,
        )
        if verdict is None:
            row["same_defect"] = "judge_failed"
        else:
            row["same_defect"] = bool(verdict["same_defect"])
            row["defect_rationale"] = str(verdict.get("rationale") or "")[:400]

    return row


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def rate(num: int, den: int) -> float:
    return num / den if den else 0.0


def summarize(
    rows: List[Dict[str, Any]], label: str, detector_mode: bool = False
) -> Dict[str, Any]:
    n = len(rows)
    # Two coverage sets, and the difference matters: an oracle hands over a
    # SELF-CONTAINED statement, so `strict` is the oracle-supply analogue and
    # `loose` only says the statement was reached at all. The ladder's rungs E
    # and F use strict for that reason; both conditionals are reported here so
    # the two tables cannot be read as disagreeing.
    loose_covered = [r for r in rows if r["coverage"] in ("covered", "partial")]
    covered = [r for r in rows if r["coverage"] == "covered"]
    strictly_covered = covered
    raw = [r for r in rows if r["flag_raw"]]
    conf = [r for r in rows if r["flag_conf"]]
    same = [r for r in rows if r["flag_raw"] and r["same_defect"] is True]
    abstained = [r for r in rows if r["abstained"]]
    decided = [r for r in covered if not r["abstained"]]
    cov_ids = {r["paper_id"] for r in covered}
    loose_ids = {r["paper_id"] for r in loose_covered}
    raw_cov = [r for r in raw if r["paper_id"] in cov_ids]
    conf_cov = [r for r in conf if r["paper_id"] in cov_ids]
    raw_loose = [r for r in raw if r["paper_id"] in loose_ids]

    # False-positive rate over the statements with NO annotated error: every
    # flag beyond the located one(s). Oracle localisation never sees these.
    other_total = sum(r["n_snippets"] - r["n_located"] for r in rows)
    other_flagged = sum(
        r["n_flagged_total"] - (1 if r["flag_raw"] else 0) for r in rows
    )
    return {
        "subset": label,
        "papers": n,
        "coverage_strict": rate(len(strictly_covered), n),
        "coverage": rate(len(loose_covered), n),
        "detect_raw": rate(len(raw), n),
        "detect_conf": rate(len(conf), n),
        # A conditional must intersect its own denominator. Dividing ALL
        # flagged papers by only the covered ones counts a flag on a
        # partial-coverage paper against a denominator that excludes it, and
        # the ratio can even exceed what rung E reports for the same quantity.
        "detect_raw_cov": rate(len(raw_cov), len(covered)),
        "detect_conf_cov": rate(len(conf_cov), len(covered)),
        "detect_raw_cov_loose": rate(len(raw_loose), len(loose_covered)),
        "detect_raw_decided": rate(
            len([r for r in raw_cov if not r["abstained"]]), len(decided)
        ),
        "abstain_rate": rate(len(abstained), n),
        "n_abstained": len(abstained),
        "n_decided": len(decided),
        "detect_same_defect": rate(len(same), n),
        "detect_same_defect_cov": rate(len(same), len(covered)),
        "n_covered": len(covered),
        "n_covered_loose": len(loose_covered),
        "n_flag_raw": len(raw),
        "n_flag_conf": len(conf),
        "n_same_defect": len(same),
        "statements_total": sum(r["n_snippets"] for r in rows),
        "statements_clean": other_total,
        "statements_clean_flagged": other_flagged,
        # None in detector-only mode: a detector has no unflagged statements,
        # so there is no clean-statement population to be wrong about.
        "fp_rate_clean": None if detector_mode else rate(other_flagged, other_total),
    }



# ---------------------------------------------------------------------------
# Protocol ladder
#
# The point of the whole exercise: ONE table from the strictest protocol (ours)
# to the most generous one a reported "accuracy on the annotated equations"
# could plausibly mean, so a headline number from another paper can be placed
# on it instead of compared to ours as if the two measured the same thing.
# ---------------------------------------------------------------------------


def load_headline_recall(csv_path: Path, system: str) -> Dict[str, bool]:
    """{paper_id: did the alignment judge match a finding to the GT error}.

    Read from evaluate_math_systems.py's per-paper CSV — the needle-in-a-
    haystack protocol behind our headline recall. Rung A below is that number,
    and the union rungs need it because the two judges disagree on a handful of
    papers: the alignment judge credits a finding our localiser placed
    elsewhere in the paper, and vice versa. Taking the union rather than
    picking one makes the lenient rungs upper bounds, which is the right
    direction for a rung meant to be generous.
    """
    if not csv_path.exists():
        print(f"NOTE: {csv_path} not found — the ladder's rung A is unavailable. "
              f"Run scripts/evaluate_math_systems.py first.")
        return {}
    out: Dict[str, bool] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("system") != system:
                continue
            try:
                out[r["paper_id"]] = int(r["matched_gt"]) > 0
            except (KeyError, ValueError):
                continue
    if not out:
        print(f"NOTE: no rows for system={system!r} in {csv_path}")
    return out


LADDER_RUNGS = [
    ("A", "full paper, judge-matched findings (our headline recall)"),
    ("B", "oracle site flagged, reason must be the annotated defect"),
    ("C", "oracle site flagged, any reason accepted"),
    ("D", "most lenient measurable: C or A"),
    ("E", "C over papers whose annotated statement we extracted"),
    ("F", "D over papers whose annotated statement we extracted"),
    ("G", "CEILING: D plus every extraction miss assumed caught"),
]


def ladder_rows(
    rows: List[Dict[str, Any]], headline: Dict[str, bool], label: str
) -> List[Dict[str, Any]]:
    n = len(rows)
    if not n:
        return []
    ids = {r["paper_id"] for r in rows}
    matched = {r["paper_id"] for r in rows if headline.get(r["paper_id"])}
    flag = {r["paper_id"] for r in rows if r["flag_raw"]}
    same = {r["paper_id"] for r in rows if r["same_defect"] is True}
    covered = {r["paper_id"] for r in rows if r["coverage"] == "covered"}
    # Papers where the annotated statement never reached the verifier AND
    # nothing credited it anyway. An oracle that hands the statement over
    # removes exactly this failure mode, so rung G assumes every one lands.
    extraction_miss = (ids - covered) - flag - matched

    sets = {
        "A": matched,
        "B": same | (matched & flag),
        "C": flag,
        "D": flag | matched,
        "E": flag & covered,
        "F": (flag | matched) & covered,
        "G": (flag | matched) | extraction_miss,
    }
    out: List[Dict[str, Any]] = []
    for rung, desc in LADDER_RUNGS:
        hits = sets[rung]
        den = len(covered) if rung in ("E", "F") else n
        if not den:
            continue
        hit_n = len(hits & covered) if rung in ("E", "F") else len(hits)
        out.append(
            {
                "subset": label,
                "rung": rung,
                "protocol": desc,
                "hits": hit_n,
                "papers": den,
                "score": rate(hit_n, den),
            }
        )
    return out


def print_ladder(all_rungs: List[Dict[str, Any]]) -> None:
    print(f"\n{'=' * 104}")
    print("PROTOCOL LADDER — the same system and the same errors, scored the way")
    print("different protocols would score them. A number quoted without its")
    print("protocol can sit anywhere on this ladder.")
    print(f"{'=' * 104}")
    subsets: List[str] = []
    for r in all_rungs:
        if r["subset"] not in subsets:
            subsets.append(r["subset"])
    head = f"{'':>3} {'Protocol':<56}" + "".join(f"{s[:16]:>17}" for s in subsets)
    print(head)
    print("-" * len(head))
    for rung, desc in LADDER_RUNGS:
        cells = ""
        for sub in subsets:
            m = next(
                (r for r in all_rungs if r["rung"] == rung and r["subset"] == sub),
                None,
            )
            cells += (
                f"{m['hits']}/{m['papers']}={m['score']:.3f}".rjust(17) if m else
                "".rjust(17)
            )
        print(f"{rung:>3} {desc:<56}{cells}")
    print(
        "\nRungs E and F change the DENOMINATOR to the papers whose annotated\n"
        "statement our extractor actually produced; A-D and G keep every paper.\n"
        "G is an upper bound, not a measurement: it assumes the verifier would\n"
        "catch every annotated statement it never got to see. Closing that gap\n"
        "needs an oracle-supply re-run, not a recomputation."
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score orchestrate-math under oracle localisation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage")[1],
    )
    ap.add_argument("--workspace-dir", default="data/spot-workspace")
    ap.add_argument("--tag", default="opus5",
                    help="orchestrate-math report tag (default: opus5)")
    ap.add_argument(
        "--from-detector",
        default=None,
        metavar="TAG",
        help="Score the DETECTION STAGE alone: read math-detector-<TAG>.json "
             "instead of an orchestrator report, treating every candidate as a "
             "flagged statement. Isolates whether detect-first lifts v1's 59.5%% "
             "coverage ceiling. In this mode read COVERAGE; detect_raw is "
             "degenerate (every candidate is flagged) and clean-FP is "
             "meaningless (a detector has no unflagged statements).",
    )
    ap.add_argument("--judge-model", default="claude-sonnet-5")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--overwrite-judge", action="store_true")
    ap.add_argument("--no-defect-check", action="store_true",
                    help="skip judge call B (localisation only)")
    ap.add_argument(
        "--headline-csv",
        default=str(
            PROJECT_ROOT
            / "task_specific_output"
            / "math-comparison-spot"
            / "math_systems_per_paper.csv"
        ),
        help="per-paper CSV from evaluate_math_systems.py, for the ladder's "
             "rung A (our headline needle-in-a-haystack recall)",
    )
    ap.add_argument(
        "--headline-system",
        default=None,
        help="system name in --headline-csv (default: orchestrate-math[:<tag>])",
    )
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = ap.parse_args()

    ws = Path(args.workspace_dir)
    wanted = (
        detector_report_name(args.from_detector)
        if args.from_detector
        else report_name(args.tag)
    )
    paper_dirs = sorted(
        d
        for d in ws.iterdir()
        if d.is_dir()
        and (d / "ground_truth.json").exists()
        and (d / wanted).exists()
    )
    if not paper_dirs:
        print(f"ERROR: no papers with {wanted} in {ws}")
        sys.exit(1)

    client = load_anthropic_client(args.api_key)
    what = (
        f"detector:{args.from_detector}" if args.from_detector else f"tag:{args.tag}"
    )
    print(f"Judge: {args.judge_model}   workspace: {ws}   {what}")
    if args.from_detector:
        print(
            "Detector-only mode: read COVERAGE. detect_* equals coverage by "
            "construction and clean-FP does not apply."
        )

    if args.limit:
        paper_dirs = paper_dirs[: args.limit]

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(
            pool.map(
                lambda d: score_paper(
                    d, args.tag, client, args.judge_model,
                    args.overwrite_judge, not args.no_defect_check,
                    from_detector=args.from_detector,
                ),
                paper_dirs,
            )
        )
    rows = [r for r in results if r is not None]
    print(f"Papers with math ground truth: {len(rows)}")

    failed = [r for r in rows if r["judge_failed"]]
    if failed:
        print(f"WARNING: localisation judge failed for "
              f"{', '.join(r['paper_id'] for r in failed)}")
    rows = [r for r in rows if not r["judge_failed"]]

    for r in sorted(rows, key=lambda r: r["paper_id"]):
        mark = "HIT " if r["flag_raw"] else "miss"
        print(
            f"  {mark} {r['paper_id'][:58]:<58} {r['paper_category'][:16]:<16} "
            f"cov={r['coverage']:<7} located={r['n_located']}/{r['n_snippets']} "
            f"conf={r['best_confidence']:.2f} same_defect={r['same_defect']}"
        )

    subsets = [
        ("all Equation/proof", rows),
        ("Math+CS", [r for r in rows if r["paper_category"] in MATH_CS]),
        ("Mathematics", [r for r in rows if r["paper_category"] == "Mathematics"]),
        ("Computer Science",
         [r for r in rows if r["paper_category"] == "Computer Science"]),
    ]
    summaries = [
        summarize(rs, label, detector_mode=bool(args.from_detector))
        for label, rs in subsets
        if rs
    ]

    print(f"\n{'=' * 118}")
    print("ORACLE-LOCALISATION SCORES — orchestrate-math on SPOT equation/proof errors")
    print(f"{'=' * 118}")
    head = (
        f"{'Subset':<20}{'N':>4}{'cover':>7}{'cover+':>8}{'detect':>8}"
        f"{'det>=.7':>9}{'det|cov':>9}{'det|cov+':>9}{'abstain':>9}"
        f"{'det&same':>10}{'clean FP':>10}"
    )
    print(head)
    print("-" * len(head))
    for s in summaries:
        print(
            f"{s['subset']:<20}{s['papers']:>4}{s['coverage_strict']:>7.3f}"
            f"{s['coverage']:>8.3f}{s['detect_raw']:>8.3f}{s['detect_conf']:>9.3f}"
            f"{s['detect_raw_cov']:>9.3f}{s['detect_raw_cov_loose']:>9.3f}"
            f"{s['abstain_rate']:>9.3f}{s['detect_same_defect']:>10.3f}"
            + (
                "       n/a" if s["fp_rate_clean"] is None
                else f"{s['fp_rate_clean']:>10.3f}"
            )
        )
    print(
        "\ncover   = annotated statement extracted and self-contained "
        "(coverage='covered')\n"
        "cover+  = cover plus partial coverage\n"
        "detect  = annotated statement flagged error_detected, any confidence\n"
        "det>=.7 = same, at the confidence that promotes a verdict to findings[]\n"
        "det|cov = detect over STRICTLY covered papers — the statement arrived\n"
        "          self-contained. The closest analogue of an accuracy reported\n"
        "          on oracle-supplied statements, and the denominator the\n"
        "          ladder's rungs E and F use.\n"
        "det|cov+= the same over cover+ (partial included), i.e. the statement\n"
        "          was reached but not necessarily in a catchable form\n"
        "abstain = papers where every located statement came back UNVERIFIABLE\n"
        "det&same= flagged AND the stated reason is the annotated defect\n"
        "clean FP= share of the OTHER extracted statements we also flag; the\n"
        "          oracle protocol never looks at these"
    )

    cov = Counter(r["coverage"] for r in rows)
    print(f"\nCoverage breakdown (all): {dict(cov)}")

    headline_system = args.headline_system or (
        f"orchestrate-math:{args.tag}" if args.tag else "orchestrate-math"
    )
    headline = load_headline_recall(Path(args.headline_csv), headline_system)
    all_rungs: List[Dict[str, Any]] = []
    for label, rs in subsets:
        if rs:
            all_rungs += ladder_rows(rs, headline, label)
    if all_rungs:
        print_ladder(all_rungs)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.from_detector:
        stem = f"oracle-detector-{args.from_detector}"
    else:
        stem = f"oracle-{args.tag}" if args.tag else "oracle"
    sum_path = out / f"{stem}_summary.csv"
    with open(sum_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)
    pp_path = out / f"{stem}_per_paper.csv"
    with open(pp_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["paper_id"]))
    print(f"\nSummary saved to:   {sum_path}")
    print(f"Per-paper saved to: {pp_path}")
    if all_rungs:
        lad_path = out / f"{stem}_ladder.csv"
        with open(lad_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rungs[0].keys()))
            w.writeheader()
            w.writerows(all_rungs)
        print(f"Ladder saved to:    {lad_path}")


if __name__ == "__main__":
    main()
