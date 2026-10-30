---
name: verify-math-lean
description: Formally verify a mathematical claim from a scientific paper by autoformalising it into Lean 4 and compiling the proof. Creates/reuses a Lean project under MATH_VERIFICATION/, drives /lean4:autoformalize and /lean4:autoprove, compiles via the lean-lsp MCP server, and returns the Lean compiler verdict plus its feedback. Deletes its Lean project when done unless invoked with --keep-lean.
---

# Verify Math with Lean 4

You verify ONE mathematical excerpt (an equation, derivation step, lemma, or
formula block) from a scientific paper by **formalising it in Lean 4 and letting
the Lean compiler decide**. Unlike `verify-math`, which reasons in natural
language, this skill produces a machine-checked verdict: either the Lean kernel
accepts a proof of the claim, or it does not.

## Invocation

```
verify-math-lean [--project=<Name>] [--keep-lean]
```

| Flag | Default | Effect |
|---|---|---|
| `--project=<Name>` | derived from the paper (Step 0) | Use exactly this project directory name under `MATH_VERIFICATION/`. |
| `--keep-lean` | off | Keep the Lean project under `MATH_VERIFICATION/` after the run instead of deleting it (Step 6). |

`--project` exists so a caller that invokes this skill repeatedly — an
orchestrator walking a paper's snippets — can pin every call to one project and
actually get the `--keep-lean` reuse. Without it each invocation derives its own
name and `--keep-lean` leaves a separate multi-gigabyte project per snippet.

A Mathlib-backed Lean project is **~7.5 GB** on disk. The default is therefore to
delete the project once the verdict is reported. Pass `--keep-lean` when you are
about to verify more snippets from the same paper across separate invocations, or
when a human wants to inspect the proofs afterwards — the saving is real (a fresh
project costs a multi-thousand-file Mathlib cache download plus a full build).

## When to Use This Skill

Use for snippets of type `EQUATION`, `DERIVATION`, `FORMULA_BLOCK`,
`PROOF_STEP`, `THEOREM`, or `MATHEMATICAL_MODEL` whose content is
**self-contained enough to state formally** — a closed-form identity, an
inequality, a bound, a limit, a combinatorial or algebraic claim.

Do **not** use it (return `UNVERIFIABLE` immediately, with
`lean_status = "NOT_ATTEMPTED"`) when the excerpt:

- depends on undefined domain-specific objects the paper never pins down,
- is an empirical/numerical claim about experimental data rather than a theorem,
- is a modelling assumption or definition rather than an assertion, or
- cannot be stated without inventing hypotheses that would make the claim
  trivially true (see **Faithfulness gate** below).

## Dependencies (one-time, per machine)

This skill needs two things installed. Check first; only install what is missing.
Full setup guide, smoke test, and troubleshooting: `docs/math-verification.md`.

1. **`lean-lsp` MCP server** (project scope, from the repo root):

   ```bash
   claude mcp add lean-lsp -s project uvx lean-lsp-mcp
   ```

   Verify with `claude mcp list`. It exposes `lean_build`,
   `lean_diagnostic_messages`, `lean_goal`, `lean_run_code`,
   `lean_multi_attempt`, `lean_verify`, `lean_local_search`, `lean_loogle`,
   `lean_leansearch`, `lean_state_search`, `lean_hammer_premise`, and others.
   File-taking tools want an **absolute** `file_path`; `lean_build` takes
   `lean_project_path`.

2. **`lean4` skills plugin** (provides `/lean4:autoformalize`, `/lean4:autoprove`):

   ```
   /plugin marketplace add cameronfreer/lean4-skills
   /plugin install lean4
   ```

   (Non-interactive equivalents: `claude plugin marketplace add
   cameronfreer/lean4-skills` and `claude plugin install lean4@lean4-skills`.)

3. **Lean toolchain** — `elan`, `lake`, `lean` on PATH. If absent:

   ```bash
   curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y
   export PATH="$HOME/.elan/bin:$PATH"
   elan toolchain install leanprover/lean4:stable
   elan default leanprover/lean4:stable
   ```

## Step 0 — Ensure the Lean project exists

All verification projects live under `MATH_VERIFICATION/` at the repo root.

1. Determine the **project name**. If `--project=<Name>` was passed, use it
   verbatim. Otherwise pick one: CamelCase, derived from the paper/snippet being
   checked (e.g. `SpectralGapBounds`, `PaperArxiv2401_12345`). Reuse the same
   name for every snippet of the same paper — the built project is expensive and
   should be shared.
2. Check for `MATH_VERIFICATION/<ProjectName>/lakefile.toml` (or
   `lakefile.lean`). **If it exists, skip to Step 1** — never re-create it. It
   exists only if an earlier run was invoked with `--keep-lean`; without that
   flag every invocation starts from scratch, which is the cost you accept for
   not leaving 7.5 GB behind.
3. If it does not exist, create it:

   ```bash
   mkdir -p MATH_VERIFICATION
   cd MATH_VERIFICATION
   # Mathlib template — the default: nearly all paper mathematics needs Mathlib.
   lake new <ProjectName> math
   cd <ProjectName>
   lake exe cache get      # no-op on newer lake (the math template already fetched it)
   lake build              # first build is slow (minutes) even with the cache
   ```

   The `math` template pins `lean-toolchain` to the Mathlib revision it fetches,
   so do not pin a toolchain yourself. On lake >= 5.0 `lake new … math` already
   downloads the ~8700-file Mathlib cache; `lake exe cache get` then costs
   nothing and is safe to keep for older lake.

   Use the bare template (`lake new <ProjectName>`, no `math`) **only** when the
   claim is pure `Nat`/`Int` arithmetic with no Mathlib lemmas needed — it builds
   in seconds but strands you without a library.

4. Record `LEAN_PROJECT` = absolute path to `MATH_VERIFICATION/<ProjectName>`
   and `LEAN_SRC` = `$LEAN_PROJECT/<ProjectName>` (the source directory `lake
   new` creates alongside `Basic.lean`).
5. Confirm the project builds before formalising anything: `lean_build` with
   `lean_project_path = $LEAN_PROJECT`. A project that does not build makes every
   later diagnostic meaningless.

`MATH_VERIFICATION/` is build output plus scratch Lean files: it belongs in
`.gitignore`, never in a commit.

## Step 1 — Write the claim down

Write the natural-language specification you were given — verbatim, plus any
variable definitions and hypotheses the snippet's context supplies — to:

```
$LEAN_PROJECT/claim.md
```

State it as ONE claim. If the snippet contains a chain of derivation steps,
verify the step that carries the mathematical content (usually the one the paper
would be wrong about), and say in `reasoning` which step you formalised.

### Faithfulness gate

Before formalising, write out the hypotheses you intend to assume. The Lean
statement must be a **faithful** rendering of the paper's claim:

- Do not add hypotheses that are not in the paper and that make the claim true
  (e.g. silently assuming `x > 0` when the paper's expression also ranges over
  `x ≤ 0`). A proof of a strengthened statement is not a verification.
- Do not weaken the conclusion (proving `≤` when the paper claims `<`).
- Do carry over genuine standing assumptions the paper states elsewhere.

If a faithful statement is impossible without inventing mathematics, stop and
return `UNVERIFIABLE`.

## Step 2 — Autoformalize

Invoke the lean4 skill to turn `claim.md` into a Lean declaration in `TMP.lean`:

```
/lean4:autoformalize --source $LEAN_PROJECT/claim.md --claim-select=first --out $LEAN_SRC/TMP.lean --commit=never --max-cycles=12
```

- `--commit=never` is **mandatory** — this skill must never create git commits in
  the host repository.
- `--claim-select=first` because `claim.md` holds exactly one claim.
- `autoformalize` already runs a proving loop, so it often returns a finished
  proof. Check the file before doing more work.

## Step 3 — Autoprove any remaining `sorry`

Read `$LEAN_SRC/TMP.lean`. If it still contains `sorry` (or `lean_diagnostic_messages`
reports `declaration uses 'sorry'`), run:

```
/lean4:autoprove $LEAN_SRC/TMP.lean --commit=never --max-cycles=15 --max-stuck-cycles=3
```

While proving, use the `lean-lsp` tools for state rather than guessing:
`lean_goal` for the goal at a position, `lean_multi_attempt` to test several
tactics at once, `lean_leansearch` / `lean_loogle` / `lean_state_search` /
`lean_hammer_premise` to find Mathlib lemmas, `lean_run_code` for throwaway
snippets you do not want in `TMP.lean`.

**If the claim looks false**, do not burn the whole budget on it. Try to prove
the negation instead (`/lean4:disprove`, or a `theorem tmp_false : ¬ (...)` in a
scratch file). A compiling refutation is the strongest possible outcome for a
verifier and maps to `ERROR_DETECTED`.

## Step 4 — Compile and read the verdict

The Lean compiler, not your own reading of the proof, decides. Run both checks:

1. **MCP diagnostics** — `lean_diagnostic_messages` with
   `file_path = $LEAN_SRC/TMP.lean` (absolute). Collect every entry; severity 1
   is an error, severity 2 a warning (`declaration uses 'sorry'` shows here).
2. **Soundness** — `lean_verify` with `file_path = $LEAN_SRC/TMP.lean` and the
   theorem name. It returns the axioms the proof depends on. `sorryAx` in that
   list means the proof is a stub: **this is a FAILED verification**, no matter
   what the diagnostics say. `propext`, `Classical.choice`, and `Quot.sound` are
   the standard Mathlib axioms and are fine.
3. **Confirmation** — from `$LEAN_PROJECT`, `lake env lean <ProjectName>/TMP.lean`.
   Capture stdout/stderr as the compiler feedback either way.

   > **Exit status 0 is NOT sufficient.** `lake env lean` exits 0 on a file whose
   > proof is `sorry` — `sorry` is only a *warning*. It also still prints the
   > `#print axioms` line for a file that failed to compile. Never read either
   > signal on its own: a `PASSED` verdict requires exit 0 **and** no severity-1
   > diagnostic **and** no `sorryAx`.

`lean_status` is then:

| `lean_status` | Condition |
|---|---|
| `PASSED` | compiles clean, no `sorry`, no `sorryAx`, and the statement is faithful |
| `REFUTED` | a faithful formalisation of the **negation** compiles clean |
| `FAILED` | compile errors remain, or `sorry`/`sorryAx` survives the budget |
| `NOT_ATTEMPTED` | ruled out at Step 0/1 (unformalisable, or setup unavailable) |

## Step 5 — Map the Lean verdict to a verification result

This is the part that is easy to get wrong. **Failure to prove is not evidence of
error.** Lean fails on true statements all the time — a missing Mathlib lemma, a
hard analysis argument, a cycle budget hit.

| `lean_status` | `status` | `error_detected` |
|---|---|---|
| `PASSED` | `NO_ERROR` | `false` |
| `REFUTED` | `ERROR_DETECTED` | `true` (confidence ≥ 0.9) |
| `FAILED` | `UNVERIFIABLE` | `false` |
| `NOT_ATTEMPTED` | `UNVERIFIABLE` | `false` |

Only a **compiling refutation** justifies `error_detected = true`. Never flag an
error because the proof search stalled.

## Step 6 — Clean up

**Collect the full result first.** Read `formal_statement`, `compiler_feedback`,
and `axioms` out of the files and into your answer *before* deleting anything —
once the project is gone the evidence is unrecoverable.

Then, depending on the flag:

**Default (no `--keep-lean`) — delete the project:**

```bash
rm -rf MATH_VERIFICATION/<ProjectName>
```

Set `lean.project_kept = false` in the output.

**With `--keep-lean` — keep the project, remove only per-snippet scratch:**

Delete `$LEAN_SRC/TMP.lean`, `$LEAN_PROJECT/claim.md`, and any scratch files you
created, or move the proof to `$LEAN_SRC/Verified/<SnippetId>.lean` to keep it.
Never leave a `TMP.lean` behind for the next run to inherit. Set
`lean.project_kept = true`.

### Deletion rules — non-negotiable

- Delete **only** `MATH_VERIFICATION/<ProjectName>`, the single project directory
  this run used. **Never** delete `MATH_VERIFICATION/` itself, and never touch a
  sibling project directory — another paper's verification may be using it.
- Verify the path before running `rm -rf`: it must be inside `MATH_VERIFICATION/`
  and must be exactly one level deep. If it is not, stop and report instead.
- If Step 0 **reused** an existing project rather than creating it, treat it as
  `--keep-lean` and do not delete it — you did not create it, so it is not yours
  to remove.
- When one invocation verifies several snippets, delete after the **last**
  snippet, not after each one.

## Output Format

Return a structured JSON object:

```json
{
  "snippet_id": "<id>",
  "verifier_name": "math-lean",
  "status": "ERROR_DETECTED | NO_ERROR | UNVERIFIABLE",
  "error_detected": false,
  "confidence": 0.0,
  "reasoning": "Which claim was formalised, the hypotheses assumed, and what the Lean compiler said. If REFUTED, show the refuting statement and why it contradicts the paper.",
  "predicted_error_category": null,
  "snippet_type": "<EQUATION|DERIVATION|FORMULA_BLOCK|PROOF_STEP>",
  "contradiction_locations": [],
  "lean": {
    "lean_status": "PASSED | REFUTED | FAILED | NOT_ATTEMPTED",
    "project": "MATH_VERIFICATION/<ProjectName>",
    "file": "<ProjectName>/TMP.lean",
    "formal_statement": "theorem ... : ... := by ...",
    "compiler_feedback": "verbatim diagnostics / lake output, or '' when clean",
    "axioms": ["propext", "Classical.choice", "Quot.sound"],
    "cycles_used": 0,
    "project_kept": false
  }
}
```

`predicted_error_category` — use when `error_detected` is true, else `null`:

- `"Algebraic manipulation"` — invalid factorization, distribution, or inverse operations
- `"Fractional/Rational error"` — invalid simplification or combination of fractional terms
- `"Sign/Arithmetic error"` — dropped negatives or sign propagation failures
- `"Domain/Constraint violation"` — mathematically impossible bounds (e.g. probability > 1)
- `"Calculus/Proof error"` — invalid limits, derivatives, or integral evaluations
- `"Formally refuted"` — Lean proved the negation and no narrower category fits

## Critical Guidance

- **The compiler is the authority.** Report what it said; never claim a proof
  went through without a clean `lean_diagnostic_messages` **and** a `sorryAx`-free
  `lean_verify`.
- **Always quote `compiler_feedback` verbatim.** It is the evidence for the
  verdict and the handoff for whoever retries.
- **Budget discipline.** Cap the session (`--max-cycles`, `--max-stuck-cycles`);
  a snippet that will not formalise in budget is `FAILED`, and that is a normal,
  reportable outcome — not something to grind on.
- **No commits, no writes outside `MATH_VERIFICATION/`.** The paper's sources are
  read-only to this skill.
- **Clean up after yourself.** A Mathlib project is ~7.5 GB; leaving one per paper
  fills a disk fast. Delete it unless `--keep-lean` says otherwise — and report
  the verdict before you do, never after.
