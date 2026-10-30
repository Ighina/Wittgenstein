#!/usr/bin/env python3
"""
make_artifacts.py — build the paper-ready figure and tables for the math-verification
system comparison on PaperAudit and SPOT.

Reads (relative to task_specific_output/):
  math-comparison/math_systems{,_strict}_{comparison,per_paper}.csv       PaperAudit
  math-comparison-spot/math_systems{,_strict}_{comparison,per_paper}.csv  SPOT
  math/{paperaudit,spot}-judge-<judge>/*_summary.csv                      judge sensitivity

Writes (into plots_and_tables/):
  fig1_results.pdf / .png          4-panel headline figure
  fig1_results_data.csv            every number plotted in fig1 (the figure's table view)
  table1_main_results.tex / .md    main comparison, both benchmarks
  table1_main_results.csv          same, machine readable, with micro averages
  table2_judge_sensitivity.tex / .md / .csv

Usage:
  python task_specific_output/plots_and_tables/make_artifacts.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
TSO = HERE.parent

# --- design tokens (dataviz skill, light chart surface) ---------------------
INK = "#0b0b0b"          # primary ink
INK_2 = "#52514e"        # secondary ink — also the "baseline system" mark
MUTED = "#898781"        # axis / tick labels
GRID = "#e1e0d9"         # hairline gridline
AXIS = "#c3c2b7"         # baseline / axis
SURFACE = "#fcfcfb"
OURS = "#2a78d6"         # categorical slot 1 — the proposed system
V_TP = "#0ca30c"         # status: good
V_DUP = "#fab219"        # status: warning
V_SPUR = "#d03b3b"       # status: critical
V_OOS = "#c3c2b7"        # neutral: excluded from the in-scope denominator

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.frameon": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# --- benchmarks -------------------------------------------------------------
BENCH = {
    "PaperAudit": dict(dir="math-comparison", ours="orchestrate-math:opus5",
                       backbone="Claude-Opus-5"),
    "SPOT": dict(dir="math-comparison-spot", ours="orchestrate-math:opus5",
                 backbone="Claude-Opus-5"),
}

# display name, row group; fixed order, never value-sorted, so that the figure
# panels, the table rows and the two benchmarks all line up. Baselines prompted
# with the PaperAudit error taxonomy (the "baseline:pa-*" runs still present in
# the source CSVs) are deliberately excluded: every system here is prompted with
# the same default taxonomy, so the comparison is like-for-like.
SYSTEMS = [
    ("orchestrate-math", "Ours", "ours"),
    ("orchestrate-math:opus5", "Ours", "ours"),
    ("baseline:claude-opus-5", "Claude-Opus-5", "ep"),
    ("baseline:claude-sonnet-5", "Claude-Sonnet-5", "ep"),
    ("baseline:deepseek-v4-pro", "DeepSeek-V4-pro", "ep"),
    ("baseline:gpt-5-6-terra", "GPT-5.6-terra", "ep"),
]
NAME = {k: n for k, n, _ in SYSTEMS}
GROUP = {k: g for k, _, g in SYSTEMS}
ORDER = {k: i for i, (k, _, _) in enumerate(SYSTEMS)}


def load(bench):
    """Per-system records for one benchmark, in fixed system order."""
    d = TSO / BENCH[bench]["dir"]
    comp = pd.read_csv(d / "math_systems_comparison.csv").set_index("system")
    strict = pd.read_csv(d / "math_systems_strict_comparison.csv").set_index("system")
    pp = pd.read_csv(d / "math_systems_per_paper.csv")
    ours = BENCH[bench]["ours"]

    # paired Wilcoxon on per-paper F1, ours vs each baseline, Holm-corrected
    piv = pp.pivot_table(index="paper_id", columns="system", values="f1")
    raw = {}
    for s in piv.columns:
        if s == ours or s not in ORDER:
            continue
        sub = piv[[ours, s]].dropna()
        raw[s] = 1.0 if (sub[ours] - sub[s]).eq(0).all() else \
            wilcoxon(sub[ours], sub[s]).pvalue
    holm, m = {}, len(raw)
    for rank, (s, p) in enumerate(sorted(raw.items(), key=lambda kv: kv[1])):
        holm[s] = min(1.0, p * (m - rank))
    for i, (s, _) in enumerate(sorted(raw.items(), key=lambda kv: kv[1])):
        if i:  # Holm is monotone non-decreasing
            holm[s] = max(holm[s], prev)
        prev = holm[s]

    recs = []
    for s in sorted((x for x in comp.index if x in ORDER), key=lambda x: ORDER[x]):
        r = comp.loc[s]
        f1_is = pp.loc[pp.system == s, "f1_in_scope"].mean()
        n_pap = int(r["papers"])
        sig = ""
        if s != ours and holm.get(s, 1.0) < 0.05:
            sig = "*" if r["macro_f1"] < comp.loc[ours, "macro_f1"] else "°"
        recs.append(dict(
            bench=bench, system=s, name=NAME[s], group=GROUP[s], is_ours=(s == ours),
            papers=n_pap, gt=int(r["total_gt"]), pred=int(r["total_pred"]),
            P=r["macro_precision"], R=r["macro_recall"], F1=r["macro_f1"],
            P_is=r["macro_precision_in_scope"], F1_is=f1_is,
            F1_strict=strict.loc[s, "macro_f1"] if s in strict.index else np.nan,
            micro_P=r["micro_precision"], micro_R=r["micro_recall"],
            micro_F1=r["micro_f1"],
            tp=int(r["pred_true_positive"]), dup=int(r["pred_duplicate"]),
            oos=int(r["pred_out_of_scope"]), spur=int(r["pred_spurious"]),
            pred_per_paper=r["total_pred"] / n_pap,
            gt_per_paper=r["total_gt"] / n_pap,
            p_holm=holm.get(s, np.nan), sig=sig,
        ))
    return pd.DataFrame(recs)


DATA = {b: load(b) for b in BENCH}

# ===========================================================================
# Figure 1 — (a,b) precision/recall position, (c,d) how the finding budget is spent
# ===========================================================================
ISO_F1 = [0.20, 0.30, 0.40]
# hand-set label placement (offset in points, h/v anchor, leader line) so that
# no two point labels collide (leader=True draws a connector where a label has
# to sit far from its mark).
LBL = {
    "PaperAudit": {
        "Ours":                   (-8, 0, "right", "center", False),
        "Claude-Sonnet-5":        (8, 1, "left", "center", False),
        "GPT-5.6-terra":          (8, -1, "left", "center", False),
        "DeepSeek-V4-pro":        (0, -9, "center", "top", False),
        "Claude-Opus-5":          (0, 9, "center", "bottom", False),
    },
    "SPOT": {
        "Ours":                   (9, 1, "left", "center", False),
        "GPT-5.6-terra":          (0, -9, "center", "top", False),
        "Claude-Opus-5":          (0, -9, "center", "top", False),
        "DeepSeek-V4-pro":        (-8, 1, "right", "center", False),
        "Claude-Sonnet-5":        (8, 0, "left", "center", False),
    },
}

fig = plt.figure(figsize=(7.1, 5.75))
gs_top = fig.add_gridspec(1, 2, left=0.088, right=0.985, top=0.952, bottom=0.580,
                          wspace=0.16)
gs_bot = fig.add_gridspec(1, 2, left=0.160, right=0.985, top=0.455, bottom=0.145,
                          wspace=0.34)
panel = "abcd"

for col, bench in enumerate(BENCH):
    df = DATA[bench]
    # ---------------- (a)/(b) precision-recall plane ----------------
    ax = fig.add_subplot(gs_top[0, col])
    xs = np.linspace(0.01, 0.95, 400)
    for f in ISO_F1:
        ys = (f * xs) / (2 * xs - f)
        ok = (ys > 0.05) & (ys < 0.5)
        ax.plot(xs[ok], ys[ok], lw=0.7, color=GRID, zorder=1)
        yr = (f * 0.745) / (2 * 0.745 - f)
        ax.text(0.745, yr + 0.004, f"F1={f:.1f}", color=MUTED, fontsize=6.2,
                ha="right", va="bottom", zorder=1)
    for _, r in df.iterrows():
        lab = r["name"]
        if r["is_ours"]:
            ax.scatter(r["R"], r["P"], s=80, marker="o", color=OURS,
                       edgecolor=SURFACE, linewidth=1.4, zorder=5)
        else:
            ax.scatter(r["R"], r["P"], s=22, marker="o", color=INK_2,
                       edgecolor=SURFACE, linewidth=0.9, zorder=6)
        dx, dy, ha, va, leader = LBL[bench][lab]
        ax.annotate(lab, (r["R"], r["P"]), textcoords="offset points",
                    xytext=(dx, dy), fontsize=6.8, zorder=7, ha=ha, va=va,
                    color=OURS if r["is_ours"] else INK_2,
                    fontweight="bold" if r["is_ours"] else "normal",
                    arrowprops=dict(arrowstyle="-", lw=0.6, color=MUTED,
                                    shrinkA=1, shrinkB=4) if leader else None)
    ax.set_xlim(0.09, 0.76)
    ax.set_ylim(0.09, 0.47)
    ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ax.set_yticks([0.1, 0.2, 0.3, 0.4])
    ax.set_xlabel("Recall")
    if col == 0:
        ax.set_ylabel("Precision")
    ax.set_title(f"({panel[col]}) {bench}: accuracy trade-off", loc="left",
                 color=INK, pad=6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if col == 0:
        ax.legend(handles=[
            Line2D([], [], marker="o", ls="", ms=6.5, mfc=OURS, mec=SURFACE,
                   label="Ours (orchestrate-math)"),
            Line2D([], [], marker="o", ls="", ms=4.0, mfc=INK_2, mec=SURFACE,
                   label="Single-call LLM baseline"),
        ], loc="lower left", fontsize=6.6, handletextpad=0.4,
            borderaxespad=0.1, labelcolor=INK_2)

    # ---------------- (c)/(d) finding budget per paper ----------------
    ax = fig.add_subplot(gs_bot[0, col])
    d2 = df.iloc[::-1]                       # top-down = same order everywhere
    y = np.arange(len(d2))
    n = d2["papers"].values
    left = np.zeros(len(d2))
    for key, color in (("tp", V_TP), ("dup", V_DUP), ("spur", V_SPUR), ("oos", V_OOS)):
        w = d2[key].values / n
        ax.barh(y, w, left=left, height=0.64, color=color,
                edgecolor=SURFACE, linewidth=1.0, zorder=3)
        for yi, (wi, li) in enumerate(zip(w, left)):
            if wi >= 0.45:                    # relief rule: label the fills
                ax.text(li + wi / 2, yi, f"{wi:.1f}", ha="center", va="center",
                        fontsize=6.2, zorder=4,
                        color=INK if color in (V_DUP, V_OOS) else "#ffffff")
        left = left + w
    top = len(d2) - 1
    gt_rate = d2["gt_per_paper"].iloc[0]
    ax.axvline(gt_rate, color=INK_2, ls=(0, (3, 2)), lw=0.9, zorder=2)
    ax.annotate(f"{gt_rate:.1f} seeded errors/paper", (gt_rate, top + 0.62),
                xytext=(3, 0), textcoords="offset points", fontsize=6.4,
                color=INK_2, ha="left", va="center")
    for yi, tot in enumerate(left):
        ax.text(tot + 0.07, yi, f"{tot:.1f}", fontsize=6.4, color=INK_2,
                ha="left", va="center")
    labels = list(d2["name"])
    ax.set_yticks(y, labels, fontsize=6.9)
    for tick, is_ours in zip(ax.get_yticklabels(), d2["is_ours"]):
        if is_ours:
            tick.set_color(OURS)
            tick.set_fontweight("bold")
    ax.set_ylim(-0.7, top + 1.0)
    ax.set_xlim(0, left.max() * 1.14)
    ax.set_xlabel("Findings reported per paper")
    ax.set_title(f"({panel[2 + col]}) {bench}: where the findings go", loc="left",
                 color=INK, pad=6)
    ax.xaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)

fig.legend(handles=[
    Patch(fc=V_TP, ec=SURFACE, label="true positive (credited)"),
    Patch(fc=V_DUP, ec=SURFACE, label="duplicate"),
    Patch(fc=V_SPUR, ec=SURFACE, label="spurious (wrong)"),
    Patch(fc=V_OOS, ec=SURFACE, label="out of scope (true, but not a seeded error)"),
], loc="lower center", ncol=4, fontsize=6.9, labelcolor=INK_2,
    handlelength=1.1, handleheight=0.9, columnspacing=1.3, borderaxespad=0.3,
    bbox_to_anchor=(0.5, 0.005))

fig.savefig(HERE / "fig1_results.pdf")
fig.savefig(HERE / "fig1_results.png", dpi=320)
plt.close(fig)

# the figure's table view
cols = ["bench", "name", "group", "papers", "gt", "pred", "P", "R", "F1",
        "tp", "dup", "spur", "oos", "pred_per_paper", "gt_per_paper"]
pd.concat(DATA.values())[cols].to_csv(HERE / "fig1_results_data.csv",
                                      index=False, float_format="%.4f")

# ===========================================================================
# Table 1 — main results
# ===========================================================================
MET = [("P", "P"), ("R", "R"), ("F1", "F1"), ("F1_is", "F1$^{+}$"),
       ("F1_strict", "F1$^{s}$")]


def cell(v, bold=False, sig=""):
    if pd.isna(v):
        return "--"
    s = f"{v:.3f}".lstrip("0")
    if sig:
        s += f"$^{{{sig}}}$"
    return f"\\textbf{{{s}}}" if bold else s


# systems that do not cover the full paper set of their benchmark
_short = []
for b in BENCH:
    full_n = DATA[b]["papers"].max()
    for _, r in DATA[b][DATA[b]["papers"] < full_n].iterrows():
        _short.append(f"{r['name']} on {b} ({int(r['papers'])} of {int(full_n)})")
if not _short:
    COVERAGE_NOTE = ""
elif len(_short) == 1:
    COVERAGE_NOTE = (f" {_short[0]} covers fewer papers than the rest of its column "
                     "and is macro-averaged over the papers it does cover.")
else:
    COVERAGE_NOTE = (" " + ", ".join(_short) + " cover fewer papers than the rest of "
                     "their column and are macro-averaged over the papers they do cover.")

rows, md_rows = [], []
best = {b: {m: DATA[b][m].max() for m, _ in MET} for b in BENCH}
seen_group = None
for key, name, group in SYSTEMS:
    if key in ("orchestrate-math:opus5",):
        continue                      # one "Ours" row, one column per benchmark
    cells, md_cells = [], []
    for b in BENCH:
        df = DATA[b]
        k = BENCH[b]["ours"] if group == "ours" else key
        hit = df[df.system == k]
        for m, _ in MET:
            if hit.empty:
                cells.append("--")
                md_cells.append("–")
                continue
            v = hit.iloc[0][m]
            sig = hit.iloc[0]["sig"] if m == "F1" else ""
            bold = (not pd.isna(v)) and abs(v - best[b][m]) < 1e-12
            cells.append(cell(v, bold, sig))
            md_cells.append(("**" if bold else "") + (f"{v:.3f}".lstrip("0") if not pd.isna(v) else "–")
                            + ("**" if bold else "") + (f"^{sig}" if sig else ""))
    label = "Ours (orchestrate-math)" if group == "ours" else name
    if group != seen_group and seen_group is not None:
        rows.append(("rule", "Single-call LLM baselines"))
        md_rows.append(("rule", "Single-call LLM baselines"))
    seen_group = group
    rows.append(("row", label, cells))
    md_rows.append(("row", label, md_cells))

tex = [
    "% Generated by task_specific_output/plots_and_tables/make_artifacts.py",
    "\\begin{table*}[t]",
    "\\centering\\small",
    "\\setlength{\\tabcolsep}{4.5pt}",
    "\\begin{tabular}{l rrrrr rrrrr}",
    "\\toprule",
    " & \\multicolumn{5}{c}{\\textbf{PaperAudit}} & \\multicolumn{5}{c}{\\textbf{SPOT}} \\\\",
    "\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}",
    "System & " + " & ".join(h for _, h in MET) + " & " +
    " & ".join(h for _, h in MET) + " \\\\",
    "\\midrule",
]
for r in rows:
    if r[0] == "rule":
        tex.append("\\midrule")
    else:
        tex.append(f"{r[1]} & " + " & ".join(r[2]) + " \\\\")
tex += [
    "\\bottomrule",
    "\\end{tabular}",
    "\\caption{\\textbf{Mathematical-error detection on PaperAudit and SPOT.} "
    "All scores are macro-averages over papers (PaperAudit: "
    f"{int(DATA['PaperAudit']['papers'].max())} papers / {int(DATA['PaperAudit']['gt'].max())} seeded errors; SPOT: "
    f"{int(DATA['SPOT']['papers'].max())} papers / {int(DATA['SPOT']['gt'].max())} errors), "
    "aligned to ground truth by the same LLM judge (Claude-Sonnet-5) for every system. "
    "$P$/$R$/F1 count every reported finding; F1$^{+}$ removes findings the judge "
    "ruled \\emph{out of scope} (true statements about the paper that are not seeded "
    "errors) from the precision denominator; F1$^{s}$ is the strict regime, in which a "
    "baseline finding counts only if the baseline itself labelled it \\emph{Equation / "
    "proof}. All systems are prompted with the same taxonomy. Best per column in "
    "bold. $^{*}$: ours is better at $p<0.05$ (paired Wilcoxon on per-paper F1, "
    "Holm-corrected within benchmark); no baseline is significantly better than ours. "
    "Ours runs the same Claude-Opus-5 backbone on both benchmarks, so no system "
    "in the table is advantaged by a stronger underlying model."
    + COVERAGE_NOTE + "}",
    "\\label{tab:main}",
    "\\end{table*}",
]
(HERE / "table1_main_results.tex").write_text("\n".join(tex) + "\n")

md = ["# Table 1 — Mathematical-error detection on PaperAudit and SPOT", "",
      "| System | " + " | ".join(f"PA {h}" for _, h in MET) + " | " +
      " | ".join(f"SPOT {h}" for _, h in MET) + " |",
      "|" + "---|" * 11]
for r in md_rows:
    if r[0] == "rule":
        md.append("| *— single-call LLM baselines —* |" + " |" * 10)
    else:
        md.append(f"| {r[1]} | " + " | ".join(r[2]) + " |")
md += ["",
       "`F1+` = out-of-scope findings removed from the precision denominator; "
       "`F1s` = strict regime (baseline must itself label the finding "
       "*Equation / proof*). All systems use the same default taxonomy. "
       "Macro-averages over papers; one shared LLM judge "
       "(Claude-Sonnet-5). `*` = ours better at p<0.05 (paired Wilcoxon on "
       "per-paper F1, Holm-corrected).",
       ""]
(HERE / "table1_main_results.md").write_text("\n".join(md).replace("$^{+}$", "+")
                                             .replace("$^{s}$", "s") + "\n")

full = pd.concat(DATA.values())[[
    "bench", "name", "group", "papers", "gt", "pred", "P", "R", "F1", "P_is",
    "F1_is", "F1_strict", "micro_P", "micro_R", "micro_F1", "tp", "dup", "oos",
    "spur", "p_holm"]]
full.to_csv(HERE / "table1_main_results.csv", index=False, float_format="%.4f")

# ===========================================================================
# Table 2 — judge sensitivity
# ===========================================================================
JUDGES = [("claude-opus-5", "Claude-Opus-5"), ("claude-sonnet-5", "Claude-Sonnet-5"),
          ("deepseek-v4-pro", "DeepSeek-V4-pro"), ("gpt-5.6-terra", "GPT-5.6-terra")]
jrec = []
for slug, disp in JUDGES:
    row = {"judge": disp}
    for b, pre in (("PaperAudit", "paperaudit"), ("SPOT", "spot")):
        f = next((TSO / "math" / f"{pre}-judge-{slug}").glob("*_summary.csv"))
        s = pd.read_csv(f).iloc[0]
        row[f"{b}_papers"] = int(s["total_papers"])
        row[f"{b}_P"] = s["macro_precision"]
        row[f"{b}_R"] = s["macro_recall"]
        row[f"{b}_F1"] = s["macro_f1"]
    jrec.append(row)
J = pd.DataFrame(jrec)
J.to_csv(HERE / "table2_judge_sensitivity.csv", index=False, float_format="%.4f")

jcols = [f"{b}_{m}" for b in BENCH for m in ("P", "R", "F1")]
t2 = [
    "% Generated by task_specific_output/plots_and_tables/make_artifacts.py",
    "\\begin{table}[t]",
    "\\centering\\small",
    "\\setlength{\\tabcolsep}{3.5pt}",
    "\\begin{tabular}{l rrr rrr}",
    "\\toprule",
    "& \\multicolumn{3}{c}{\\textbf{PaperAudit}} & \\multicolumn{3}{c}{\\textbf{SPOT}} \\\\",
    "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
    "Judge & P & R & F1 & P & R & F1 \\\\",
    "\\midrule",
]
for _, r in J.iterrows():
    t2.append(r["judge"] + " & " + " & ".join(f"{r[c]:.3f}".lstrip("0") for c in jcols) + " \\\\")
t2.append("\\midrule")
t2.append("spread (max $-$ min) & " +
          " & ".join(f"{J[c].max() - J[c].min():.3f}".lstrip("0") for c in jcols) + " \\\\")
t2 += [
    "\\bottomrule",
    "\\end{tabular}",
    "\\caption{\\textbf{Judge sensitivity.} The same output of our pipeline "
    "(Claude-Opus-5 backbone), re-aligned to ground truth by four different "
    "LLM judges, over the full paper sets "
    f"(PaperAudit: {int(J['PaperAudit_papers'].iloc[0])} papers / 94 seeded errors; "
    f"SPOT: {int(J['SPOT_papers'].iloc[0])} papers / 37 errors). Recall moves by at "
    f"most {J['PaperAudit_R'].max() - J['PaperAudit_R'].min():.3f} (PaperAudit) and "
    f"{J['SPOT_R'].max() - J['SPOT_R'].min():.3f} (SPOT) across judges, so absolute "
    "scores carry a judge-dependent offset of that order --- comparable to several "
    "of the system-to-system gaps in Table~\\ref{tab:main}, which controls for it "
    "by scoring every system with one fixed judge (Claude-Sonnet-5, the second row "
    "here). This table scores the same papers and the same pipeline output as the "
    "\\emph{Ours} row of Table~\\ref{tab:main}, but through the task-specific "
    "verifier scorer rather than the system-comparison scorer, so its absolute "
    "levels are not expected to coincide; the four rows are mutually consistent, "
    "which is what the spread measures.}",
    "\\label{tab:judge}",
    "\\end{table}",
]
(HERE / "table2_judge_sensitivity.tex").write_text("\n".join(t2) + "\n")

m2 = ["# Table 2 — Judge sensitivity (our pipeline, Claude-Opus-5 backbone)", "",
      "| Judge | PA P | PA R | PA F1 | SPOT P | SPOT R | SPOT F1 |",
      "|---|---|---|---|---|---|---|"]
for _, r in J.iterrows():
    m2.append(f"| {r['judge']} | " + " | ".join(f"{r[c]:.3f}".lstrip("0") for c in jcols) + " |")
m2.append("| **spread (max - min)** | " +
          " | ".join(f"{J[c].max() - J[c].min():.3f}".lstrip("0") for c in jcols) + " |")
m2 += ["", "Full paper sets: PaperAudit 46 papers / 94 seeded errors, SPOT 37 / 37.", ""]
(HERE / "table2_judge_sensitivity.md").write_text("\n".join(m2) + "\n")

print("wrote:", *sorted(p.name for p in HERE.iterdir() if p.suffix in
                        {".pdf", ".png", ".tex", ".md", ".csv"}), sep="\n  ")
