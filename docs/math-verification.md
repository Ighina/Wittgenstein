# Mathematical verification — setup and layout

The math skills (`/verify-math`, `/verify-math-python`, `/verify-math-lean`,
`/verify-math-hybrid`, `/math-routing`, and the `/orchestrate-math*`
orchestrators) reference this file for environment setup. Everything below is
one-time, per machine.

## What needs installing

### 1. Lean toolchain

Needed by `/verify-math-lean` and the Lean stage of `/verify-math-hybrid`.

```bash
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y
export PATH="$HOME/.elan/bin:$PATH"
elan toolchain install leanprover/lean4:stable
elan default leanprover/lean4:stable
```

Check: `lake --version` and `lean --version` both answer.

Do not pin a toolchain by hand. `lake new <Name> math` writes a
`lean-toolchain` matching the Mathlib revision it fetches, and overriding it is
the fastest way to a project that will not build.

### 2. `lean-lsp` MCP server

This is how the skills read goals, diagnostics and axioms — they never parse
compiler output by eye.

```bash
claude mcp add lean-lsp -s project uvx lean-lsp-mcp
```

Check: `claude mcp list` shows `lean-lsp`. It exposes `lean_build`,
`lean_diagnostic_messages`, `lean_goal`, `lean_verify`, `lean_run_code`,
`lean_multi_attempt`, `lean_local_search`, `lean_loogle`, `lean_leansearch`,
`lean_hammer_premise` and others. File-taking tools want an **absolute**
`file_path`; `lean_build` takes `lean_project_path`.

### 3. `lean4` skills plugin

Provides `/lean4:autoformalize` and `/lean4:autoprove`, which
`/verify-math-lean` drives.

```
/plugin marketplace add cameronfreer/lean4-skills
/plugin install lean4
```

Non-interactive: `claude plugin marketplace add cameronfreer/lean4-skills` then
`claude plugin install lean4@lean4-skills`.

### 4. Python

`/verify-math-python` and the Python stage of `/verify-math-hybrid` need
`sympy`, `numpy`, `scipy` and `mpmath`:

```bash
pip install sympy numpy scipy mpmath
```

## The `MATH_VERIFICATION/` directory

Lean projects live at the repo root under `MATH_VERIFICATION/<ProjectName>`. A
Mathlib-backed project is **~7.4 GB** of build output. The directory is
gitignored and must never be committed.

### `MATH_VERIFICATION/Paper` is shared

Project names are derived from the paper filename, and every paper in
`data/spot-workspace/` and `data/paperaudit-workspace/` is `paper.md`. So they
all map to the same project: **`MATH_VERIFICATION/Paper`**.

That makes it shared state across a whole batch run, with two consequences:

- **Pass `--keep-lean` to every Lean-touching call**, including the last one of a
  run. `/orchestrate-math-v3` does this unconditionally for exactly this reason.
  A final call without it deletes the project and charges the next paper a full
  Mathlib download and build.
- **Never `rm -rf MATH_VERIFICATION/Paper`** as cleanup, and never delete
  `MATH_VERIFICATION/` itself. A sibling project may belong to another run.

If the project is gone, the next `/verify-math-lean` recreates it — correctly,
but at the cost of a multi-thousand-file cache fetch plus a build of several
minutes.

## Smoke test

Confirm the project builds and Mathlib is reachable before trusting any verdict:

```bash
cd MATH_VERIFICATION/Paper && lake build
```

then compile a two-line file that uses Mathlib:

```lean
import Mathlib
example (a b : ℕ) : a + b = b + a := by ring
```

If that works, the environment is fine.

## Troubleshooting

**A failed `import` is usually not an unbuilt project.** Mathlib renames and
splits modules between releases — `Mathlib.MeasureTheory.Integral.Bochner`
became a *directory* whose contents moved to `…Bochner.Basic` and siblings, so
the old path errors while Mathlib is perfectly built. Run the smoke test above
first. If it passes, the project is fine and the import path is stale: find the
real one with `lean_local_search`, `lean_leansearch` or `lean_loogle`.

Reporting "Mathlib is not built" when it is sends the next reader through a
7.5 GB rebuild that changes nothing.

**Exit status 0 does not mean the proof went through.** `lake env lean` exits 0
on a file whose proof is `sorry`, because `sorry` is only a warning. A `PASSED`
verdict needs exit 0 **and** no severity-1 diagnostic **and** no `sorryAx` in
`lean_verify`'s axiom list.

**A `FAILED` is not evidence of an error.** Lean fails on true statements
constantly — a missing lemma, a hard analysis step, a spent cycle budget. Only a
compiling proof of the *negation* (`REFUTED`) is formal evidence that a paper's
claim is wrong.

## Where reports land

| Producer | Artefact |
|---|---|
| `scripts/run_math_detector.py` | `<paper_dir>/math-detector-<tag>.json` |
| `/math-routing` | `<paper_dir>/math-routing-<tag>.json` |
| `/orchestrate-math*` | `<paper_dir>/<paper_name>_verified_math[-<tag>].json` |
| `scripts/evaluate_math_systems.py` | `task_specific_output/math-comparison*/` |
| `scripts/evaluate_math_oracle.py` | `task_specific_output/math-oracle-spot/` |

Batch driver: `scripts/batch-verify-math.sh --skill <skill> [--paper-list FILE]`.
