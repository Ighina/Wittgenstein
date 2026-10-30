#!/usr/bin/env python3
"""
compare_llms.py — Run the single-call baseline (scripts/run_baseline.py) across
several LLMs so their error-detection quality and cost can be compared.

Models may come from any provider supported by scripts/llm_providers.py —
Anthropic, OpenAI, Google, or DeepSeek — mixed freely in one run. The provider
is inferred from the model name, or stated as `provider:model`. Each provider's
API key must be in the environment or the project .env file; keys are checked
up front so a long run doesn't die halfway through.

Each model gets its own report tag, so runs sit side by side in the workspace:

    verification-report-<tag-prefix>-<model-slug>.json

Score them all with:

    python scripts/evaluate_verifier.py --llm <name> --evaluate-llms \\
        --workspace-dir <workspace>

Usage:
  # cross-provider comparison
  python scripts/compare_llms.py --workspace-dir data/paperaudit-workspace \\
      --models claude-opus-5,gpt-5.6-terra,gemini-3.5-flash,deepseek-v4-pro

  # one provider's tiers
  python scripts/compare_llms.py --workspace-dir data/paperaudit-workspace \\
      --models claude-opus-5,claude-sonnet-5,claude-haiku-4-5

  # different taxonomy, distinct tag namespace
  python scripts/compare_llms.py --workspace-dir data/paperaudit-workspace \\
      --models claude-opus-5,gpt-5.6-luna \\
      --taxonomy configs/taxonomies/paperaudit.json --tag-prefix baseline-pa
"""

import argparse
import sys
from pathlib import Path
from typing import List

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from llm_providers import (  # noqa: E402
    api_key_for,
    get_client,
    missing_key_message,
    resolve_provider,
)
from model_costs import load_pricing, model_slug, price_for  # noqa: E402
from run_baseline import (  # noqa: E402
    BASELINE_MAX_TOKENS,
    DEFAULT_TAXONOMY,
    load_taxonomy,
    run_baseline_batch,
)

DEFAULT_MODELS = ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"]


def parse_models(values: List[str]) -> List[str]:
    """Accept repeated --models flags and/or comma-separated lists."""
    models: List[str] = []
    for value in values:
        for m in value.split(","):
            m = m.strip()
            if m and m not in models:
                models.append(m)
    return models


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the single-call baseline across several LLMs for comparison.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--workspace-dir",
        type=str,
        required=True,
        help="Workspace with prepared papers (each dir has paper.md + ground_truth.json)",
    )
    parser.add_argument(
        "--models",
        action="append",
        default=None,
        help="Models to compare — repeat the flag or pass a comma-separated list. "
        "Any provider: claude-*, gpt-*/o3, gemini-*, deepseek-*, or an explicit "
        f"provider:model (default: {','.join(DEFAULT_MODELS)})",
    )
    parser.add_argument(
        "--taxonomy",
        type=str,
        default=str(DEFAULT_TAXONOMY),
        help=f"Path to taxonomy JSON (default: {DEFAULT_TAXONOMY})",
    )
    parser.add_argument(
        "--tag-prefix",
        type=str,
        default="baseline",
        help="Tag prefix; each model writes verification-report-<prefix>-<model-slug>.json "
        "(default: baseline)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max papers per model")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=BASELINE_MAX_TOKENS,
        help=f"Output token budget per call (default: {BASELINE_MAX_TOKENS})",
    )
    parser.add_argument(
        "--concurrency", type=int, default=8, help="Parallel calls (default: 8)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run papers that already have a report for this model's tag",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Override the API key for every provider in this run. Normally omit "
        "it and let each provider read its own env var / .env entry.",
    )
    args = parser.parse_args()

    ws_base = Path(args.workspace_dir)
    if not ws_base.exists():
        print(f"ERROR: workspace not found: {ws_base}")
        sys.exit(1)

    models = parse_models(args.models) if args.models else list(DEFAULT_MODELS)
    taxonomy_path = Path(args.taxonomy)
    taxonomy = load_taxonomy(taxonomy_path)  # fail fast before any API call

    pricing = load_pricing()

    # Resolve providers up front so an unknown model name fails before any call.
    try:
        providers = {m: resolve_provider(m)[0] for m in models}
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    unpriced = [m for m in models if price_for(m, pricing) is None]

    print(f"{'=' * 70}")
    print(f"LLM COMPARISON — {len(models)} models")
    print(f"  Workspace: {ws_base}")
    print(f"  Taxonomy:  {taxonomy.get('name')} ({len(taxonomy['categories'])} categories)")
    for m in models:
        print(f"    - {m}  [{providers[m]}]")
    if unpriced:
        print(
            f"  NOTE: no price configured for {', '.join(unpriced)} — "
            f"cost will be reported as unknown. Add them to configs/model_pricing.json."
        )
    print(f"{'=' * 70}\n")

    # Preflight every provider's credentials so a long multi-model run doesn't
    # die halfway through on a missing key.
    needed = sorted(set(providers.values()))
    missing = [p for p in needed if not api_key_for(p, args.api_key)]
    if missing:
        print("ERROR: missing API keys for the requested models:")
        for p in missing:
            models_for_p = [m for m in models if providers[m] == p]
            print(f"  {p} (needed by {', '.join(models_for_p)}): {missing_key_message(p)}")
        sys.exit(1)

    # One client per provider, reused across that provider's models.
    clients = {}
    for p in needed:
        try:
            clients[p] = get_client(p, args.api_key)
        except RuntimeError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

    results = []
    for i, model in enumerate(models, 1):
        provider = providers[model]
        tag = f"{args.tag_prefix}-{model_slug(model)}"
        print(f"\n[{i}/{len(models)}] === {model} ({provider}) (tag={tag}) ===")
        try:
            summary = run_baseline_batch(
                workspace_dir=ws_base,
                model=model,
                taxonomy_path=taxonomy_path,
                tag=tag,
                limit=args.limit,
                concurrency=args.concurrency,
                overwrite=args.overwrite,
                client=clients[provider],
                max_tokens=args.max_tokens,
            )
        except Exception as e:
            # One provider failing shouldn't discard the others' completed runs.
            print(f"  ERROR running {model}: {type(e).__name__}: {e}")
            summary = {"total": 0, "ok": 0, "failed": 0, "error": str(e)}
        results.append({"model": model, "provider": provider, "tag": tag, **summary})

    print(f"\n{'=' * 70}")
    print("BASELINE RUNS COMPLETE")
    for r in results:
        status = f"{r['ok']}/{r['total']} papers OK"
        if r.get("error"):
            status = f"ERROR: {r['error'][:60]}"
        print(f"  {r['model']:<26} [{r['provider']:<9}] tag={r['tag']:<32} {status}")
    print(f"{'=' * 70}")
    print(
        "\nScore and compare (metrics + cost) with:\n"
        f"  python scripts/evaluate_verifier.py --llm <name> --evaluate-llms "
        f"--workspace-dir {ws_base}"
    )


if __name__ == "__main__":
    main()
