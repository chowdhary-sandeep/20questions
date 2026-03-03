#!/usr/bin/env python3
"""
Summarize vf-eval results for balrog-prime from outputs/evals/**/results.jsonl.

- Prints per-env summaries (episodes, avg reward, and rubric components)
- Prints a combined summary across all found runs
- Robust to missing rubric fields

Usage (from prime-environments/environments/balrog_prime):
  uv run python scripts/summarize_results.py --root outputs/evals
  # optionally save to file:
  uv run python scripts/summarize_results.py --root outputs/evals --save outputs/summaries/latest.txt
"""
import argparse
import glob
import json
import os
import re
import sys
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="outputs/evals",
        help="Root directory under which to search for results.jsonl files (default: outputs/evals)",
    )
    ap.add_argument(
        "--save",
        default=None,
        help="Optional path to save the printed summary to a file, in addition to stdout",
    )
    return ap.parse_args()


def env_from_row(row: Dict[str, Any]) -> str:
    # Prefer the adapter's 'task' field, e.g. 'balrog-prime::babyai'
    t = row.get("task") or ""
    m = re.search(r"balrog-prime::([A-Za-z0-9_\-/.]+)", t)
    if m:
        return m.group(1)
    # Fallback to info.env_name if present
    info = row.get("info") or {}
    env = info.get("env_name") or info.get("env")
    if env:
        return str(env)
    return "unknown"


def mean(total: float, n: int) -> float:
    return (total / max(1, n)) if n else 0.0


def summarize(paths: List[str]) -> str:
    # stats template
    def new_stats():
        return {
            "n": 0,
            "reward": 0.0,
            "success_reward": 0.0,
            "progress_reward": 0.0,
            "efficiency_reward": 0.0,
            "format_reward": 0.0,
        }

    per: Dict[str, Dict[str, Any]] = {}
    tot: Dict[str, Any] = new_stats()

    # accumulate
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                for ln in f:
                    row = json.loads(ln)
                    env = env_from_row(row)
                    if env not in per:
                        per[env] = new_stats()
                    # reward
                    r = float(row.get("reward", 0.0) or 0.0)
                    per[env]["n"] += 1
                    per[env]["reward"] += r
                    tot["n"] += 1
                    tot["reward"] += r
                    # rubric components are optional
                    rc = row.get("rubric_components") or {}
                    for k in ("success_reward", "progress_reward", "efficiency_reward", "format_reward"):
                        v = float(rc.get(k, 0.0) or 0.0)
                        per[env][k] += v
                        tot[k] += v
        except Exception as e:
            print(f"[warn] Failed to read {p}: {e}", file=sys.stderr)

    # render
    lines: List[str] = []
    lines.append("=== Per-env Summaries ===")
    for env in sorted(per.keys()):
        s = per[env]
        n = int(s["n"])
        lines.append(f"--- {env} ---")
        lines.append(f"episodes: {n}")
        lines.append(f"avg reward: {mean(s['reward'], n):.4f}")
        lines.append(f"success_reward: {mean(s['success_reward'], n):.4f}")
        lines.append(f"progress_reward: {mean(s['progress_reward'], n):.4f}")
        lines.append(f"efficiency_reward: {mean(s['efficiency_reward'], n):.4f}")
        lines.append(f"format_reward: {mean(s['format_reward'], n):.4f}")
        lines.append("")

    lines.append("=== Combined Summary ===")
    n = int(tot["n"])
    lines.append(f"episodes: {n}")
    lines.append(f"avg reward: {mean(tot['reward'], n):.4f}")
    lines.append(f"success_reward: {mean(tot['success_reward'], n):.4f}")
    lines.append(f"progress_reward: {mean(tot['progress_reward'], n):.4f}")
    lines.append(f"efficiency_reward: {mean(tot['efficiency_reward'], n):.4f}")
    lines.append(f"format_reward: {mean(tot['format_reward'], n):.4f}")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    pattern = os.path.join(args.root, "**", "results.jsonl")
    paths = glob.glob(pattern, recursive=True)
    if not paths:
        print(f"No results.jsonl files found under {args.root}", file=sys.stderr)
        sys.exit(1)

    out = summarize(paths)
    print(out)
    if args.save:
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        with open(args.save, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"[info] Summary saved to {args.save}", file=sys.stderr)


if __name__ == "__main__":
    main()
