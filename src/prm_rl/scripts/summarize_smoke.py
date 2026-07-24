"""Collect eval_results.json across arms and print a comparison table.

Usage:
    python -m prm_rl.scripts.summarize_smoke outputs/arm*_smoke/eval_results.json
    python -m prm_rl.scripts.summarize_smoke --out outputs/smoke_summary.md ...

The table columns are the same as the Colab notebook's `results_df`
plus process_correctness when the eval was run with a PRM.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


COLUMNS = [
    "arm",
    "accuracy",
    "process_correctness",
    "avg_tokens",
    "avg_steps",
    "avg_self_rougeL",
    "exploit_rate",
    "trap_solve_rate",
    "CRHS",
]


def _fmt(v, ndigits: int = 3) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, (int,)):
        return str(v)
    if isinstance(v, float):
        return f"{v:.{ndigits}f}"
    return str(v)


def _row_from_results(path: Path) -> dict:
    r = json.loads(path.read_text())
    arm = path.parent.name.replace("_smoke", "").replace("outputs/", "")
    behavior = r.get("behavior", {})
    traps = r.get("traps", {})
    process = r.get("process", {})
    crhs = r.get("crhs", {})
    return {
        "arm": arm,
        "accuracy": r.get("accuracy", {}).get("accuracy"),
        "process_correctness": process.get("process_correctness"),
        "avg_tokens": behavior.get("avg_tokens"),
        "avg_steps": behavior.get("avg_steps"),
        "avg_self_rougeL": behavior.get("avg_self_rougeL"),
        "exploit_rate": traps.get("exploit_rate"),
        "trap_solve_rate": traps.get("trap_solve_rate"),
        "CRHS": crhs.get("CRHS"),
    }


def _md_table(rows: list[dict]) -> str:
    header = "| " + " | ".join(COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in COLUMNS) + " |"
    body_lines = []
    for row in rows:
        body_lines.append(
            "| " + " | ".join(_fmt(row.get(c)) for c in COLUMNS) + " |"
        )
    return "\n".join([header, sep, *body_lines])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results", nargs="+", help="paths to eval_results.json files")
    p.add_argument("--out", default=None, help="write the markdown table here (also always prints to stdout)")
    args = p.parse_args()

    rows: list[dict] = []
    for pathstr in args.results:
        path = Path(pathstr)
        if not path.exists():
            print(f"[skip] {path} does not exist")
            continue
        rows.append(_row_from_results(path))

    if not rows:
        raise SystemExit("no results to summarize")

    rows.sort(key=lambda r: r["arm"])

    table = _md_table(rows)
    print(table)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table + "\n")
        print(f"\nwrote: {out}")


if __name__ == "__main__":
    main()
