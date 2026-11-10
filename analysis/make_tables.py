"""Generate markdown tables from results/results.csv.

Usage:
    python -m analysis.make_tables
    python -m analysis.make_tables --input results/results.csv --out results/tables.md
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import statistics
from pathlib import Path

import numpy as np

DEFAULT_INPUT = "results/results.csv"


def _num(v):
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() in ("na", "nan", "none", "null"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def percentile(values: list[float], p: float) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return float(np.percentile(vals, p))  # linear interpolation (numpy default)


def fmt(x, decimals=1):
    if x is None:
        return "—"
    if isinstance(x, float):
        if math.isnan(x):
            return "—"
        if decimals == 0:
            return f"{x:.0f}"
        return f"{x:.{decimals}f}"
    return str(x)


def group_stats(rows: list[dict], key: str | None = None) -> list[tuple[str, list[dict]]]:
    if key is None:
        return [("ALL", rows)]
    groups = {}
    for r in rows:
        g = r.get(key, "") or "(missing)"
        groups.setdefault(g, []).append(r)
    return sorted(groups.items())


def compute_block(rows: list[dict]) -> dict:
    ft = [_num(r.get("first_token_ms")) for r in rows]
    tot = [_num(r.get("total_ms")) for r in rows]
    ft_valid = [v for v in ft if v is not None]
    tot_valid = [v for v in tot if v is not None]
    n = len(rows)
    n_ft_missing = len(ft) - len(ft_valid)
    n_tot_missing = len(tot) - len(tot_valid)
    statuses = [ (r.get("status") or "").strip().lower() for r in rows]
    ok = sum(1 for s in statuses if s == "ok")
    rl = sum(1 for s in statuses if s == "rate_limited")
    to = sum(1 for s in statuses if s == "timeout")
    err = sum(1 for s in statuses if s == "error")
    retries = [float(r.get("retries") or 0) for r in rows]
    tokens = [_num(r.get("tokens_out")) for r in rows]
    tokens_valid = [v for v in tokens if v is not None]
    return {
        "n": n,
        "ft_p50": percentile(ft_valid, 50),
        "ft_p95": percentile(ft_valid, 95),
        "ft_p99": percentile(ft_valid, 99),
        "tot_p50": percentile(tot_valid, 50),
        "tot_p95": percentile(tot_valid, 95),
        "tot_p99": percentile(tot_valid, 99),
        "success_pct": (ok / n * 100) if n else 0.0,
        "rl_pct": (rl / n * 100) if n else 0.0,
        "to_pct": (to / n * 100) if n else 0.0,
        "err_pct": (err / n * 100) if n else 0.0,
        "retries_per_req": (sum(retries) / n) if n else 0.0,
        "tokens_per_answer": (sum(tokens_valid) / len(tokens_valid)) if tokens_valid else None,
        "n_ft_missing": n_ft_missing,
        "n_tot_missing": n_tot_missing,
    }


def make_table(title: str, rows_groups: list[tuple[str, list[dict]]]) -> str:
    lines = []
    lines.append(f"### {title}")
    lines.append("")
    lines.append("| Group | n | p50 FT (ms) | p95 FT (ms) | p99 FT (ms) | p50 Total (ms) | p95 Total (ms) | p99 Total (ms) | Success % | Rate-limited % | Timed-out % | Retries/req | Tokens/answer |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, rows in rows_groups:
        b = compute_block(rows)
        lines.append(
            f"| {name} | {b['n']} | {fmt(b['ft_p50'],0)} | {fmt(b['ft_p95'],0)} | {fmt(b['ft_p99'],0)} | "
            f"{fmt(b['tot_p50'],0)} | {fmt(b['tot_p95'],0)} | {fmt(b['tot_p99'],0)} | "
            f"{fmt(b['success_pct'],1)} | {fmt(b['rl_pct'],1)} | {fmt(b['to_pct'],1)} | "
            f"{fmt(b['retries_per_req'],2)} | {fmt(b['tokens_per_answer'],1)} |"
        )
    lines.append("")
    # missing-value flags
    missing_notes = []
    for name, rows in rows_groups:
        b = compute_block(rows)
        if b["n_ft_missing"] or b["n_tot_missing"]:
            missing_notes.append(f"- {name}: excluded {b['n_ft_missing']} missing first_token_ms, {b['n_tot_missing']} missing total_ms from percentiles.")
    if missing_notes:
        lines.append("**Missing-value flags:**")
        lines.extend(missing_notes)
        lines.append("")
    return "\n".join(lines)


def read_trace_log(trace_glob: str) -> dict:
    """Aggregate peak queue length and cost from trace files if available."""
    files = sorted(glob.glob(trace_glob))
    peak_q = None
    cost = None
    cost_tokens = 0
    n = 0
    for fp in files:
        try:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        n += 1
        # peak queue length
        for key in ("peak_queue_length", "max_queue_length", "peak_queue"):
            if key in data:
                v = data[key]
                if isinstance(v, (int, float)):
                    peak_q = max(peak_q or 0, v)
        # also search steps/decisions
        for step in data.get("steps", []):
            for key in ("peak_queue_length", "max_queue_length", "peak_queue"):
                if key in (step.get("extra") or {}):
                    v = step["extra"][key]
                    if isinstance(v, (int, float)):
                        peak_q = max(peak_q or 0, v)
        # cost: if provided
        for key in ("cost_usd", "cost", "total_cost_usd"):
            if key in data:
                v = data[key]
                if isinstance(v, (int, float)):
                    cost = (cost or 0) + v
        # token-based cost estimate if price provided via env
        for step in data.get("steps", []):
            pt = step.get("prompt_tokens") or 0
            ct = step.get("completion_tokens") or 0
            cost_tokens += pt + ct
    return {
        "files": len(files),
        "peak_queue_length": peak_q,
        "cost_usd": cost,
        "total_tokens": cost_tokens if files else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--out", default=None, help="Write markdown to file instead of stdout")
    parser.add_argument("--trace-glob", default="results/trace_*.json", help="Glob for trace JSON files")
    args = parser.parse_args()

    rows = load_rows(Path(args.input))
    if not rows:
        print("No rows found.")
        return

    # Percentile method note
    note = (
        "**Percentile method:** numpy `np.percentile` with linear interpolation (default). "
        "Rows with missing/empty `first_token_ms` or `total_ms` are excluded from the corresponding "
        "percentile calculation and flagged below.\n"
    )

    out_lines = []
    out_lines.append("# Latency & Reliability Tables\n")
    out_lines.append(note)
    out_lines.append("")

    # Table 2a: overall
    out_lines.append(make_table("Table 2a — Overall", group_stats(rows, None)))

    # Table 2b: per scenario
    out_lines.append(make_table("Table 2b — Per Scenario", group_stats(rows, "scenario")))

    # Table 3: per retrieval setting
    out_lines.append(make_table("Table 3 — Per Retrieval Setting", group_stats(rows, "retrieval_setting")))

    # Peak queue length & cost from trace log
    tr = read_trace_log(args.trace_glob)
    out_lines.append("### Peak Queue Length & Cost (from trace log)")
    out_lines.append("")
    out_lines.append(f"- Trace files matched: {tr['files']} (glob: `{args.trace_glob}`)")
    out_lines.append(f"- Peak queue length: {fmt(tr['peak_queue_length'], 0) if tr['peak_queue_length'] is not None else 'N/A (not recorded in trace)'}")
    if tr["cost_usd"] is not None:
        out_lines.append(f"- Cost (USD): {tr['cost_usd']:.4f}")
    else:
        out_lines.append("- Cost (USD): N/A (not recorded in trace; compute from tokens if pricing available)")
    if tr["total_tokens"] is not None:
        out_lines.append(f"- Total tokens across traces: {tr['total_tokens']}")
    out_lines.append("")
    out_lines.append(
        "_Note: peak queue length and cost are not present in results.csv; they are exported from the "
        "separate trace log when available. If your trace files store these under different keys, "
        "update `read_trace_log` in analysis/make_tables.py._"
    )
    out_lines.append("")

    text = "\n".join(out_lines)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Written: {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
