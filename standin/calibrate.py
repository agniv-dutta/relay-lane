"""Calibrate stand-in config from results/probe_raw.jsonl."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

DEFAULT_INPUT = "results/probe_raw.jsonl"
DEFAULT_OUTPUT = "standin/config.json"
DEFAULT_CHUNKS = "data/chunks/chunks_400.jsonl"


def load_records(path: Path) -> list[dict]:
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    return recs


def fit_lognormal(values: list[float], default_mu: float = 0.0, default_sigma: float = 1.0) -> dict:
    vals = [v for v in values if v is not None and v > 0]
    if len(vals) < 5:
        return {"dist": "lognormal", "mu": default_mu, "sigma": default_sigma, "n": len(vals)}
    arr = np.array(vals, dtype=float)
    # scipy lognorm fit with floc=0
    s, loc, scale = stats.lognorm.fit(arr, floc=0)
    mu = math.log(scale)
    sigma = s
    # KS goodness-of-fit
    ks_stat, ks_p = stats.kstest(arr, "lognorm", args=(sigma, 0, scale))
    return {
        "dist": "lognormal",
        "mu": round(float(mu), 4),
        "sigma": round(float(sigma), 4),
        "n": len(vals),
        "median": round(float(np.median(arr)), 4),
        "mean": round(float(np.mean(arr)), 4),
        "ks_stat": round(float(ks_stat), 4),
        "ks_pvalue": round(float(ks_p), 4),
    }


def most_common_header(records: list[dict], header: str) -> int | None:
    vals = []
    for r in records:
        h = r.get("rate_limit_headers", {}) or {}
        # headers stored lowercased
        for k, v in h.items():
            if k.lower() == header.lower():
                try:
                    vals.append(int(float(v)))
                except (ValueError, TypeError):
                    pass
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]


def infer_limits(records: list[dict]) -> dict:
    rpm = most_common_header(records, "x-ratelimit-limit-requests")
    tpm = most_common_header(records, "x-ratelimit-limit-tokens")
    # RPD: try daily variant
    rpd = most_common_header(records, "x-ratelimit-limit-requests-daily")
    if rpd is None:
        rpd = 0
    # max concurrency from concurrent ramp
    max_conc = None
    by_level: dict[int, list[dict]] = {}
    for r in records:
        if r.get("mode") == "concurrent" and r.get("concurrency_level"):
            by_level.setdefault(r["concurrency_level"], []).append(r)
    # highest level where all success
    for level in sorted(by_level.keys(), reverse=True):
        recs = by_level[level]
        if recs and all(r.get("status") == 200 for r in recs):
            max_conc = level
            break
    if max_conc is None:
        max_conc = 4
    return {
        "rpm": rpm if rpm is not None else 30,
        "tpm": tpm if tpm is not None else 6000,
        "rpd": rpd,
        "max_concurrency": max_conc,
        "_observed": {
            "rpm_header": rpm is not None,
            "tpm_header": tpm is not None,
            "rpd_header": most_common_header(records, "x-ratelimit-limit-requests-daily") is not None,
            "max_conc_from_ramp": max_conc,
        },
    }


def infer_load_slowdown(records: list[dict]) -> dict:
    # group concurrent records by level, compute median ttft
    by_level: dict[int, list[float]] = {}
    for r in records:
        if r.get("mode") == "concurrent" and r.get("concurrency_level") and r.get("ttft_s") is not None:
            by_level.setdefault(r["concurrency_level"], []).append(r["ttft_s"])
    if 1 not in by_level or len(by_level) < 2:
        return {"enabled": True, "factor_per_request": 0.15, "cap": 5.0, "_note": "insufficient concurrent data"}
    base = float(np.median(by_level[1]))
    factors = []
    for level, vals in by_level.items():
        if level <= 1:
            continue
        med = float(np.median(vals))
        if base > 0 and level > 1:
            alpha = (med / base - 1.0) / (level - 1)
            factors.append(alpha)
    if not factors:
        return {"enabled": True, "factor_per_request": 0.15, "cap": 5.0, "_note": "no higher levels"}
    alpha = float(np.median(factors))
    alpha = max(0.0, min(alpha, 2.0))
    return {"enabled": True, "factor_per_request": round(alpha, 4), "cap": 5.0}


def infer_failures(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {"error_rate": 0.02, "hang_rate": 0.01, "hang_seconds": [5.0, 30.0],
                "slow_success_rate": 0.05, "slow_success_extra_ms": [1000.0, 5000.0]}
    errors = sum(1 for r in records if (r.get("status") or 0) >= 500)
    # hangs: requests with very large total time (top 1%) 
    totals = sorted(r.get("total_s") or 0 for r in records)
    threshold = totals[int(0.99 * (len(totals) - 1))] if totals else 0
    hangs = sum(1 for r in records if (r.get("total_s") or 0) >= max(threshold, 1) and (r.get("status") or 0) == 200)
    error_rate = errors / n
    hang_rate = hangs / n
    return {
        "error_rate": round(max(0.0, min(error_rate, 0.5)), 4),
        "hang_rate": round(max(0.0, min(hang_rate, 0.2)), 4),
        "hang_seconds": [5.0, 30.0],
        "slow_success_rate": 0.05,
        "slow_success_extra_ms": [1000.0, 5000.0],
        "_observed": {"n": n, "errors": errors, "hangs_estimated": hangs},
    }


def limitations_report(records, cfg, ttft_fit, pt_fit, limits) -> list[str]:
    out = []
    out.append("Stand-in limitations (least like real Groq):")
    out.append("1. Latency model treats TTFT and per-token delay as independent lognormals; real latency likely correlates with prompt length and load.")
    out.append("2. No network jitter, TCP/TLS handshake, or cold-start effects.")
    out.append("3. Rate limiting uses simple sliding windows; real Groq uses token buckets with burst capacity and per-model windows.")
    out.append(f"4. RPD {'observed from headers' if limits.get('_observed', {}).get('rpd_header') else 'NOT observed; set to ' + str(limits.get('rpd', 0)) + ' (default/estimated)'}")
    out.append("5. Failure injection is i.i.d. Bernoulli; real errors are bursty and correlated with load.")
    out.append("6. Hang/slow-success rates are estimated crudely (top-1% latency); not directly observable from probe.")
    out.append("7. Canned reply content is fixed and does not reflect model variability or prompt-dependent length.")
    out.append("8. Load slowdown assumed linear in in-flight count; real scheduling may be nonlinear.")
    out.append("9. No prompt-dependent output length; completion length fixed.")
    out.append("10. No streaming backpressure or client-disconnect handling.")
    out.append("11. Lognormal fit may underfit tails; check KS p-values in config.")
    out.append("12. Stand-in does not simulate regional routing, model warm-up, or version changes.")
    if ttft_fit.get("ks_pvalue", 1) < 0.05:
        out.append(f"13. TTFT lognormal fit rejected by KS test (p={ttft_fit['ks_pvalue']}); consider another distribution.")
    if pt_fit.get("ks_pvalue", 1) < 0.05:
        out.append(f"14. Per-token lognormal fit rejected by KS test (p={pt_fit['ks_pvalue']}); consider another distribution.")
    if not records:
        out.append("15. No probe data provided; config uses defaults.")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--chunks", default=DEFAULT_CHUNKS)
    args = parser.parse_args()

    input_path = Path(args.input)
    records = load_records(input_path) if input_path.exists() else []

    # TTFT fit
    ttfts = [r.get("ttft_s") for r in records if r.get("ttft_s") is not None]
    ttft_fit = fit_lognormal(ttfts, default_mu=-1.6, default_sigma=0.5)

    # per-token delay: (total - ttft) / completion_tokens
    per_tokens = []
    for r in records:
        if r.get("ttft_s") is None or r.get("total_s") is None:
            continue
        usage = r.get("usage") or {}
        ct = usage.get("completion_tokens")
        if not ct:
            # estimate from completion preview length
            preview = r.get("completion_preview") or ""
            ct = max(1, len(preview.split()))
        delay = (r["total_s"] - r["ttft_s"]) / max(1, ct)
        if delay > 0:
            per_tokens.append(delay)
    pt_fit = fit_lognormal(per_tokens, default_mu=-5.0, default_sigma=0.3)

    limits = infer_limits(records)
    slowdown = infer_load_slowdown(records)
    failures = infer_failures(records)

    cfg = {
        "latency": {
            "ttft": {"dist": "lognormal", "mu": ttft_fit["mu"], "sigma": ttft_fit["sigma"]},
            "per_token": {"dist": "lognormal", "mu": pt_fit["mu"], "sigma": pt_fit["sigma"]},
            "load_slowdown": slowdown,
        },
        "limits": {
            "rpm": limits["rpm"],
            "tpm": limits["tpm"],
            "rpd": limits["rpd"],
            "max_concurrency": limits["max_concurrency"],
        },
        "failures": {
            "error_rate": failures["error_rate"],
            "hang_rate": failures["hang_rate"],
            "hang_seconds": failures["hang_seconds"],
            "slow_success_rate": failures["slow_success_rate"],
            "slow_success_extra_ms": failures["slow_success_extra_ms"],
        },
        "content": {
            "chunks_path": args.chunks,
            "max_citations": 3,
        },
        "_fit": {
            "ttft": ttft_fit,
            "per_token": pt_fit,
            "n_records": len(records),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"Wrote config: {out_path}")

    # limitations report
    report = limitations_report(records, cfg, ttft_fit, pt_fit, limits)
    report_path = out_path.with_name("calibration_report.txt")
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote report: {report_path}")
    for line in report:
        print(line)


if __name__ == "__main__":
    main()
