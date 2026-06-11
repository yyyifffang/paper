#!/usr/bin/env python3
"""
Post-hoc Macro F1 + Tail F1 plateau analysis for Banking77 experiment logs.

Purpose:
- Parse existing experiment logs without rerunning LLaMA/Qwen.
- Recompute plateau stopping points with a patience-style rule:
    ΔMacroF1_t < delta AND ΔTailF1_t < delta for k consecutive iterations.
- Export iteration-level metrics, per-run stopping summary, and sensitivity summary.

Important limitation:
- Current logs print per-iteration Head/Tail F1 only for Generate-only and Generate-and-Verify.
- Passive/Active logs print only Test F1 and Accuracy per iteration, so strict Macro+Tail plateau
  cannot be computed for them from the current logs alone. The script marks this explicitly.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


METHOD_HEADERS = {
    "PASSIVE LEARNING EXPERIMENT": "Passive",
    "ACTIVE LEARNING BASELINE": "Active",
    "PROPOSED FRAMEWORK (Generate-only)": "Generate-only",
    "PROPOSED FRAMEWORK (Generate-and-Verify)": "Generate-and-Verify",
}

PERF_RE = re.compile(
    r"Performance Summary\s*\|\s*Macro F1:\s*(?P<macro>[0-9.]+)\s*\|\s*"
    r"Head Macro F1 \([^)]+\):\s*(?P<head>[0-9.]+)\s*\|\s*"
    r"Tail Macro F1 \([^)]+\):\s*(?P<tail>[0-9.]+)",
    re.IGNORECASE,
)

TEST_RE = re.compile(
    r"Test\s*-\s*F1:\s*(?P<macro>[0-9.]+),\s*Accuracy:\s*(?P<acc>[0-9.]+)",
    re.IGNORECASE,
)

ITER_RE = re.compile(r"---\s*Iteration\s+(?P<iteration>\d+)\s*---", re.IGNORECASE)
SEED_RE = re.compile(r"Random seed:\s*(?P<seed>\d+)", re.IGNORECASE)
LOG_TS_RE = re.compile(r"experiment_log_Banking77_Refactored_(?P<tag>\d{8}_\d{6})\.txt")

ACCEPT_NO_VAL_RE = re.compile(
    r"Accepted without validator:\s*(?P<accepted>\d+)\s*/\s*(?P<generated>\d+)",
    re.IGNORECASE,
)
ACCEPT_VAL_RE = re.compile(
    r"Accepted:\s*(?P<accepted>\d+)\s*/\s*(?P<verified>\d+)",
    re.IGNORECASE,
)
SYN_ADDED_RE = re.compile(r"Synthetic added:\s*(?P<synthetic>\d+)", re.IGNORECASE)


def detect_method(line: str) -> Optional[str]:
    for marker, method in METHOD_HEADERS.items():
        if marker in line:
            return method
    return None


def parse_log_file(log_path: Path, batch_size: int = 40) -> pd.DataFrame:
    """Parse one experiment log into iteration-level records."""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    seed_match = SEED_RE.search(text)
    seed = int(seed_match.group("seed")) if seed_match else None

    tag_match = LOG_TS_RE.search(text)
    run_tag = tag_match.group("tag") if tag_match else log_path.stem

    records: List[Dict] = []
    current_method: Optional[str] = None
    current_iter: Optional[int] = None
    current_key: Optional[Tuple[str, int]] = None

    # Keep record index for adding generated/accepted counts after metric line.
    index_by_key: Dict[Tuple[str, int], int] = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        method = detect_method(line)
        if method:
            current_method = method
            current_iter = None
            current_key = None
            continue

        iter_match = ITER_RE.search(line)
        if iter_match and current_method:
            current_iter = int(iter_match.group("iteration"))
            current_key = (current_method, current_iter)
            continue

        if current_method is None or current_iter is None:
            continue

        perf_match = PERF_RE.search(line)
        test_match = TEST_RE.search(line)

        if perf_match:
            row = {
                "seed": seed,
                "run_tag": run_tag,
                "log_file": log_path.name,
                "method": current_method,
                "iteration": current_iter,
                "macro_f1": float(perf_match.group("macro")),
                "head_f1": float(perf_match.group("head")),
                "tail_f1": float(perf_match.group("tail")),
                "accuracy": np.nan,
                "has_tail_per_iteration": True,
                "queried_samples_this_iter": batch_size if current_iter < 40 else 0,
                "generated_this_iter": 0,
                "verified_this_iter": 0,
                "accepted_this_iter": 0,
                "synthetic_added_this_iter": 0,
            }
            index_by_key[current_key] = len(records)
            records.append(row)
            continue

        if test_match:
            # Passive/Active logs currently lack Head/Tail F1 per iteration.
            row = {
                "seed": seed,
                "run_tag": run_tag,
                "log_file": log_path.name,
                "method": current_method,
                "iteration": current_iter,
                "macro_f1": float(test_match.group("macro")),
                "head_f1": np.nan,
                "tail_f1": np.nan,
                "accuracy": float(test_match.group("acc")),
                "has_tail_per_iteration": False,
                "queried_samples_this_iter": batch_size if current_iter < 40 else 0,
                "generated_this_iter": 0,
                "verified_this_iter": 0,
                "accepted_this_iter": 0,
                "synthetic_added_this_iter": 0,
            }
            index_by_key[current_key] = len(records)
            records.append(row)
            continue

        # Add generation/verification counts to the existing row for this method/iteration.
        if current_key in index_by_key:
            rec = records[index_by_key[current_key]]
            m = ACCEPT_NO_VAL_RE.search(line)
            if m:
                rec["generated_this_iter"] = int(m.group("generated"))
                rec["verified_this_iter"] = 0
                rec["accepted_this_iter"] = int(m.group("accepted"))
                continue

            m = ACCEPT_VAL_RE.search(line)
            if m:
                # In Generate-and-Verify, denominator is valid samples sent to validator.
                rec["generated_this_iter"] = int(m.group("verified"))
                rec["verified_this_iter"] = int(m.group("verified"))
                rec["accepted_this_iter"] = int(m.group("accepted"))
                continue

            m = SYN_ADDED_RE.search(line)
            if m:
                rec["synthetic_added_this_iter"] = int(m.group("synthetic"))
                continue

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # infer max iteration from log, and correct queried count for final iteration
    max_iter = int(df["iteration"].max())
    df.loc[df["iteration"] >= max_iter, "queried_samples_this_iter"] = 0

    return df.sort_values(["seed", "method", "iteration"]).reset_index(drop=True)


def find_plateau_stop(
    df_method: pd.DataFrame,
    delta: float,
    k: int,
    require_tail: bool = True,
) -> Optional[int]:
    """Return first iteration where both metric gains stay below delta for k consecutive rounds."""
    df = df_method.sort_values("iteration").copy()
    if df.empty or len(df) < k + 1:
        return None

    if require_tail and df["tail_f1"].isna().any():
        return None

    df["delta_macro"] = df["macro_f1"].diff()
    if df["tail_f1"].isna().all():
        df["delta_tail"] = np.nan
        plateau_flag = df["delta_macro"] < delta
    else:
        df["delta_tail"] = df["tail_f1"].diff()
        plateau_flag = (df["delta_macro"] < delta) & (df["delta_tail"] < delta)

    flags = plateau_flag.fillna(False).to_numpy()
    iterations = df["iteration"].to_numpy()

    consecutive = 0
    for i, flag in enumerate(flags):
        if flag:
            consecutive += 1
        else:
            consecutive = 0
        if consecutive >= k:
            return int(iterations[i])
    return None


def summarize_plateau(
    metrics_df: pd.DataFrame,
    deltas: Iterable[float],
    ks: Iterable[int],
    batch_size: int,
    require_tail: bool = True,
) -> pd.DataFrame:
    summaries: List[Dict] = []

    group_cols = ["seed", "run_tag", "method"]
    for (seed, run_tag, method), g in metrics_df.groupby(group_cols, dropna=False):
        g = g.sort_values("iteration")
        max_iter = int(g["iteration"].max())
        final = g.iloc[-1]
        has_tail = bool(g["tail_f1"].notna().all())

        for delta in deltas:
            for k in ks:
                stop_iter = find_plateau_stop(g, delta=delta, k=k, require_tail=require_tail)
                status = "ok"
                if stop_iter is None:
                    if require_tail and not has_tail:
                        status = "not_computable_missing_tail_per_iteration"
                    else:
                        status = "not_triggered"
                    stop_iter_out = max_iter
                else:
                    stop_iter_out = stop_iter

                stopped_row = g[g["iteration"] == stop_iter_out].iloc[-1]
                post_stop = g[g["iteration"] > stop_iter_out]

                saved_iterations = max_iter - stop_iter_out
                saved_query = int(post_stop["queried_samples_this_iter"].sum())
                saved_gen = int(post_stop["generated_this_iter"].sum())
                saved_verify = int(post_stop["verified_this_iter"].sum())
                saved_accepted = int(post_stop["accepted_this_iter"].sum())

                summaries.append(
                    {
                        "seed": seed,
                        "run_tag": run_tag,
                        "method": method,
                        "delta": delta,
                        "k": k,
                        "plateau_status": status,
                        "plateau_stop_iteration": stop_iter if stop_iter is not None else np.nan,
                        "effective_stop_iteration": stop_iter_out,
                        "max_iteration": max_iter,
                        "saved_iterations": saved_iterations,
                        "saved_queried_samples": saved_query,
                        "saved_generated_samples": saved_gen,
                        "saved_verified_samples": saved_verify,
                        "saved_accepted_synthetic_samples": saved_accepted,
                        "stop_macro_f1": float(stopped_row["macro_f1"]),
                        "stop_tail_f1": float(stopped_row["tail_f1"]) if pd.notna(stopped_row["tail_f1"]) else np.nan,
                        "final_macro_f1": float(final["macro_f1"]),
                        "final_tail_f1": float(final["tail_f1"]) if pd.notna(final["tail_f1"]) else np.nan,
                        "macro_f1_gap_to_final": float(final["macro_f1"] - stopped_row["macro_f1"]),
                        "tail_f1_gap_to_final": (
                            float(final["tail_f1"] - stopped_row["tail_f1"])
                            if pd.notna(final["tail_f1"]) and pd.notna(stopped_row["tail_f1"])
                            else np.nan
                        ),
                    }
                )

    return pd.DataFrame(summaries)


def aggregate_sensitivity(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df

    numeric_cols = [
        "effective_stop_iteration",
        "saved_iterations",
        "saved_queried_samples",
        "saved_generated_samples",
        "saved_verified_samples",
        "stop_macro_f1",
        "stop_tail_f1",
        "final_macro_f1",
        "final_tail_f1",
        "macro_f1_gap_to_final",
        "tail_f1_gap_to_final",
    ]

    ok_df = summary_df[summary_df["plateau_status"].isin(["ok", "not_triggered"])].copy()
    grouped = ok_df.groupby(["method", "delta", "k"], dropna=False)
    agg = grouped[numeric_cols].agg(["mean", "std", "min", "max"]).reset_index()
    agg.columns = ["_".join([str(c) for c in col if c]) for col in agg.columns.to_flat_index()]
    return agg


def parse_float_list(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc plateau analysis for Banking77 experiment logs")
    parser.add_argument("--logs", nargs="+", required=True, help="Experiment log text files")
    parser.add_argument("--out-dir", default="posthoc_plateau_outputs", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=40, help="Selected/query samples per iteration")
    parser.add_argument("--deltas", default="0.002,0.005,0.01", help="Comma-separated delta values")
    parser.add_argument("--ks", default="3,5,7", help="Comma-separated k/patience values")
    parser.add_argument(
        "--allow-macro-only-for-missing-tail",
        action="store_true",
        help="Allow Macro-F1-only plateau for methods missing per-iteration Tail F1. Not recommended for final thesis claim.",
    )
    args = parser.parse_args()

    log_paths = [Path(p) for p in args.logs]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_parts = []
    for path in log_paths:
        if not path.exists():
            raise FileNotFoundError(path)
        df = parse_log_file(path, batch_size=args.batch_size)
        if df.empty:
            print(f"[WARN] No records parsed from {path}")
        metrics_parts.append(df)

    metrics_df = pd.concat(metrics_parts, ignore_index=True) if metrics_parts else pd.DataFrame()
    metrics_path = out_dir / "iteration_metrics_posthoc.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    deltas = parse_float_list(args.deltas)
    ks = parse_int_list(args.ks)
    summary_df = summarize_plateau(
        metrics_df,
        deltas=deltas,
        ks=ks,
        batch_size=args.batch_size,
        require_tail=not args.allow_macro_only_for_missing_tail,
    )
    summary_path = out_dir / "plateau_summary_posthoc.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    sensitivity_df = aggregate_sensitivity(summary_df)
    sensitivity_path = out_dir / "plateau_sensitivity_summary.csv"
    sensitivity_df.to_csv(sensitivity_path, index=False, encoding="utf-8-sig")

    print("\nDone.")
    print(f"Iteration metrics: {metrics_path}")
    print(f"Plateau summary:   {summary_path}")
    print(f"Sensitivity:       {sensitivity_path}")

    missing_tail = summary_df[summary_df["plateau_status"] == "not_computable_missing_tail_per_iteration"]
    if not missing_tail.empty:
        methods = ", ".join(sorted(missing_tail["method"].dropna().unique()))
        print("\n[NOTE]")
        print(
            "Strict Macro+Tail plateau cannot be computed for methods missing per-iteration Tail F1: "
            f"{methods}. To include these methods, rerun code after printing/saving head_f1 and tail_f1 "
            "for Passive/Active, or run this script with --allow-macro-only-for-missing-tail for exploratory analysis only."
        )


if __name__ == "__main__":
    main()
