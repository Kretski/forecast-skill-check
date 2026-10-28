#!/usr/bin/env python3
"""
forecast_skill_check.py v2
Independent, reproducible checks for ENTSO-E-style day-ahead load forecasts.

Key safeguards:
- validates columns, numeric values, timestamps, duplicates, gaps, and zero actuals;
- does not silently discard conflicting duplicate timestamps;
- computes each forecast-vs-baseline comparison on the same valid rows;
- reports the observed sample window and missing-data diagnostics;
- uses a month-stratified circular moving-block bootstrap over days for skill CIs.

Important:
ENTSO-E exports labelled CET/CEST may contain local wall-clock timestamps without
UTC offsets. This script deliberately keeps those timestamps naive and treats
naive24/naive48 as wall-clock lags. If a dataset includes repeated local timestamps
(e.g. the autumn DST transition), it stops and asks for timestamps with explicit
offsets or a documented DST disambiguation; it will not guess.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def parse_timestamp_column(series: pd.Series) -> pd.Series:
    """Parse interval start strings such as '01/10/2025 00:00 - 01/10/2025 01:00'."""
    start = series.astype(str).str.split(" - ", n=1).str[0].str.strip()
    # ENTSO-E GUI exports normally use day/month/year.
    return pd.to_datetime(start, format="%d/%m/%Y %H:%M", errors="coerce")


def load_files(
    paths: Iterable[str],
    time_col: str,
    actual_col: str,
    forecast_col: str,
) -> tuple[pd.DataFrame, dict]:
    frames = []
    diagnostics = {
        "input_files": 0,
        "input_rows": 0,
        "invalid_timestamps": 0,
        "invalid_actual_values": 0,
        "invalid_forecast_values": 0,
        "duplicate_timestamps": 0,
        "conflicting_duplicate_timestamps": 0,
        "missing_hour_gaps": 0,
        "missing_actual_values": 0,
        "missing_forecast_values": 0,
        "zero_actual_values": 0,
        "negative_actual_values": 0,
    }

    paths = list(paths)
    if not paths:
        raise ValueError("No input files matched the supplied path/pattern.")

    for path in paths:
        try:
            raw = pd.read_csv(path)
        except Exception as exc:
            raise ValueError(f"Could not read CSV {path!r}: {exc}") from exc

        required = [time_col, actual_col, forecast_col]
        missing_cols = [c for c in required if c not in raw.columns]
        if missing_cols:
            raise ValueError(
                f"{path}: missing required column(s) {missing_cols}. "
                f"Available columns: {list(raw.columns)}"
            )

        diagnostics["input_files"] += 1
        diagnostics["input_rows"] += int(len(raw))
        ts = parse_timestamp_column(raw[time_col])
        actual = pd.to_numeric(raw[actual_col], errors="coerce")
        forecast = pd.to_numeric(raw[forecast_col], errors="coerce")
        diagnostics["invalid_timestamps"] += int(ts.isna().sum())
        diagnostics["invalid_actual_values"] += int((raw[actual_col].notna() & actual.isna()).sum())
        diagnostics["invalid_forecast_values"] += int((raw[forecast_col].notna() & forecast.isna()).sum())
        diagnostics["missing_actual_values"] += int(actual.isna().sum())
        diagnostics["missing_forecast_values"] += int(forecast.isna().sum())
        diagnostics["zero_actual_values"] += int((actual == 0).sum())
        diagnostics["negative_actual_values"] += int((actual < 0).sum())

        frame = pd.DataFrame({
            "ts": ts,
            "actual": actual,
            "fc": forecast,
            "source_file": Path(path).name,
        })
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["ts"]).copy()

    # Check duplicate timestamps before any de-duplication.
    duplicate_mask = df.duplicated("ts", keep=False)
    diagnostics["duplicate_timestamps"] = int(df.loc[duplicate_mask, "ts"].nunique())
    if duplicate_mask.any():
        dup = df.loc[duplicate_mask].sort_values("ts")
        conflict_count = 0
        for _, group in dup.groupby("ts"):
            # A repeated timestamp is conflicting if actual/forecast values differ.
            if group[["actual", "fc"]].drop_duplicates().shape[0] > 1:
                conflict_count += 1
        diagnostics["conflicting_duplicate_timestamps"] = conflict_count
        sample = dup["ts"].drop_duplicates().astype(str).head(5).tolist()
        raise ValueError(
            f"Found {diagnostics['duplicate_timestamps']} duplicate timestamp(s) "
            f"({conflict_count} with conflicting values). Refusing to silently drop them. "
            f"Examples: {sample}. If these are repeated DST hours, provide timestamps "
            "with UTC offsets or resolve the source export explicitly."
        )

    df = df.sort_values("ts").set_index("ts")
    if df.empty:
        raise ValueError("No rows have valid timestamps.")

    # Count gaps in the observed sequence. Do not fill them with invented data.
    diffs = df.index.to_series().diff().dropna()
    diagnostics["missing_hour_gaps"] = int((diffs > pd.Timedelta(hours=1)).sum())
    diagnostics["non_hourly_steps"] = int((diffs != pd.Timedelta(hours=1)).sum())

    return df, diagnostics


def add_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Wall-clock lags: actual value exactly 24/48 hours earlier in the index."""
    out = df.copy()
    actual = out["actual"]
    out["naive24"] = actual.reindex(out.index - pd.Timedelta(hours=24)).to_numpy()
    out["naive48"] = actual.reindex(out.index - pd.Timedelta(hours=48)).to_numpy()
    return out


def mae(actual: np.ndarray | pd.Series, prediction: np.ndarray | pd.Series) -> float:
    return float(np.mean(np.abs(np.asarray(prediction, dtype=float) - np.asarray(actual, dtype=float))))


def safe_mape(actual: pd.Series, forecast: pd.Series) -> float | None:
    """MAPE excluding zero actuals; zero count is separately reported."""
    mask = actual.notna() & forecast.notna() & (actual != 0)
    if not mask.any():
        return None
    return float((np.abs(forecast[mask] - actual[mask]) / np.abs(actual[mask])).mean() * 100.0)


def skill_value(d: pd.DataFrame, baseline_col: str) -> float:
    e_fc = np.abs(d["fc"].to_numpy() - d["actual"].to_numpy())
    e_base = np.abs(d[baseline_col].to_numpy() - d["actual"].to_numpy())
    base_mae = float(e_base.mean())
    if not np.isfinite(base_mae) or base_mae == 0:
        return float("nan")
    return float(1.0 - e_fc.mean() / base_mae)


def block_bootstrap_ci(
    d: pd.DataFrame,
    baseline_col: str,
    n_resamples: int = 5000,
    seed: int = 7,
    block_days: int = 3,
) -> tuple[float, float, float, int, int]:
    """
    Month-stratified circular moving-block bootstrap.
    Resamples contiguous blocks of calendar days within each observed month.
    This preserves month composition and some short-range day-to-day dependence.
    """
    cols = ["actual", "fc", baseline_col]
    d = d.dropna(subset=cols).copy()
    if d.empty:
        return float("nan"), float("nan"), float("nan"), 0, 0

    d["day"] = d.index.normalize()
    d["month"] = d.index.to_period("M").astype(str)
    month_days: dict[str, list[pd.Timestamp]] = {}
    day_groups: dict[pd.Timestamp, pd.DataFrame] = {}
    for day, group in d.groupby("day", sort=True):
        day_groups[pd.Timestamp(day)] = group
    for month, group in d.groupby("month", sort=True):
        days = [pd.Timestamp(x) for x in group["day"].drop_duplicates().sort_values()]
        month_days[month] = days

    point = skill_value(d, baseline_col)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_resamples):
        sampled_groups = []
        for _, days in month_days.items():
            if not days:
                continue
            target_n = len(days)
            picked: list[pd.Timestamp] = []
            # Circular blocks are restricted to the same month.
            while len(picked) < target_n:
                start = int(rng.integers(0, len(days)))
                for offset in range(min(block_days, target_n - len(picked))):
                    picked.append(days[(start + offset) % len(days)])
                    if len(picked) >= target_n:
                        break
            sampled_groups.extend(day_groups[day] for day in picked)
        if not sampled_groups:
            continue
        sample = pd.concat(sampled_groups, ignore_index=True)
        estimates.append(skill_value(sample, baseline_col))

    estimates = np.asarray(estimates, dtype=float)
    estimates = estimates[np.isfinite(estimates)]
    if not len(estimates):
        return point, float("nan"), float("nan"), int(len(d)), int(d["day"].nunique())
    lo, hi = np.percentile(estimates, [2.5, 97.5])
    return point, float(lo), float(hi), int(len(d)), int(d["day"].nunique())


def calculate(df: pd.DataFrame, diagnostics: dict, n_boot: int, seed: int, block_days: int) -> dict:
    df = add_baselines(df)
    actual = df["actual"]
    fc = df["fc"]
    valid_forecast = actual.notna() & fc.notna()
    if not valid_forecast.any():
        raise ValueError("No rows have both valid actual and forecast values.")
    base_df = df.loc[valid_forecast].copy()
    err = fc[valid_forecast] - actual[valid_forecast]
    abs_err_pct = (err.abs() / actual[valid_forecast].abs()).where(actual[valid_forecast] != 0)

    result = {
        "version": "2.0",
        "files": diagnostics["input_files"],
        "input_rows": diagnostics["input_rows"],
        "hours": int(len(df)),
        "from": str(df.index.min()),
        "to": str(df.index.max()),
        "timestamp_semantics": "naive local wall-clock timestamps; naive24/naive48 are wall-clock lags",
        "diagnostics": diagnostics,
        "forecast": {
            "valid_hours": int(valid_forecast.sum()),
            "MAE_MW": mae(actual[valid_forecast], fc[valid_forecast]),
            "RMSE_MW": float(np.sqrt(np.mean(np.square(err.to_numpy())))),
            "MAPE_pct": safe_mape(actual, fc),
            "bias_MW": float(err.mean()),
            "share_hours_err_gt_5pct": float((abs_err_pct > 0.05).mean()),
            "share_hours_err_gt_10pct": float((abs_err_pct > 0.10).mean()),
            "mape_excluded_zero_actual_hours": int((valid_forecast & (actual == 0)).sum()),
        },
        "skill": {},
        "by_month": [],
        "by_hour": {},
        "worst_days_MAE": {},
        "runs_of_3plus_identical_forecast_hours": 0,
        "bootstrap": {
            "method": "month-stratified circular moving-block bootstrap over calendar days",
            "resamples": int(n_boot),
            "block_days": int(block_days),
            "seed": int(seed),
        },
    }

    for baseline in ("naive48", "naive24"):
        paired = df.dropna(subset=["actual", "fc", baseline]).copy()
        pt, lo, hi, n, nd = block_bootstrap_ci(
            paired, baseline, n_resamples=n_boot, seed=seed, block_days=block_days
        )
        result["skill"][baseline] = {
            "skill": pt,
            "ci95": [lo, hi],
            "hours": n,
            "days": nd,
            "MAE_baseline_MW": mae(paired["actual"], paired[baseline]) if n else None,
            "MAE_forecast_same_rows_MW": mae(paired["actual"], paired["fc"]) if n else None,
            "interpretation": (
                "CI entirely above zero: evidence of improvement over this baseline"
                if np.isfinite(lo) and lo > 0 else
                "CI includes zero: improvement not established by this interval"
                if np.isfinite(lo) and np.isfinite(hi) else
                "insufficient valid paired data"
            ),
        }

    df["month"] = df.index.to_period("M").astype(str)
    for month, g in df.groupby("month", sort=True):
        valid = g.dropna(subset=["actual", "fc"])
        paired = g.dropna(subset=["actual", "fc", "naive48"])
        result["by_month"].append({
            "month": month,
            "hours": int(len(g)),
            "valid_forecast_hours": int(len(valid)),
            "MAE_fc": mae(valid["actual"], valid["fc"]) if len(valid) else None,
            "MAPE_fc_pct": safe_mape(valid["actual"], valid["fc"]) if len(valid) else None,
            "MAE_naive48": mae(paired["actual"], paired["naive48"]) if len(paired) else None,
            "MAE_fc_same_rows": mae(paired["actual"], paired["fc"]) if len(paired) else None,
            "skill48": skill_value(paired, "naive48") if len(paired) else None,
            "paired_hours_naive48": int(len(paired)),
            "note": "Monthly estimates are based only on observed rows; not a full-month evaluation."
        })

    valid = df.dropna(subset=["actual", "fc"]).copy()
    for hour, group in valid.groupby(valid.index.hour):
        result["by_hour"][str(int(hour))] = {
            "MAE": mae(group["actual"], group["fc"]),
            "bias": float((group["fc"] - group["actual"]).mean()),
            "hours": int(len(group)),
        }

    valid["day"] = valid.index.normalize()
    daily_mae = valid.groupby("day").apply(lambda g: mae(g["actual"], g["fc"]), include_groups=False)
    result["worst_days_MAE"] = {
        str(day.date()): float(value)
        for day, value in daily_mae.sort_values(ascending=False).head(5).items()
    }

    f = df["fc"].to_numpy()
    idx = df.index
    runs = 0
    current = 1
    for i in range(1, len(f)):
        consecutive = (idx[i] - idx[i - 1]) == pd.Timedelta(hours=1)
        same = pd.notna(f[i]) and pd.notna(f[i - 1]) and f[i] == f[i - 1]
        if consecutive and same:
            current += 1
            if current == 3:
                runs += 1
        else:
            current = 1
    result["runs_of_3plus_identical_forecast_hours"] = int(runs)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate day-ahead load forecasts against naive baselines."
    )
    parser.add_argument("files", nargs="+", help="CSV paths or glob patterns, e.g. data/*.csv")
    parser.add_argument("--time", default="MTU (CET/CEST)", help="Timestamp/interval column")
    parser.add_argument("--actual", default="Actual Total Load (MW)", help="Actual load column")
    parser.add_argument("--forecast", default="Day-ahead Total Load Forecast (MW)", help="Forecast column")
    parser.add_argument("--json", default="results.json", help="Output JSON path")
    parser.add_argument("--bootstrap", type=int, default=5000, help="Bootstrap resamples (default: 5000)")
    parser.add_argument("--block-days", type=int, default=3, help="Bootstrap block length in days (default: 3)")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    args = parser.parse_args()

    if args.bootstrap < 100:
        parser.error("--bootstrap must be at least 100.")
    if args.block_days < 1:
        parser.error("--block-days must be >= 1.")

    paths = sorted(set(sum([glob.glob(p) for p in args.files], [])))
    if not paths:
        parser.error("No files matched the supplied path/pattern.")

    try:
        df, diagnostics = load_files(paths, args.time, args.actual, args.forecast)
        result = calculate(df, diagnostics, args.bootstrap, args.seed, args.block_days)
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        print(f"\nWrote: {out_path}", file=sys.stderr)
        return 0
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
