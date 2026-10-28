#!/usr/bin/env python3
"""Plot forecast-skill-check v2 JSON output.

Usage:
    python make_chart.py results_v2.json example/chart.png

Produces:
    - hourly forecast bias (forecast - actual) and MAE
    - monthly skill versus naive48, with paired-hour counts
    - 95% bootstrap interval for overall skill versus each baseline (if present)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results_v2.json")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("example/chart.png")

    with src.open("r", encoding="utf-8") as f:
        r = json.load(f)

    out.parent.mkdir(parents=True, exist_ok=True)

    # 1) Hourly bias and MAE
    by_hour = r.get("by_hour", {})
    hours = sorted(int(k) for k in by_hour.keys())
    bias = [float(by_hour[str(h)]["bias"]) for h in hours]
    mae = [float(by_hour[str(h)]["MAE"]) for h in hours]

    # 2) Monthly skill vs naive48
    months = r.get("by_month", [])
    month_labels = [str(row["month"]) for row in months]
    monthly_skill = [
        100.0 * float(row["skill48"])
        if row.get("skill48") is not None else np.nan
        for row in months
    ]
    paired_hours = [
        int(row.get("paired_hours_naive48", 0) or 0)
        for row in months
    ]

    # 3) Overall skill and CI
    skills = r.get("skill", {})
    baseline_names = []
    point_estimates = []
    lower = []
    upper = []
    for key, label in (("naive48", "naive48"), ("naive24", "naive24")):
        item = skills.get(key)
        if not item or item.get("skill") is None:
            continue
        ci = item.get("ci95")
        if not isinstance(ci, list) or len(ci) != 2:
            continue
        baseline_names.append(label)
        point_estimates.append(100.0 * float(item["skill"]))
        lower.append(100.0 * float(ci[0]))
        upper.append(100.0 * float(ci[1]))

    fig, ax = plt.subplots(3, 1, figsize=(12, 11), constrained_layout=True)

    # Panel 1: hourly bias and MAE
    ax[0].bar(hours, bias, label="Bias (forecast - actual), MW")
    ax[0].plot(hours, mae, marker="o", linewidth=1.4, label="MAE, MW")
    ax[0].axhline(0, linewidth=0.8)
    ax[0].set_title("Hourly forecast error")
    ax[0].set_xlabel("Hour of day (input timestamp convention)")
    ax[0].set_ylabel("MW")
    ax[0].set_xticks(hours)
    ax[0].grid(axis="y", alpha=0.25)
    ax[0].legend()

    # Panel 2: monthly skill
    x = np.arange(len(month_labels))
    ax[1].bar(x, monthly_skill)
    ax[1].axhline(0, linewidth=0.8)
    ax[1].set_title("Monthly skill vs naive48 (descriptive; sampled weeks only)")
    ax[1].set_ylabel("Skill (%)")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(month_labels, rotation=45, ha="right")
    ax[1].grid(axis="y", alpha=0.25)

    # Label each monthly bar with number of paired hours
    for xi, val, n in zip(x, monthly_skill, paired_hours):
        if np.isfinite(val):
            offset = 1.0 if val >= 0 else -2.0
            ax[1].text(xi, val + offset, f"n={n}", ha="center",
                       va="bottom" if val >= 0 else "top", fontsize=7)

    # Panel 3: overall skill with 95% CI
    if baseline_names:
        y = np.arange(len(baseline_names))
        points = np.asarray(point_estimates)
        lows = np.asarray(lower)
        highs = np.asarray(upper)
        xerr = np.vstack([points - lows, highs - points])
        ax[2].errorbar(points, y, xerr=xerr, fmt="o", capsize=4)
        ax[2].axvline(0, linewidth=0.8)
        ax[2].set_yticks(y)
        ax[2].set_yticklabels(baseline_names)
        ax[2].set_xlabel("Skill (%)")
        ax[2].set_title("Overall skill and 95% bootstrap interval")
        ax[2].grid(axis="x", alpha=0.25)
    else:
        ax[2].text(0.5, 0.5, "No overall skill confidence intervals in JSON",
                   ha="center", va="center", transform=ax[2].transAxes)
        ax[2].set_axis_off()

    diagnostics = r.get("diagnostics", {})
    subtitle = (
        f"Files: {r.get('files', diagnostics.get('input_files', '?'))} | "
        f"Hours: {r.get('hours', r.get('input_rows', '?'))} | "
        f"Bootstrap: {r.get('bootstrap', {}).get('resamples', '?')} resamples, "
        f"block={r.get('bootstrap', {}).get('block_days', '?')} days"
    )
    fig.suptitle("forecast-skill-check v2\n" + subtitle, fontsize=11)

    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Chart saved to: {out}")


if __name__ == "__main__":
    main()
