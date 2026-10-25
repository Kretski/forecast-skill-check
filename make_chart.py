#!/usr/bin/env python3
"""make_chart.py: draws hourly bias and monthly skill from results.json -> chart.png"""
import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
r = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results.json"))
out = sys.argv[2] if len(sys.argv) > 2 else "chart.png"
h = sorted(map(int, r["by_hour"]))
bias = [r["by_hour"][str(x)]["bias"] for x in h]
mae = [r["by_hour"][str(x)]["MAE"] for x in h]
m = [b["month"] for b in r["by_month"]]
sk = [100 * (b["skill48"] or 0) for b in r["by_month"]]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].bar(h, bias, color=["#c0504d" if b < 0 else "#4f81bd" for b in bias])
ax[0].plot(h, mae, color="black", lw=1.2, label="MAE")
ax[0].axhline(0, color="gray", lw=.8); ax[0].legend()
ax[0].set_title("Bias by hour (forecast - actual), MW"); ax[0].set_xlabel("hour of day (local)")
ax[1].bar(range(len(m)), sk, color="#4f81bd")
ax[1].set_xticks(range(len(m))); ax[1].set_xticklabels(m, rotation=60, ha="right", fontsize=8)
ax[1].axhline(0, color="gray", lw=.8)
ax[1].set_title("Skill vs naive 48h baseline by month, % (1 week/month)")
plt.tight_layout(); plt.savefig(out, dpi=140)
