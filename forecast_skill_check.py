#!/usr/bin/env python3
"""forecast_skill_check.py v1
Independent check: is a day-ahead load forecast better than naive baselines, and how sure are we?
Input : one or more ENTSO-E-style CSVs  ("MTU (CET/CEST)","Area","Actual Total Load (MW)","Day-ahead Total Load Forecast (MW)")
        or any CSV with a time column + actual + forecast (use --time/--actual/--forecast).
Output: printed summary + results.json + (optional) --plots PNG files.
Baselines (both use ONLY actuals):
  naive48 : actual 48 h earlier  (fully known when a day-ahead forecast is issued -> the fair one)
  naive24 : actual 24 h earlier  (NOT fully known at issue time for evening hours -> optimistic reference)
Skill = 1 - MAE_forecast / MAE_baseline  (>0 means the forecast beats the baseline).
Uncertainty: block bootstrap over calendar days (2000 resamples), 95% interval.
"""
import argparse, glob, json, re, sys
import numpy as np, pandas as pd

def load(paths, tcol, acol, fcol):
    frames = []
    for p in paths:
        d = pd.read_csv(p)
        t = d[tcol].astype(str).str.split(" - ").str[0]
        ts = pd.to_datetime(t, format="%d/%m/%Y %H:%M", errors="coerce")
        out = pd.DataFrame({"ts": ts, "actual": pd.to_numeric(d[acol], errors="coerce"),
                            "fc": pd.to_numeric(d[fcol], errors="coerce"), "src": p.split("/")[-1]})
        frames.append(out.dropna(subset=["ts"]))
    df = pd.concat(frames).drop_duplicates("ts").sort_values("ts").set_index("ts")
    return df

def add_baselines(df):
    a = df["actual"]
    df["naive24"] = a.reindex(df.index - pd.Timedelta(hours=24)).values
    df["naive48"] = a.reindex(df.index - pd.Timedelta(hours=48)).values
    return df

def mae(x, y): return float(np.mean(np.abs(x - y)))

def skill_ci(d, base, n=2000, seed=7):
    d = d.dropna(subset=["actual", "fc", base]).copy()
    d["day"] = d.index.normalize()
    days = d["day"].unique()
    g = {k: v for k, v in d.groupby("day")}
    rng = np.random.default_rng(seed)
    pt = 1 - mae(d["fc"], d["actual"]) / mae(d[base], d["actual"])
    bs = []
    for _ in range(n):
        pick = rng.choice(len(days), len(days), replace=True)
        e_f = np.concatenate([np.abs(g[days[i]]["fc"] - g[days[i]]["actual"]).values for i in pick])
        e_b = np.concatenate([np.abs(g[days[i]][base] - g[days[i]]["actual"]).values for i in pick])
        bs.append(1 - e_f.mean() / e_b.mean())
    return pt, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(d), len(days)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--time", default="MTU (CET/CEST)")
    ap.add_argument("--actual", default="Actual Total Load (MW)")
    ap.add_argument("--forecast", default="Day-ahead Total Load Forecast (MW)")
    ap.add_argument("--json", default="results.json")
    a = ap.parse_args()
    paths = sorted(sum([glob.glob(f) for f in a.files], []))
    df = add_baselines(load(paths, a.time, a.actual, a.forecast))
    res = {"files": len(paths), "hours": int(len(df)), "from": str(df.index.min()), "to": str(df.index.max())}
    e = df["fc"] - df["actual"]
    res["forecast"] = {"MAE_MW": mae(df["fc"], df["actual"]),
                       "RMSE_MW": float(np.sqrt((e**2).mean())),
                       "MAPE_pct": float((e.abs() / df["actual"]).mean() * 100),
                       "bias_MW": float(e.mean()),
                       "share_hours_err_gt_5pct": float(((e.abs() / df["actual"]) > 0.05).mean()),
                       "share_hours_err_gt_10pct": float(((e.abs() / df["actual"]) > 0.10).mean())}
    res["skill"] = {}
    for b in ["naive48", "naive24"]:
        pt, lo, hi, n, nd = skill_ci(df, b)
        sub = df.dropna(subset=[b])
        res["skill"][b] = {"skill": pt, "ci95": [lo, hi], "hours": n, "days": nd,
                           "MAE_baseline_MW": mae(sub[b], sub["actual"]), "MAE_forecast_same_rows_MW": mae(sub["fc"], sub["actual"])}
    df["month"] = df.index.to_period("M").astype(str)
    bym = []
    for m, g in df.groupby("month"):
        r = {"month": m, "hours": int(len(g)), "MAE_fc": mae(g["fc"], g["actual"]),
             "MAPE_fc_pct": float((np.abs(g["fc"] - g["actual"]) / g["actual"]).mean() * 100)}
        gg = g.dropna(subset=["naive48"])
        r["MAE_naive48"] = mae(gg["naive48"], gg["actual"]) if len(gg) else None
        r["skill48"] = (1 - r["MAE_fc"] / r["MAE_naive48"]) if len(gg) else None
        r["MAE_fc_same_rows"] = mae(gg["fc"], gg["actual"]) if len(gg) else None
        r["skill48"] = (1 - r["MAE_fc_same_rows"] / r["MAE_naive48"]) if len(gg) else None
        bym.append(r)
    res["by_month"] = bym
    hr = df.groupby(df.index.hour).apply(lambda g: pd.Series({"MAE": mae(g["fc"], g["actual"]), "bias": float((g["fc"] - g["actual"]).mean())}))
    res["by_hour"] = {int(k): {"MAE": float(v["MAE"]), "bias": float(v["bias"])} for k, v in hr.iterrows()}
    # worst days
    df["day"] = df.index.normalize()
    dd = df.groupby("day").apply(lambda g: mae(g["fc"], g["actual"])).sort_values(ascending=False)
    res["worst_days_MAE"] = {str(k.date()): float(v) for k, v in dd.head(5).items()}
    # data quirks: runs of identical consecutive forecast values (>=3 hours)
    runs, cur, start = 0, 1, None
    f = df["fc"].values
    for i in range(1, len(f)):
        if f[i] == f[i-1] and (df.index[i] - df.index[i-1]) == pd.Timedelta(hours=1):
            cur += 1
            if cur == 3: runs += 1
        else:
            cur = 1
    res["runs_of_3plus_identical_forecast_hours"] = runs
    json.dump(res, open(a.json, "w"), indent=2, default=str)
    print(json.dumps(res, indent=2, default=str))

if __name__ == "__main__":
    main()
