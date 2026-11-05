#!/usr/bin/env python3
"""
IBV / SM Backtest
-----------------
Tests the IBV / SM buy-zone strategy on ~10+ years of daily data for the S&P 500,
Nasdaq-100 and your watchlist, using the SAME signal logic as scanner.py.

What it measures
  1. Trade results for a grid of exit rules (stop x target), in R-multiples and %.
  2. A random-entry CONTROL: same stocks, same uptrend filter, same exits, but entries on
     random uptrend days instead of zone entries. The zone only "works" if it beats this.
  3. An event study: forward returns 5/10/20/60 days after a zone entry vs. every other
     uptrend day, raw and minus SPY over the same window.
  4. In-sample (2012-2019) vs out-of-sample (2020-today) split, plus breakdowns by zone,
     market regime (SPY vs its 200D MA), RSI at entry, year, and watchlist vs index.
  5. A portfolio simulation (max 10 positions, 10% each) vs SPY buy-and-hold.

Rules (primary)
  Signal : daily close enters IBV (0.618-0.786) or SM (0.786-0.826) with bullish structure
  Entry  : next day's open
  Stop   : daily close below the 0.826 level -> exit next open
  Target : swing high (Fib 0 level)
  Time   : exit at the close after 60 trading days
  Costs  : 0.10% round trip
Run: python backtest.py   (DRY_RUN=1 writes backtest_report.html instead of emailing)
"""
from __future__ import annotations

import datetime as dt
import html
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import scanner as s

# ── Settings ────────────────────────────────────────────────────────────────
DATA_START = "2011-01-01"          # warm-up for the 200D MA
SIGNAL_START = "2012-01-01"
OOS_START = "2020-01-01"           # out-of-sample begins here
COST_PCT = 0.10                    # round-trip cost, % of price
TIME_STOP = 60                     # trading days
FWD = (5, 10, 20, 60)              # event-study horizons
CONTROL_MULT = 3                   # random control entries per real signal
MAX_POS, POS_PCT, START_CAPITAL = 10, 0.10, 100_000
INCLUDE_CRYPTO = False
SEED = 42

STOPS = {"0.826 close": "z3", "0.886 close": "f886", "Swing-low close": "leg_lo"}
ATR_FLOOR = 1.0                    # R uses risk = max(entry - stop, 1 x ATR14) so tiny stops do not inflate R
TARGETS = {"Swing high": "hi", "1.272 ext": "ext"}
PRIMARY = ("0.826 close", "Swing high")

BASE = Path(__file__).resolve().parent
C = s.C


# ── Data ────────────────────────────────────────────────────────────────────
def load():
    universe, watch, notes, meta = s.build_universe()
    if not INCLUDE_CRYPTO:
        universe = [t for t in universe if not t.endswith("-USD")]
    s.HISTORY = "max"
    data = s.download(universe + ["SPY"])
    data = {t: d[d.index >= DATA_START] for t, d in data.items()}
    data = {t: d for t, d in data.items() if len(d) > s.SMA_LEN + 50}
    spy = data.pop("SPY", None)
    return data, spy, set(watch), notes, meta


# ── Trade simulation ────────────────────────────────────────────────────────
def simulate(o, h, lo, c, e, stop, target):
    """Enter at open[e]. Returns (exit_bar, exit_px, reason) or None if still open."""
    n = len(c)
    last = min(e + TIME_STOP - 1, n - 1)
    for j in range(e, last + 1):
        if o[j] >= target:                       # gapped through target
            return j, o[j], "target"
        if c[j] < stop:                          # close-based stop -> next open
            return (j + 1, o[j + 1], "stop") if j + 1 < n else None
        if h[j] >= target:
            return j, target, "target"
    if e + TIME_STOP - 1 <= n - 1:
        return last, c[last], "time"
    return None                                  # trade still open at end of data


def levels(hist, i, stop_key, tgt_key):
    hi, lo_, z3 = hist["leg_hi"][i], hist["leg_lo"][i], hist["z3"][i]
    stop = {"z3": z3, "leg_lo": lo_, "f886": hi - 0.886 * (hi - lo_)}[stop_key]
    target = hi if tgt_key == "hi" else hi + 0.272 * (hi - lo_)
    return stop, target


def run_trades(t, df, hist, entries, stop_key, tgt_key, kind):
    o, h, lo, c = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    dates = df.index
    rows, busy_until = [], -1
    for i in entries:
        e = i + 1
        if e >= len(c) or e <= busy_until:
            continue
        stop, target = levels(hist, i, stop_key, tgt_key)
        entry = o[e] * (1 + COST_PCT / 200)
        risk = max(entry - stop, ATR_FLOOR * hist["atr"][i]) if np.isfinite(hist["atr"][i]) else entry - stop
        if not (np.isfinite(stop) and np.isfinite(target)) or entry <= stop or entry >= target:
            continue
        res = simulate(o, h, lo, c, e, stop, target)
        if res is None:
            continue
        xb, xp, why = res
        xp *= (1 - COST_PCT / 200)
        busy_until = xb
        rows.append({
            "ticker": t, "kind": kind, "zone": s.ZONE_NAMES.get(int(hist["zone"][i]), ""),
            "signal_date": dates[i], "entry_date": dates[e], "exit_date": dates[xb],
            "entry": entry, "stop": stop, "target": target, "exit": xp, "reason": why,
            "bars": xb - e + 1, "ret_pct": (xp / entry - 1) * 100, "R": (xp - entry) / risk,
            "rsi": hist["rsi"][i], "sig_bar": i,
        })
    return rows


def atr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    h, l, c = (df[k] for k in ("High", "Low", "Close"))
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean().to_numpy()


# ── Stats ───────────────────────────────────────────────────────────────────
def stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"Trades": 0}
    R = df["R"].to_numpy()
    wins, losses = R[R > 0], R[R <= 0]
    rng = np.random.default_rng(SEED)
    boot = rng.choice(R, (1000, len(R))).mean(axis=1) if len(R) > 1 else np.array([R.mean()])
    t_stat = R.mean() / (R.std(ddof=1) / np.sqrt(len(R))) if len(R) > 2 and R.std() > 0 else np.nan
    return {
        "Trades": len(R),
        "Win %": 100 * len(wins) / len(R),
        "Avg win R": wins.mean() if len(wins) else np.nan,
        "Avg loss R": losses.mean() if len(losses) else np.nan,
        "Expectancy R": R.mean(),
        "95% CI": f"{np.percentile(boot, 2.5):+.2f} to {np.percentile(boot, 97.5):+.2f}",
        "t-stat": t_stat,
        "Profit factor": wins.sum() / -losses.sum() if losses.sum() < 0 else np.nan,
        "Expectancy %": df["ret_pct"].mean(),
        "Median days": df["bars"].median(),
        "Target / Stop / Time %": " / ".join(f"{100 * (df['reason'] == k).mean():.0f}" for k in ("target", "stop", "time")),
    }


def portfolio(trades: pd.DataFrame, data: dict, spy: pd.DataFrame) -> tuple[pd.Series, dict, int]:
    days = spy.index[spy.index >= SIGNAL_START]
    tr = trades.copy()
    tr["prio"] = tr["zone"].map({"SM": 0, "IBV": 1}).fillna(2)
    tr = tr.sort_values(["entry_date", "prio", "rsi"])
    by_entry = {d: g for d, g in tr.groupby("entry_date")}
    cash, open_pos, curve, taken = START_CAPITAL, [], [], 0
    closes = {t: d["Close"] for t, d in data.items()}
    equity = START_CAPITAL
    for d in days:
        still = []
        for p in open_pos:
            if p["exit_date"] == d:
                cash += p["shares"] * p["exit"]
            else:
                still.append(p)
        open_pos = still
        if d in by_entry:
            for _, r in by_entry[d].iterrows():
                if len(open_pos) >= MAX_POS:
                    break
                alloc = min(POS_PCT * equity, cash)
                if alloc <= 0:
                    break
                cash -= alloc
                open_pos.append({"ticker": r["ticker"], "shares": alloc / r["entry"], "exit": r["exit"],
                                 "exit_date": r["exit_date"]})
                taken += 1
        mv = 0.0
        for p in open_pos:
            px = closes[p["ticker"]].asof(d)
            mv += p["shares"] * (px if px == px else 0)
        equity = cash + mv
        curve.append(equity)
    eq = pd.Series(curve, index=days)
    return eq, perf(eq), taken


def perf(eq: pd.Series) -> dict:
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    rets = eq.pct_change().dropna()
    dd = (eq / eq.cummax() - 1).min()
    return {"CAGR %": 100 * ((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1),
            "Max drawdown %": 100 * dd,
            "Sharpe": rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else np.nan,
            "Total return %": 100 * (eq.iloc[-1] / eq.iloc[0] - 1)}


# ── Charts ──────────────────────────────────────────────────────────────────
def _fig(h=2.6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5})
    fig, ax = plt.subplots(figsize=(6.4, h), dpi=200)
    ax.grid(axis="y", color=C["grid"], linewidth=0.6)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(C["muted"])
    ax.tick_params(colors=C["ink2"], length=0)
    return fig, ax, plt


def _png(fig, plt):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def chart_equity(eq, spy_eq):
    fig, ax, plt = _fig()
    a, b = 100 * eq / eq.iloc[0], 100 * spy_eq / spy_eq.iloc[0]
    ax.plot(a.index, a, color=C["sma50"], linewidth=1.4, label="IBV/SM portfolio")
    ax.plot(b.index, b, color=C["sma200"], linewidth=1.4, label="SPY buy & hold")
    for ser, col in ((a, C["sma50"]), (b, C["sma200"])):
        ax.annotate(f"{ser.iloc[-1]:.0f}", xy=(ser.index[-1], ser.iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=7, color=C["ink2"])
    ax.axvline(pd.Timestamp(OOS_START), color=C["muted"], linewidth=0.7, linestyle=(0, (3, 3)))
    ax.text(pd.Timestamp(OOS_START), ax.get_ylim()[1], " out-of-sample →", va="top", fontsize=6.5, color=C["ink2"])
    ax.legend(loc="upper left", frameon=False, fontsize=7, labelcolor=C["ink2"])
    ax.set_title("Growth of 100 (log scale)", loc="left", fontsize=8.5, color=C["ink"])
    ax.set_yscale("log")
    from matplotlib.ticker import ScalarFormatter, NullFormatter
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    return _png(fig, plt)


def chart_r_hist(sig, ctl):
    fig, ax, plt = _fig()
    bins = np.linspace(-2, 4, 31)
    ax.hist(np.clip(ctl["R"], -2, 4), bins=bins, density=True, color=C["muted"], alpha=0.45, label="Random-entry control")
    ax.hist(np.clip(sig["R"], -2, 4), bins=bins, density=True, histtype="step", linewidth=1.6,
            color=C["sma50"], label="IBV/SM entries")
    ax.axvline(0, color=C["ink2"], linewidth=0.7)
    ax.legend(frameon=False, fontsize=7, labelcolor=C["ink2"])
    ax.set_title("Distribution of trade results (R-multiples, clipped at -2 / +4)", loc="left", fontsize=8.5, color=C["ink"])
    return _png(fig, plt)


def chart_by_year(sig):
    fig, ax, plt = _fig(2.3)
    g = sig.groupby(sig["entry_date"].dt.year)["R"].mean()
    ax.bar(g.index.astype(str), g.values, color=C["sma50"], width=0.6)
    ax.axhline(0, color=C["ink2"], linewidth=0.7)
    lo_, hi_ = min(0, g.min()), max(0, g.max())
    ax.set_ylim(lo_ - 0.15 * (hi_ - lo_ or 1), hi_ + 0.15 * (hi_ - lo_ or 1))
    for x, v in zip(g.index.astype(str), g.values):
        ax.annotate(f"{v:+.2f}", xy=(x, v), xytext=(0, 3 if v >= 0 else -9), textcoords="offset points",
                    ha="center", fontsize=6.5, color=C["ink2"])
    ax.set_title("Expectancy (avg R per trade) by entry year", loc="left", fontsize=8.5, color=C["ink"])
    return _png(fig, plt)


# ── Report helpers ──────────────────────────────────────────────────────────
def f(v, nd=2, pct=False):
    if isinstance(v, str):
        return html.escape(v)
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    return f"{v:,.{nd}f}{'%' if pct else ''}"


def table(rows: list[tuple[str, dict]], cols: list[str], first="") -> str:
    head = f"<tr><th>{first}</th>" + "".join(f"<th class=r>{html.escape(c)}</th>" for c in cols) + "</tr>"
    body = "".join("<tr><td><b>" + html.escape(name) + "</b></td>" +
                   "".join(f"<td class=r>{f(d.get(c))}</td>" for c in cols) + "</tr>" for name, d in rows)
    return f"<table>{head}{body}</table>"


TRADE_COLS = ["Trades", "Win %", "Avg win R", "Avg loss R", "Expectancy R", "95% CI", "t-stat",
              "Profit factor", "Expectancy %", "Median days", "Target / Stop / Time %"]


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return diff, diff / se if se > 0 else np.nan


def verdict(P, Pc, is_df, oos_df, ev20) -> str:
    """Edge is judged in % return per trade (same capital per trade), not R,
    because random entries have much wider stops than zone entries."""
    d, t = welch(P["ret_pct"], Pc["ret_pct"])
    lines = [f"Primary rules: <b>{P['ret_pct'].mean():+.2f}%</b> per trade ({P['R'].mean():+.2f}R) over {len(P):,} trades; "
             f"random uptrend entries with the same exits: {Pc['ret_pct'].mean():+.2f}% per trade."]
    if np.isfinite(t) and t >= 2 and d > 0:
        lines.append(f"✅ Zone entries beat random entries by <b>{d:+.2f}% per trade</b> (t = {t:.1f}): statistically meaningful.")
    elif np.isfinite(d) and d > 0:
        lines.append(f"⚠️ Zone entries beat random entries by {d:+.2f}% per trade, but it is not statistically meaningful (t = {t:.1f}).")
    else:
        lines.append(f"❌ Zone entries did <b>not</b> beat random uptrend entries ({d:+.2f}% per trade, t = {t:.1f}). "
                     "The exits may make money, but the Fib-zone timing itself adds no measurable edge.")
    if ev20 is not None:
        lines.append(f"Event study (20 days after entry, vs SPY): zone entries {ev20[0]:+.2f}% vs every other uptrend day "
                     f"{ev20[1]:+.2f}% → difference {ev20[0] - ev20[1]:+.2f}% (t = {ev20[2]:.1f}).")
    ei, eo = is_df["ret_pct"].mean(), oos_df["ret_pct"].mean()
    if len(is_df) and len(oos_df):
        hold = "held up" if eo > 0 and eo >= 0.5 * ei else "weakened" if eo > 0 else "failed"
        lines.append(f"In-sample {ei:+.2f}% vs out-of-sample {eo:+.2f}% per trade: results <b>{hold}</b> on unseen data.")
    return "<br>".join(lines)


CSS = s.CSS + """
.verdict{background:#f6f6f8;border-left:4px solid #2a78d6;padding:10px 12px;font-size:13px;line-height:1.55}
"""


# ── Main ────────────────────────────────────────────────────────────────────
def main() -> int:
    rng = np.random.default_rng(SEED)
    data, spy, watch, notes, meta = load()
    if spy is None or not data:
        print("No data downloaded", file=sys.stderr)
        return 1
    print(f"Backtesting {len(data)} symbols")
    spy_c = spy["Close"]
    spy_up = (spy_c > spy_c.rolling(200).mean())

    sig_rows = {k: [] for k in [(a, b) for a in STOPS for b in TARGETS]}
    ctl_rows = {k: [] for k in sig_rows}
    fwd_sig, fwd_base = {k: [] for k in FWD}, {k: [] for k in FWD}

    for n_done, (t, df) in enumerate(data.items(), 1):
        r = s.analyze(df, history=True)
        hist = r["hist"]
        hist["rsi"] = s.rsi_wilder(df["Close"].to_numpy(float), s.RSI_LEN)
        hist["atr"] = atr(df)
        idx = df.index
        ok_time = idx >= SIGNAL_START
        bull = hist["bull_ok"].astype(bool) & ok_time
        sig = np.where(hist["entered"].astype(bool) & np.isin(hist["zone"], [1, 2]) & ok_time)[0]
        base = np.where(bull & (hist["zone"] == 0))[0]
        k = min(len(base), CONTROL_MULT * len(sig))
        ctl = np.sort(rng.choice(base, k, replace=False)) if k else np.array([], int)

        for (sk, tk) in sig_rows:
            sig_rows[(sk, tk)] += run_trades(t, df, hist, sig, STOPS[sk], TARGETS[tk], "signal")
            ctl_rows[(sk, tk)] += run_trades(t, df, hist, ctl, STOPS[sk], TARGETS[tk], "control")

        # Event study: entry next open -> close k days later, raw and minus SPY
        o, c = df["Open"].to_numpy(float), df["Close"].to_numpy(float)
        spy_al = spy_c.reindex(idx).ffill().to_numpy(float)
        spy_open = spy["Open"].reindex(idx).ffill().to_numpy(float)
        for bucket, bars in ((fwd_sig, sig), (fwd_base, base)):
            for hzn in FWD:
                b = bars[bars + hzn < len(c)]
                if len(b):
                    raw = c[b + hzn] / o[b + 1] - 1
                    mkt = spy_al[b + hzn] / spy_open[b + 1] - 1
                    bucket[hzn].append(np.column_stack([raw, raw - mkt]))
        if n_done % 100 == 0:
            print(f"  {n_done} symbols done")

    sig_df = {k: pd.DataFrame(v) for k, v in sig_rows.items()}
    ctl_df = {k: pd.DataFrame(v) for k, v in ctl_rows.items()}
    P, Pc = sig_df[PRIMARY], ctl_df[PRIMARY]
    if P.empty:
        print("No signals found", file=sys.stderr)
        return 1
    P["spy_up"] = P["signal_date"].map(lambda d: bool(spy_up.asof(d)))
    P["period"] = np.where(P["signal_date"] < OOS_START, "In-sample 2012–2019", f"Out-of-sample 2020–{dt.date.today().year}")
    P["universe"] = np.where(P["ticker"].isin(watch), "Watchlist", "Index")
    P["rsi_bucket"] = pd.cut(P["rsi"], [0, 30, 50, 100], labels=["RSI ≤30", "RSI 30–50", "RSI >50"])

    # Tables
    grid_rows = []
    for k in sig_df:
        grid_rows.append((f"{k[0]} / {k[1]} — zone entries", stats(sig_df[k])))
        grid_rows.append((f"{k[0]} / {k[1]} — random control", stats(ctl_df[k])))
    is_s = stats(P[P["period"].str.startswith("In")])
    oos_s = stats(P[P["period"].str.startswith("Out")])
    brk = [("All", stats(P)), ("In-sample 2012–2019", is_s), (f"Out-of-sample 2020+", oos_s)]
    brk += [(f"Zone: {z}", stats(P[P["zone"] == z])) for z in ("IBV", "SM")]
    brk += [("SPY above 200D", stats(P[P["spy_up"]])), ("SPY below 200D", stats(P[~P["spy_up"]]))]
    brk += [(str(b), stats(P[P["rsi_bucket"] == b])) for b in P["rsi_bucket"].cat.categories]
    brk += [(u, stats(P[P["universe"] == u])) for u in ("Index", "Watchlist")]

    ev_rows = []
    for hzn in FWD:
        sg = np.vstack(fwd_sig[hzn]) if fwd_sig[hzn] else np.empty((0, 2))
        bs = np.vstack(fwd_base[hzn]) if fwd_base[hzn] else np.empty((0, 2))
        ev_rows.append((f"{hzn} days", {
            "Signals": len(sg), "Signal avg %": 100 * sg[:, 0].mean() if len(sg) else np.nan,
            "Signal % positive": 100 * (sg[:, 0] > 0).mean() if len(sg) else np.nan,
            "Baseline avg %": 100 * bs[:, 0].mean() if len(bs) else np.nan,
            "Baseline % positive": 100 * (bs[:, 0] > 0).mean() if len(bs) else np.nan,
            "Signal vs SPY %": 100 * sg[:, 1].mean() if len(sg) else np.nan,
            "Baseline vs SPY %": 100 * bs[:, 1].mean() if len(bs) else np.nan,
        }))

    ev20 = None
    if fwd_sig.get(20) and fwd_base.get(20):
        a, b = np.vstack(fwd_sig[20])[:, 1] * 100, np.vstack(fwd_base[20])[:, 1] * 100
        _, tt = welch(a, b)
        ev20 = (a.mean(), b.mean(), tt)

    eq, pstats, taken = portfolio(P, data, spy)
    spy_eq = spy_c.reindex(eq.index).ffill()
    sstats = perf(spy_eq)

    images = [("bt-equity", chart_equity(eq, spy_eq)), ("bt-hist", chart_r_hist(P, Pc)), ("bt-year", chart_by_year(P))]
    p_sig, p_ctl = stats(P), stats(Pc)
    first, last = P["signal_date"].min().date(), P["signal_date"].max().date()

    body = f"""<html><head><style>{CSS}</style></head><body>
<h1 style="font-size:20px;margin-bottom:2px">IBV / SM Strategy Backtest</h1>
<p class="sub">{len(data)} symbols · signals {first} → {last} · entry next open · costs {COST_PCT}% round trip ·
time stop {TIME_STOP} days · one position per stock at a time</p>

<h2>Verdict</h2>
<div class="verdict">{verdict(P, Pc, P[P["period"].str.startswith("In")], P[P["period"].str.startswith("Out")], ev20)}</div>

<h2>1 · Exit rules vs. random-entry control</h2>
<p class="sub">R = profit ÷ initial risk, where risk = entry − stop but never less than 1×ATR(14) so very tight stops can't inflate R. Expectancy R / % = average result per trade; the 95% CI (on R) is bootstrapped.
The control buys random uptrend days (not in a zone) in the same stocks with the same exits — the zone must beat it.</p>
{table(grid_rows, TRADE_COLS, "Stop / Target")}

<h2>2 · Event study: what happens after a zone entry?</h2>
<p class="sub">Return from next open to the close N days later. Baseline = every other uptrend day in the same stocks.
"vs SPY" subtracts SPY's return over the same window.</p>
{table(ev_rows, list(ev_rows[0][1].keys()), "Horizon")}

<h2>3 · Primary rules ({PRIMARY[0]} stop / {PRIMARY[1]} target): breakdowns</h2>
{table(brk, TRADE_COLS, "Slice")}
<img class="chart" src="cid:bt-year" width="640">
<img class="chart" src="cid:bt-hist" width="640">

<h2>4 · Portfolio simulation vs SPY</h2>
<p class="sub">Max {MAX_POS} positions, {POS_PCT:.0%} of equity each, SM before IBV and lowest RSI first when signals compete;
{taken:,} of {len(P):,} trades taken. Marked to market daily.</p>
{table([("IBV/SM portfolio", pstats), ("SPY buy & hold", sstats)], list(pstats.keys()), "")}
<img class="chart" src="cid:bt-equity" width="640">

<h2>How to read this</h2>
<p class="sub" style="font-size:12.5px;line-height:1.55">
• <b>Edge</b> = zone result minus random-entry result, in % per trade (same capital per trade). Positive with t ≥ 2 is evidence; below that it is noise.
R is shown for risk context but is not used to judge the edge, because random entries sit much further above their stops.<br>
• <b>Out-of-sample</b> is the honest number — rules were not tuned on 2020+.<br>
• <b>Survivorship bias:</b> the universe is <i>today's</i> index members, so past losers that were dropped are missing.
This flatters both the strategy and the control; compare them to each other, not to zero.<br>
• Close-based stops can exit below the stop on gaps, so losses can exceed −1R.<br>
• Not investment advice. Full trade list attached (CSV).</p>
<p class="sub">Universe: {html.escape(' · '.join(notes))}</p>
</body></html>"""

    trades_csv = pd.concat([P.assign(rules=f"{PRIMARY[0]} / {PRIMARY[1]}")] +
                           [d.assign(rules=f"{k[0]} / {k[1]}") for k, d in sig_df.items() if k != PRIMARY],
                           ignore_index=True).drop(columns=["sig_bar"], errors="ignore").round({"entry": 4, "stop": 4, "target": 4, "exit": 4, "ret_pct": 3, "R": 3, "rsi": 1})
    csv_bytes = trades_csv.to_csv(index=False).encode()
    (BASE / "backtest_trades.csv").write_bytes(csv_bytes)
    subject = (f"IBV/SM Backtest: {P['ret_pct'].mean():+.2f}%/trade vs random {Pc['ret_pct'].mean():+.2f}% "
               f"· {p_sig['Trades']:,} trades · CAGR {pstats['CAGR %']:.1f}% vs SPY {sstats['CAGR %']:.1f}%")

    if os.environ.get("DRY_RUN") == "1" or not os.environ.get("EMAIL_USER"):
        for cid, png in images:
            (BASE / f"{cid}.png").write_bytes(png)
            body = body.replace(f"cid:{cid}", f"{cid}.png")
        (BASE / "backtest_report.html").write_text(body)
        print(f"DRY RUN: {subject}")
    else:
        (BASE / "backtest_report.html").write_text(body)
        s.send_email(subject, body, csv_bytes, images, csv_name="backtest_trades.csv")
        print(f"Emailed: {subject}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
