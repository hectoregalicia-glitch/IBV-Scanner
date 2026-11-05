#!/usr/bin/env python3
"""
IBV / SM Backtest v2 — exit-strategy comparison + "core holding" rule
---------------------------------------------------------------------
Same buy signal as scanner.py (daily close enters IBV 0.618-0.786 or SM 0.786-0.826 in a
bullish trend). What changes is HOW and WHEN we sell. Nine strategies are tested side by side,
each against a random-entry control that uses the identical selling rules.

  A  Current rules        stop: close < 0.826 · target: swing high
  B  Wider stop           stop: close < swing low · target: swing high
  C  Hold 60 days         stop: close < swing low · sell after 60 trading days
  D  Hold 120 days        stop: close < swing low · sell after 120 trading days
  E  Trail 50-day         stop: close < swing low · once price is back above its 50-day MA,
                          sell on the first close below it (no fixed target)
  F  Trail 3xATR          stop: close < swing low · once above the 50-day MA, sell on a close
                          more than 3 x ATR(14) below the highest close since entry
  G  Half + trail         stop: close < swing low · sell half at the swing high, trail the
                          other half with the 50-day rule (E)
  H  Runner +25% -> core  like E, but once a trade is up 25% it becomes a CORE holding:
                          all other rules switch off and it is sold only on a close below
                          its 200-day MA (the long-term trend breaking)
  I  Runner +50% -> core  same as H with a 50% threshold
  J  Structure + ATH trim  sell only on a break of structure (close below the latest higher low,
                          stepped up as higher lows form). Sell 25% at the first new all-time high;
                          after that each break sells another 25% and the net resets to the next
                          swing low. No break = no sale.
  K  ATH trims + weekly   daily structure net until the first new all-time high -> sell 25%; then
                          the net is the WEEKLY uptrend (weekly close below its latest higher low
                          sells everything); each new ATH 30+ days after the last trim sells 25%
                          of what is left. Repeat until stopped or the data ends.
  L  ATH trims + monthly  same as K with the MONTHLY uptrend as the net
  M  Claude's pick        entries only when SPY > 200-day MA and RSI <= 50 (control filtered the
                          same way); net at the swing low; at +25% or a new ATH sell 25% once,
                          then hold the rest until a weekly break of structure
  N  20% trailing stop    nothing but a broker-style trailing stop 20% below the highest price
                          since entry (triggers intraday; gaps fill at the open)

Fair-play rules
  - Entry next day's open; stops/trails act on the close and exit next open; 0.10% round trip.
  - One open trade per stock per strategy.
  - Trades still open at the end of the data are marked at the last close (not dropped),
    so long-holding strategies are not flattered.
  - Edge = strategy's average % per trade minus the random-entry control's (Welch t-test).
  - In-sample 2012-2019 vs out-of-sample 2020-today; entry filters (SPY above 200-day,
    RSI <= 50) reported as a slice of each strategy, not used to pick it.
  - Portfolio: max 10 positions x 10% of equity, marked to market daily, vs SPY.

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
DATA_START = "2011-01-01"
SIGNAL_START = "2012-01-01"
OOS_START = "2020-01-01"
COST_PCT = 0.10
CONTROL_MULT = 3
MAX_POS, POS_PCT, START_CAPITAL = 10, 0.10, 100_000
MAX_HOLD = 500                     # safety cap for non-core trades (~2 years)
FWD = (5, 10, 20, 60, 120)
INCLUDE_CRYPTO = False
SEED = 42

STRATS = {
    "A": dict(name="Current rules", stop="z3", target="hi"),
    "B": dict(name="Wider stop (swing low)", stop="leg_lo", target="hi"),
    "C": dict(name="Hold 60 days", stop="leg_lo", time=60),
    "D": dict(name="Hold 120 days", stop="leg_lo", time=120),
    "E": dict(name="Trail 50-day MA", stop="leg_lo", trail="ma50"),
    "F": dict(name="Trail 3x ATR", stop="leg_lo", trail="atr3"),
    "G": dict(name="Half at target + trail", stop="leg_lo", target="hi", half=True, trail="ma50"),
    "H": dict(name="Runner: +25% becomes core", stop="leg_lo", trail="ma50", core=0.25),
    "I": dict(name="Runner: +50% becomes core", stop="leg_lo", trail="ma50", core=0.50),
    "J": dict(name="Structure stop + trim at ATH", stop="leg_lo", special="bos_ath"),
    "K": dict(name="ATH trims + weekly structure stop", stop="leg_lo", special="htf", htf="w",
              pre="ratchet", repeat=True),
    "L": dict(name="ATH trims + monthly structure stop", stop="leg_lo", special="htf", htf="m",
              pre="ratchet", repeat=True),
    "M": dict(name="Claude's pick: filtered entry, weekly trail", stop="leg_lo", special="htf", htf="w",
              pre="static", switch_gain=0.25, repeat=False, filter=True),
    "N": dict(name="20% trailing stop only", stop="none", special="tpct", pct=0.20),
}
# Note: in the portfolio simulation a partly-sold trade (G, J) is held whole until its final sale at the
# blended price, which slightly overstates how long capital is tied up.
BASELINE = "A"

BASE = Path(__file__).resolve().parent
C = s.C


# ── Data ────────────────────────────────────────────────────────────────────
def load():
    universe, watch, notes, meta = s.build_universe()
    if not INCLUDE_CRYPTO:
        universe = [t for t in universe if not t.endswith("-USD")]
    s.HISTORY = "max"
    data = s.download(universe + ["SPY"])
    global PRIOR_ATH
    PRIOR_ATH = {t: float(d.loc[d.index < DATA_START, "Close"].max()) if (d.index < DATA_START).any() else -np.inf
                 for t, d in data.items()}
    data = {t: d[d.index >= DATA_START] for t, d in data.items()}
    data = {t: d for t, d in data.items() if len(d) > s.SMA_LEN + 50}
    spy = data.pop("SPY", None)
    return data, spy, set(watch), notes, meta


PRIOR_ATH: dict[str, float] = {}


def pivot_lows(lo: np.ndarray, L: int) -> np.ndarray:
    """pl[j] = value of a swing low CONFIRMED on bar j (it sits at j-L), else NaN. Same rule as scanner."""
    out = np.full(len(lo), np.nan)
    for j in range(2 * L, len(lo)):
        ci = j - L
        if lo[ci] <= lo[j - 2 * L:j + 1].min() and lo[ci] < lo[j - 2 * L:ci].min():
            out[j] = lo[ci]
    return out


def atr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    h, l, c = (df[k] for k in ("High", "Low", "Close"))
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean().to_numpy()


# ── One trade under one strategy ────────────────────────────────────────────
def simulate(px: dict, e: int, stop: float, target: float | None, cfg: dict):
    """Returns dict(exit_bar, exit_px, reason, core, half_px) — exit_bar None = still open."""
    o, h, c = px["o"], px["h"], px["c"]
    ma50, ma200, a = px["ma50"], px["ma200"], px["atr"]
    n = len(c)
    entry = o[e]
    trail, time_exit, core_thr, half = cfg.get("trail"), cfg.get("time"), cfg.get("core"), cfg.get("half", False)
    armed = False            # trail activates after a close above the 50-day MA
    core = False
    peak = c[e]
    half_px = None

    def nxt(j):
        return (j + 1, o[j + 1]) if j + 1 < n else (None, c[j])

    for j in range(e, n):
        peak = max(peak, c[j])
        if core:
            if np.isfinite(ma200[j]) and c[j] < ma200[j]:
                xb, xp = nxt(j)
                return dict(exit_bar=xb, exit_px=xp, reason="core exit (200D)", core=True, half_px=half_px)
            continue
        # hard stop
        if c[j] < stop:
            xb, xp = nxt(j)
            return dict(exit_bar=xb, exit_px=xp, reason="stop", core=False, half_px=half_px)
        # fixed target (full or half)
        if target is not None and half_px is None:
            hit = o[j] if o[j] >= target else (target if h[j] >= target else None)
            if hit is not None:
                if half:
                    half_px = hit
                else:
                    return dict(exit_bar=j, exit_px=hit, reason="target", core=False, half_px=None)
        # trailing exits
        if trail:
            if not armed and np.isfinite(ma50[j]) and c[j] > ma50[j]:
                armed = True
            if armed and (target is None or not half or half_px is not None):
                out = False
                if trail == "ma50" and np.isfinite(ma50[j]) and c[j] < ma50[j]:
                    out = True
                if trail == "atr3" and np.isfinite(a[j]) and c[j] < peak - 3 * a[j]:
                    out = True
                if out:
                    xb, xp = nxt(j)
                    return dict(exit_bar=xb, exit_px=xp, reason="trail", core=False, half_px=half_px)
        # time exit
        if time_exit and j - e + 1 >= time_exit:
            return dict(exit_bar=j, exit_px=c[j], reason="time", core=False, half_px=half_px)
        # promote to core
        if core_thr and c[j] >= entry * (1 + core_thr):
            core = True
            continue
        if j - e + 1 >= MAX_HOLD:
            return dict(exit_bar=j, exit_px=c[j], reason="max hold", core=False, half_px=half_px)
    return dict(exit_bar=None, exit_px=c[-1], reason="open", core=core, half_px=half_px)


HTF_PIVOT = {"w": 3, "m": 2}          # swing confirmation: 3 weeks / 2 months each side


def htf_arrays(df: pd.DataFrame, which: str):
    """Weekly ('w') or monthly ('m') structure mapped onto daily bars.
    Returns (pl, last_pl, is_end): pl[j] = weekly/monthly swing low confirmed on day j;
    last_pl[j] = most recent confirmed one as of day j; is_end[j] = last trading day of the week/month."""
    key = df.index.to_period("W-FRI" if which == "w" else "M")
    pos = pd.Series(np.arange(len(df)), index=df.index)
    g = pd.DataFrame({"low": df["Low"].to_numpy(), "pos": pos.to_numpy()}, index=df.index).groupby(key)
    lows = g["low"].min().to_numpy()
    ends = g["pos"].max().to_numpy()
    L = HTF_PIVOT[which]
    pl = np.full(len(df), np.nan)
    for k in range(2 * L, len(lows)):
        ci = k - L
        if lows[ci] <= lows[k - 2 * L:k + 1].min() and lows[ci] < lows[k - 2 * L:ci].min():
            pl[ends[k]] = lows[ci]
    is_end = np.zeros(len(df), bool)
    is_end[ends] = True
    last_pl = pd.Series(pl).ffill().to_numpy()
    return pl, last_pl, is_end


def simulate_htf(px: dict, e: int, stop: float, cfg: dict):
    """K / L / M. Before the switch: daily safety net (stepped up with higher lows for K/L,
    fixed at the swing low for M); a close below it sells everything.
    Switch (K/L: first new all-time high; M: first new ATH or +25%): trim 25%, then the safety
    net becomes the weekly/monthly structure (a week/month CLOSE below its latest higher low
    sells everything). K/L: each later new ATH at least 30 days after the last trim sells 25%
    of what is left."""
    o, c, ath, dts = px["o"], px["c"], px["ath_before"], px["d"]
    pl_d = px["pl"]
    pl_h, last_h, is_end = px[f"htf_{cfg['htf']}"]
    n = len(c)
    entry = o[e]
    rem, sales = 1.0, []
    struct, mode, last_trim, hstruct = stop, "daily", None, None
    month = np.timedelta64(30, "D")

    def done(xb, xp, why):
        sales.append((rem, xp))
        return dict(exit_bar=xb, exit_px=sum(f * p for f, p in sales), reason=why,
                    core=mode == "htf", half_px=None)

    for j in range(e, n):
        if mode == "daily":
            if cfg.get("pre") == "ratchet" and np.isfinite(pl_d[j]) and pl_d[j] > struct:
                struct = pl_d[j]
        elif np.isfinite(pl_h[j]) and (hstruct is None or pl_h[j] > hstruct):
            hstruct = pl_h[j]
        new_high = c[j] > ath[j]
        if mode == "daily":
            gain_hit = cfg.get("switch_gain") and c[j] >= entry * (1 + cfg["switch_gain"])
            if new_high or gain_hit:
                sales.append((0.25 * rem, c[j]))
                rem *= 0.75
                mode, last_trim = "htf", j
                hstruct = last_h[j] if np.isfinite(last_h[j]) else struct
                continue
            if c[j] < struct:
                xb, xp = (j + 1, o[j + 1]) if j + 1 < n else (j, c[j])
                return done(xb, xp, "stop")
        else:
            if cfg.get("repeat") and new_high and dts[j] - dts[last_trim] >= month:
                sales.append((0.25 * rem, c[j]))
                rem *= 0.75
                last_trim = j
                continue
            if is_end[j] and hstruct is not None and c[j] < hstruct:
                xb, xp = (j + 1, o[j + 1]) if j + 1 < n else (j, c[j])
                return done(xb, xp, f"{'weekly' if cfg['htf'] == 'w' else 'monthly'} structure break")
    r = done(None, c[-1], "open (partly sold)" if mode == "htf" else "open")
    return r


def simulate_tpct(px: dict, e: int, pct: float):
    """N: a broker-style trailing stop pct below the highest price since entry. It triggers
    intraday when the low touches it (fill at the stop, or at the open if the stock gaps below)."""
    o, h, lo, c = px["o"], px["h"], px["lo"], px["c"]
    peak = o[e]
    for j in range(e, len(c)):
        level = peak * (1 - pct)
        if lo[j] <= level:
            return dict(exit_bar=j, exit_px=min(o[j], level), reason="trailing stop", core=False, half_px=None)
        peak = max(peak, h[j])
    return dict(exit_bar=None, exit_px=c[-1], reason="open", core=False, half_px=None)


def simulate_j(px: dict, e: int, stop: float):
    """J: break-of-structure stop that steps up with higher lows; trim 25% at the first new
    all-time high; after that each break of structure sells another 25% of the original
    position and the safety net resets to the next swing low that forms."""
    o, c, pl, ath = px["o"], px["c"], px["pl"], px["ath_before"]
    n = len(c)
    rem, sales = 1.0, []          # sales = [(fraction, price)]
    struct, trimmed, last_bar = stop, False, None
    for j in range(e, n):
        if np.isfinite(pl[j]):                      # a new swing low just confirmed
            if struct is None:
                struct = pl[j]                       # reset after a break
            elif pl[j] > struct:
                struct = pl[j]                       # higher low -> step the net up
        if not trimmed and c[j] > ath[j]:           # first new all-time high close
            sales.append((0.25, c[j]))
            rem -= 0.25
            trimmed, last_bar = True, j
            continue
        if struct is not None and c[j] < struct:    # break of structure
            xb, xp = (j + 1, o[j + 1]) if j + 1 < n else (j, c[j])
            frac = rem if not trimmed else min(0.25, rem)
            sales.append((frac, xp))
            rem -= frac
            last_bar = xb
            if rem <= 1e-9:
                return dict(exit_bar=xb, exit_px=sum(f * p for f, p in sales), reason="structure break",
                            core=trimmed, half_px=None)
            struct = None                           # wait for the next swing low
    sales.append((rem, c[-1]))
    return dict(exit_bar=None, exit_px=sum(f * p for f, p in sales),
                reason="open (partly sold)" if trimmed else "open", core=trimmed, half_px=None)


def run_trades(t, df, hist, px, entries, key, kind):
    cfg = STRATS[key]
    dates = df.index
    n = len(df)
    rows, busy_until = [], -1
    for i in entries:
        e = i + 1
        if e >= n or e <= busy_until:
            continue
        if cfg.get("filter") and not (px["spy_up"][i] and hist["rsi"][i] <= 50):
            continue
        hi, lo_, z3 = hist["leg_hi"][i], hist["leg_lo"][i], hist["z3"][i]
        stop = z3 if cfg["stop"] == "z3" else (-np.inf if cfg["stop"] == "none" else lo_)
        target = {"hi": hi, None: None}[cfg.get("target")]
        entry = px["o"][e] * (1 + COST_PCT / 200)
        if not np.isfinite(stop) or entry <= stop or (target is not None and entry >= target):
            continue
        sp = cfg.get("special")
        if sp == "bos_ath":
            r = simulate_j(px, e, stop)
        elif sp == "htf":
            r = simulate_htf(px, e, stop, cfg)
        elif sp == "tpct":
            r = simulate_tpct(px, e, cfg["pct"])
        else:
            r = simulate(px, e, stop, target, cfg)
        xp = r["exit_px"]
        if r["half_px"] is not None:
            xp = 0.5 * r["half_px"] + 0.5 * xp
        xp *= (1 - COST_PCT / 200)
        xb = r["exit_bar"]
        busy_until = xb if xb is not None else n
        last = xb if xb is not None else n - 1
        rows.append({
            "strategy": key, "ticker": t, "kind": kind,
            "zone": s.ZONE_NAMES.get(int(hist["zone"][i]), ""),
            "signal_date": dates[i], "entry_date": dates[e],
            "exit_date": dates[xb] if xb is not None else pd.NaT,
            "entry": entry, "stop": stop, "exit": xp, "reason": r["reason"],
            "became_core": bool(r["core"]), "open": xb is None,
            "days": last - e + 1, "ret_pct": (xp / entry - 1) * 100, "rsi": hist["rsi"][i],
        })
    return rows


# ── Stats ───────────────────────────────────────────────────────────────────
def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan
    d = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return d, d / se if se > 0 else np.nan


def trade_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"Trades": 0}
    r = df["ret_pct"].to_numpy()
    srt = np.sort(r)[::-1]
    top = srt[: max(1, len(r) // 10)].sum()
    pos = r[r > 0].sum()
    return {
        "Trades": len(r),
        "Avg % / trade": r.mean(),
        "Median %": np.median(r),
        "Win %": 100 * (r > 0).mean(),
        "Avg winner %": r[r > 0].mean() if (r > 0).any() else np.nan,
        "Avg loser %": r[r <= 0].mean() if (r <= 0).any() else np.nan,
        "≥ +50% trades": 100 * (r >= 50).mean(),
        "Runner % (core / ATH trim)": 100 * df["became_core"].mean(),
        "Median days": float(np.median(df["days"])),
        "Top 10% share of gains": 100 * top / pos if pos > 0 else np.nan,
    }


def portfolio(trades: pd.DataFrame, closes: dict, days: pd.DatetimeIndex) -> tuple[pd.Series, int]:
    tr = trades.copy()
    tr["prio"] = tr["zone"].map({"SM": 0, "IBV": 1}).fillna(2)
    tr = tr.sort_values(["entry_date", "prio", "rsi"])
    by_entry = {d: g for d, g in tr.groupby("entry_date")}
    cash, pos, curve, taken, equity = START_CAPITAL, [], [], 0, START_CAPITAL
    for d in days:
        keep = []
        for p in pos:
            if pd.notna(p["exit_date"]) and p["exit_date"] <= d:
                cash += p["shares"] * p["exit"]
            else:
                keep.append(p)
        pos = keep
        for _, r in by_entry.get(d, pd.DataFrame()).iterrows():
            if len(pos) >= MAX_POS:
                break
            alloc = min(POS_PCT * equity, cash)
            if alloc <= 0:
                break
            cash -= alloc
            pos.append({"t": r["ticker"], "shares": alloc / r["entry"], "exit": r["exit"], "exit_date": r["exit_date"]})
            taken += 1
        mv = 0.0
        for p in pos:
            v = closes[p["t"]].asof(d)
            mv += p["shares"] * (v if v == v else 0)
        equity = cash + mv
        curve.append(equity)
    return pd.Series(curve, index=days), taken


def perf(eq: pd.Series) -> dict:
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    rets = eq.pct_change().dropna()
    return {"CAGR %": 100 * ((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1),
            "Max drawdown %": 100 * (eq / eq.cummax() - 1).min(),
            "Sharpe": rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else np.nan}


# ── Charts ──────────────────────────────────────────────────────────────────
def _fig(h=2.8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5})
    fig, ax = plt.subplots(figsize=(6.4, h), dpi=200)
    ax.set_axisbelow(True)
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


def chart_equity(curves: dict, spy_eq: pd.Series, best: str):
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    fig, ax, plt = _fig(3.0)
    for k, eq in curves.items():
        a = 100 * eq / eq.iloc[0]
        hl = k in (best, BASELINE)
        ax.plot(a.index, a, color=C["sma50"] if k == best else C["ink2"] if k == BASELINE else C["grid"],
                linewidth=1.5 if hl else 0.8, zorder=3 if hl else 1,
                label=f"{k} {STRATS[k]['name']}" if hl else None)
        if hl:
            ax.annotate(f"{k} {a.iloc[-1]:.0f}", xy=(a.index[-1], a.iloc[-1]), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=6.8, color=C["ink2"])
    b = 100 * spy_eq / spy_eq.iloc[0]
    ax.plot(b.index, b, color=C["sma200"], linewidth=1.3, label="SPY buy & hold", zorder=2)
    ax.annotate(f"SPY {b.iloc[-1]:.0f}", xy=(b.index[-1], b.iloc[-1]), xytext=(4, 0),
                textcoords="offset points", va="center", fontsize=6.8, color=C["ink2"])
    ax.axvline(pd.Timestamp(OOS_START), color=C["muted"], linewidth=0.7, linestyle=(0, (3, 3)))
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.legend(loc="upper left", frameon=False, fontsize=7, labelcolor=C["ink2"])
    ax.set_title("Growth of $100 — best strategy vs current rules vs SPY (other strategies in light grey)",
                 loc="left", fontsize=8.2, color=C["ink"])
    return _png(fig, plt)


def chart_fwd(fwd_sig, fwd_base):
    fig, ax, plt = _fig(2.4)
    hz = [h for h in FWD if fwd_sig.get(h)]
    diff = [100 * (np.vstack(fwd_sig[h])[:, 1].mean() - np.vstack(fwd_base[h])[:, 1].mean()) for h in hz]
    xs = [f"{h} days" for h in hz]
    ax.bar(xs, diff, color=C["sma50"], width=0.55)
    ax.axhline(0, color=C["ink2"], linewidth=0.7)
    for x, v in zip(xs, diff):
        ax.annotate(f"{v:+.2f}%", xy=(x, v), xytext=(0, 3 if v >= 0 else -10), textcoords="offset points",
                    ha="center", fontsize=7, color=C["ink2"])
    lo_, hi_ = min(0, min(diff)), max(0, max(diff))
    pad = 0.18 * ((hi_ - lo_) or 1)
    ax.set_ylim(lo_ - pad, hi_ + pad)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.2f}%")
    ax.set_title("Zone advantage by holding period (zone entries minus other uptrend days, net of SPY)",
                 loc="left", fontsize=8.2, color=C["ink"], pad=10)
    return _png(fig, plt)


# ── Report helpers ──────────────────────────────────────────────────────────
def f(v, nd=2):
    if isinstance(v, str):
        return html.escape(v)
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    return f"{v:,.{nd}f}"


def table(rows, cols, first=""):
    head = f"<tr><th>{first}</th>" + "".join(f"<th class=r>{html.escape(c)}</th>" for c in cols) + "</tr>"
    body = "".join("<tr><td><b>" + html.escape(nm) + "</b></td>" +
                   "".join(f"<td class=r>{f(d.get(c))}</td>" for c in cols) + "</tr>" for nm, d in rows)
    return f"<table>{head}{body}</table>"


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
    print(f"Backtesting {len(data)} symbols x {len(STRATS)} strategies")
    spy_c = spy["Close"]
    spy_up = spy_c > spy_c.rolling(200).mean()

    sig_rows, ctl_rows = [], []
    fwd_sig, fwd_base = {k: [] for k in FWD}, {k: [] for k in FWD}

    for n_done, (t, df) in enumerate(data.items(), 1):
        hist = s.analyze(df, history=True)["hist"]
        c = df["Close"].to_numpy(float)
        hist["rsi"] = s.rsi_wilder(c, s.RSI_LEN)
        px = {"o": df["Open"].to_numpy(float), "h": df["High"].to_numpy(float), "lo": df["Low"].to_numpy(float), "c": c,
              "ma50": pd.Series(c).rolling(50).mean().to_numpy(),
              "ma200": pd.Series(c).rolling(200).mean().to_numpy(), "atr": atr(df),
              "pl": pivot_lows(df["Low"].to_numpy(float), s.PIVOT_LEN)}
        px["d"] = df.index.to_numpy()
        px["htf_w"], px["htf_m"] = htf_arrays(df, "w"), htf_arrays(df, "m")
        px["spy_up"] = spy_up.reindex(df.index).ffill().fillna(False).to_numpy(bool)
        prev_max = pd.Series(c).cummax().shift(1).to_numpy()
        px["ath_before"] = np.fmax(np.nan_to_num(prev_max, nan=-np.inf), PRIOR_ATH.get(t, -np.inf))
        ok_time = df.index >= SIGNAL_START
        bull = hist["bull_ok"].astype(bool) & ok_time
        sig = np.where(hist["entered"].astype(bool) & np.isin(hist["zone"], [1, 2]) & ok_time)[0]
        base = np.where(bull & (hist["zone"] == 0))[0]
        k = min(len(base), CONTROL_MULT * len(sig))
        ctl = np.sort(rng.choice(base, k, replace=False)) if k else np.array([], int)
        for key in STRATS:
            sig_rows += run_trades(t, df, hist, px, sig, key, "signal")
            ctl_rows += run_trades(t, df, hist, px, ctl, key, "control")
        o = px["o"]
        spy_cl = spy_c.reindex(df.index).ffill().to_numpy(float)
        spy_op = spy["Open"].reindex(df.index).ffill().to_numpy(float)
        for bucket, bars in ((fwd_sig, sig), (fwd_base, base)):
            for hz in FWD:
                b = bars[bars + hz < len(c)]
                if len(b):
                    raw = c[b + hz] / o[b + 1] - 1
                    bucket[hz].append(np.column_stack([raw, raw - (spy_cl[b + hz] / spy_op[b + 1] - 1)]))
        if n_done % 100 == 0:
            print(f"  {n_done} symbols done")

    S, K = pd.DataFrame(sig_rows), pd.DataFrame(ctl_rows)
    if S.empty:
        print("No signals found", file=sys.stderr)
        return 1
    S["oos"] = S["signal_date"] >= OOS_START
    S["spy_up"] = S["signal_date"].map(lambda d: bool(spy_up.asof(d)))
    S["filtered"] = S["spy_up"] & (S["rsi"] <= 50)

    closes = {t: d["Close"] for t, d in data.items()}
    days = spy.index[spy.index >= SIGNAL_START]
    spy_eq = spy_c.reindex(days).ffill()
    spy_perf = perf(spy_eq)

    summary, curves, detail = [], {}, []
    for key, cfg in STRATS.items():
        g, gc = S[S["strategy"] == key], K[K["strategy"] == key]
        edge, t = welch(g["ret_pct"], gc["ret_pct"])
        eq, taken = portfolio(g, closes, days)
        curves[key] = eq
        p = perf(eq)
        row = {
            "Trades": len(g), "Avg % / trade": g["ret_pct"].mean(), "Random entries %": gc["ret_pct"].mean(),
            "Edge %": edge, "Edge t": t,
            "2012–19 %": g.loc[~g["oos"], "ret_pct"].mean(), "2020+ %": g.loc[g["oos"], "ret_pct"].mean(),
            "Filtered %": g.loc[g["filtered"], "ret_pct"].mean(),
            "CAGR %": p["CAGR %"], "Max DD %": p["Max drawdown %"], "Sharpe": p["Sharpe"],
        }
        summary.append((key, row))
        detail.append((f"{key} {cfg['name']}", trade_stats(g)))

    # Best = highest portfolio Sharpe among strategies whose edge is significant and positive in both periods
    def score(item):
        k, r = item
        ok = (r["Edge t"] or 0) >= 2 and (r["2012–19 %"] or 0) > 0 and (r["2020+ %"] or 0) > 0
        return (ok, r["Sharpe"] if np.isfinite(r["Sharpe"]) else -9)
    best_key, best = max(summary, key=score)
    best_ok = score((best_key, best))[0]
    base_row = dict(summary)[BASELINE]

    lines = [
        f"<b>{'Best strategy' if best_ok else 'Highest Sharpe ratio'}: {best_key} — {STRATS[best_key]['name']}.</b> "
        f"{best['Avg % / trade']:+.2f}% per trade vs {best['Random entries %']:+.2f}% for random entries "
        f"(edge {best['Edge %']:+.2f}%, t = {best['Edge t']:.1f}); 2012–19 {best['2012–19 %']:+.2f}%, 2020+ {best['2020+ %']:+.2f}%.",
        f"Portfolio: {best['CAGR %']:.1f}% a year, worst drop {best['Max DD %']:.1f}%, Sharpe {best['Sharpe']:.2f} — "
        f"vs current rules {base_row['CAGR %']:.1f}% / {base_row['Max DD %']:.1f}% / {base_row['Sharpe']:.2f} "
        f"and SPY {spy_perf['CAGR %']:.1f}% / {spy_perf['Max drawdown %']:.1f}% / {spy_perf['Sharpe']:.2f}.",
    ]
    if not best_ok:
        lines.append("⚠️ No strategy had a statistically meaningful edge that was positive in both periods; "
                     "the pick above is simply the highest Sharpe ratio. Treat it as unproven.")
    beats_spy = best["Sharpe"] > spy_perf["Sharpe"] and best["CAGR %"] > spy_perf["CAGR %"]
    lines.append("✅ It beat SPY on both return and risk-adjusted return." if beats_spy else
                 "❌ It did not beat SPY on both return and risk-adjusted return — buying and holding SPY was as good or better.")

    cols = ["Trades", "Avg % / trade", "Random entries %", "Edge %", "Edge t", "2012–19 %", "2020+ %",
            "Filtered %", "CAGR %", "Max DD %", "Sharpe"]
    srows = [(f"{k} {STRATS[k]['name']}", r) for k, r in summary] + [("SPY buy & hold", {
        "CAGR %": spy_perf["CAGR %"], "Max DD %": spy_perf["Max drawdown %"], "Sharpe": spy_perf["Sharpe"]})]
    dcols = list(detail[0][1].keys())

    images = [("bt-equity", chart_equity(curves, spy_eq, best_key)), ("bt-fwd", chart_fwd(fwd_sig, fwd_base))]
    first_d, last_d = S["signal_date"].min().date(), S["signal_date"].max().date()

    rules_html = "".join(
        f"<tr><td><b>{k}</b></td><td>{html.escape(v['name'])}</td><td>{html.escape(desc)}</td></tr>"
        for (k, v), desc in zip(STRATS.items(), [
            "Sell on a close below 0.826, or at the swing high",
            "Sell on a close below the swing low, or at the swing high",
            "Sell on a close below the swing low, or after 60 trading days",
            "Sell on a close below the swing low, or after 120 trading days",
            "Sell on a close below the swing low; once back above the 50-day MA, sell on the first close below it",
            "Sell on a close below the swing low; once above the 50-day MA, sell on a close 3×ATR below the highest close",
            "Sell half at the swing high; sell the rest with the 50-day rule (E)",
            "Like E, but once up 25% it becomes a core holding: sold only on a close below the 200-day MA",
            "Like H, with a 50% threshold",
            "Only sells on a break of the uptrend (close below the latest higher low). Sell 25% at the first new all-time high; "
            "after that, each break sells another 25% and the net resets to the next swing low",
            "Daily structure net until the first new all-time high: sell 25%, then the net becomes the WEEKLY uptrend "
            "(weekly close below its latest higher low sells everything). Each new high 30+ days after the last trim sells 25% of what is left",
            "Same as K, but the net after the first trim is the MONTHLY uptrend",
            "Only buys when SPY is above its 200-day MA and RSI ≤ 50. Net at the swing low; at +25% or a new all-time high "
            "sell 25% once, then hold the rest until a weekly break of structure",
            "No other rules: a trailing stop 20% below the highest price since buying; sells only when it is hit",
        ]))

    body = f"""<html><head><style>{CSS}</style></head><body>
<h1 style="font-size:20px;margin-bottom:2px">IBV / SM Backtest v2 — When to sell</h1>
<p class="sub">{len(data)} stocks · signals {first_d} → {last_d} · same buy signal for every strategy · entry next open ·
{COST_PCT}% round-trip costs · open trades marked at the last close</p>

<h2>Verdict</h2>
<div class="verdict">{'<br>'.join(lines)}</div>

<h2>1 · All strategies side by side</h2>
<p class="sub">Edge = average % per trade minus random uptrend entries with the same selling rules (t ≥ 2 = meaningful).
"Filtered" = only signals with SPY above its 200-day MA and RSI ≤ 50. CAGR / Max DD / Sharpe = portfolio of max
{MAX_POS} positions at {POS_PCT:.0%} each.</p>
{table(srows, cols, "Strategy")}
<img class="chart" src="cid:bt-equity" width="640">

<h2>2 · What the trades look like</h2>
<p class="sub">"Runner %" = share of trades that became core holdings (H, I), trimmed at a new all-time high (J, K, L) or switched to the weekly net (M). "Top 10% share of gains" = how much of
all profit came from the best tenth of trades (higher = more reliant on big winners).</p>
{table(detail, dcols, "Strategy")}

<h2>3 · Does the zone keep working the longer you hold?</h2>
<img class="chart" src="cid:bt-fwd" width="640">

<h2>4 · The strategies</h2>
<table><tr><th></th><th>Name</th><th>Selling rule (all buy on the next open after a zone close)</th></tr>{rules_html}</table>

<p class="sub" style="margin-top:18px;font-size:12.5px;line-height:1.55">
• Pick a strategy that is good in <b>both</b> periods and beats random entries — not just the biggest number.<br>
• Survivorship bias: the stock list is today's index members, which flatters long holds most. Compare strategies to
each other and to SPY, and expect less in live trading.<br>
• Not investment advice. Full trade list attached (CSV).</p>
<p class="sub">Universe: {html.escape(' · '.join(notes))}</p>
</body></html>"""

    out = S.drop(columns=["spy_up"]).round({"entry": 4, "stop": 4, "exit": 4, "ret_pct": 3, "rsi": 1})
    csv_bytes = out.to_csv(index=False).encode()
    (BASE / "backtest_trades.csv").write_bytes(csv_bytes)
    subject = (f"IBV/SM Backtest v2: {'best' if best_ok else 'top Sharpe (unproven)'} = {best_key} {STRATS[best_key]['name']} · "
               f"CAGR {best['CAGR %']:.1f}% vs SPY {spy_perf['CAGR %']:.1f}% · Sharpe {best['Sharpe']:.2f} vs {spy_perf['Sharpe']:.2f}")

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
