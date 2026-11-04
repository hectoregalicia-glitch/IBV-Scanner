#!/usr/bin/env python3
"""
IBV / SM Nightly Scanner
------------------------
Python port of the TradingView "IBV / SM Zone Scanner" indicator.

Bullish structure (higher high, no lower high since, close > 200 SMA):
    IBV = Institutional Buy Zone  -> 0.618 to 0.786 retrace of swing low -> swing high
    SM  = Smart Money Buy Zone    -> 0.786 to 0.826 retrace

Each run:
  1. Builds the universe: S&P 500 + Nasdaq-100 (from Wikipedia) + my_watchlist.txt
  2. Downloads daily prices from Yahoo Finance (yfinance)
  3. Runs the same swing / Fib / trend logic as the Pine script
  4. Emails an HTML report (+ CSV of every symbol) via Gmail

Environment variables (set as GitHub Secrets):
  EMAIL_USER          Gmail address that sends the report
  EMAIL_APP_PASSWORD  16-character Gmail App Password
  EMAIL_TO            Optional: recipient(s), comma-separated. Defaults to EMAIL_USER
  DRY_RUN=1           Optional: write report.html / results.csv instead of emailing
"""
from __future__ import annotations

import datetime as dt
import html
import os
import re
import smtplib
import ssl
import sys
from email.message import EmailMessage
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ── Settings (match the TradingView indicator inputs) ──────────────────────
PIVOT_LEN = 10        # swing confirmation: bars on each side
MIN_MOVE = 0.15       # minimum swing size (15%)
FIB_A = 0.618         # IBV start
FIB_B = 0.786         # IBV end / SM start
FIB_C = 0.826         # SM end
SMA_LEN = 200         # trend filter
USE_SMA = True        # require close above SMA for bullish setups
TRIGGER = "close"     # "close" or "wick"
NEAR_PCT = 3.0        # "approaching" = within this % above the 0.618 level
RSI_LEN = 14          # RSI period (Wilder, same as TradingView)
MAX_CHARTS = 20       # charts embedded per email (new + in-zone setups first)
CHART_BARS = 260      # ~1 year of daily bars per chart (extended to show the swing)
HISTORY = "5y"        # daily history to download
CHUNK = 100           # tickers per Yahoo request

BASE = Path(__file__).resolve().parent
WATCHLIST_FILE = BASE / "my_watchlist.txt"
EXTRA_INDEX_FILE = BASE / "index_tickers.txt"   # optional fallback universe

INDEX_SOURCES = {
    "S&P 500": ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", ["Symbol"]),
    "Nasdaq-100": ("https://en.wikipedia.org/wiki/Nasdaq-100", ["Ticker", "Symbol"]),
}

ZONE_NAMES = {1: "IBV", 2: "SM", -1: "IBS", -2: "SMS", 0: ""}


# ── Universe ────────────────────────────────────────────────────────────────
def normalize(t: str) -> str:
    t = t.strip().upper()
    # Class shares: BRK.B -> BRK-B (Yahoo format). Exchange suffixes like SHOP.TO stay.
    if re.fullmatch(r"[A-Z]+\.[A-Z]", t):
        t = t.replace(".", "-")
    return t


def read_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        t = line.split("#", 1)[0].strip()
        if t:
            out.append(normalize(t))
    return out


def _find_col(columns, *keys, exclude=()):
    for col in columns:
        name = str(col).lower()
        if any(k in name for k in keys) and not any(x in name for x in exclude):
            return col
    return None


def fetch_index(url: str, cols: list[str]) -> tuple[list[str], dict[str, dict]]:
    """Returns (tickers, meta) where meta[ticker] = {name, sector, industry}."""
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; IBV-SM-Scanner/1.0)"}, timeout=30)
    r.raise_for_status()
    for table in pd.read_html(StringIO(r.text)):
        if len(table) < 90:
            continue
        if isinstance(table.columns, pd.MultiIndex):
            table.columns = [" ".join(map(str, c)) for c in table.columns]
        tcol = _find_col(table.columns, *[c.lower() for c in cols], "ticker", "symbol")
        if tcol is None:
            continue
        ncol = _find_col(table.columns, "security", "company", "name")
        scol = _find_col(table.columns, "sector")
        icol = _find_col(table.columns, "sub-industry", "industry")
        tickers, meta = [], {}
        for _, row in table.iterrows():
            t = normalize(str(row[tcol]))
            if not re.fullmatch(r"[A-Z]{1,5}(-[A-Z])?", t):
                continue
            tickers.append(t)
            meta[t] = {
                "name": str(row[ncol]) if ncol is not None else "",
                "sector": str(row[scol]) if scol is not None else "",
                "industry": str(row[icol]) if icol is not None else "",
            }
        if len(tickers) >= 90:
            return tickers, meta
    raise ValueError("ticker table not found")


def fetch_meta_yahoo(tickers: list[str]) -> dict[str, dict]:
    """Company name / industry for symbols not covered by the index tables."""
    import yfinance as yf

    out = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).get_info() or {}
        except Exception:  # noqa: BLE001
            info = {}
        name = info.get("longName") or info.get("shortName") or ""
        if t.endswith("-USD"):
            out[t] = {"name": name or t.replace("-USD", ""), "sector": "Crypto", "industry": "Cryptocurrency"}
        else:
            out[t] = {"name": name, "sector": info.get("sector") or "", "industry": info.get("industry") or ""}
    return out


def build_universe() -> tuple[list[str], list[str], list[str], dict[str, dict]]:
    """Returns (all_tickers, watchlist, notes, meta)."""
    notes, index, meta = [], [], {}
    for name, (url, cols) in INDEX_SOURCES.items():
        try:
            tick, m = fetch_index(url, cols)
            index += tick
            for k, v in m.items():
                meta.setdefault(k, v)
            notes.append(f"{name}: {len(tick)} symbols")
        except Exception as e:  # noqa: BLE001
            notes.append(f"{name}: could not load ({e.__class__.__name__})")
    if not index:
        index = read_list(EXTRA_INDEX_FILE)
        if index:
            notes.append(f"Used fallback index_tickers.txt: {len(index)} symbols")
    watch = read_list(WATCHLIST_FILE)
    seen, universe = set(), []
    for t in watch + index:
        if t not in seen:
            seen.add(t)
            universe.append(t)
    return universe, watch, notes, meta


# ── Data ────────────────────────────────────────────────────────────────────
def download(tickers: list[str]) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    data: dict[str, pd.DataFrame] = {}
    today_utc = dt.datetime.now(dt.timezone.utc).date()
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i:i + CHUNK]
        try:
            df = yf.download(chunk, period=HISTORY, interval="1d", group_by="ticker",
                             auto_adjust=False, threads=True, progress=False)
        except Exception as e:  # noqa: BLE001
            print(f"download error for chunk {i}: {e}", file=sys.stderr)
            continue
        for t in chunk:
            try:
                sub = df[t] if isinstance(df.columns, pd.MultiIndex) else df
                sub = sub[["Open", "High", "Low", "Close"]].dropna()
            except (KeyError, TypeError):
                continue
            # Crypto trades 24/7: drop the still-forming UTC day
            if t.endswith("-USD") and len(sub) and pd.Timestamp(sub.index[-1]).date() >= today_utc:
                sub = sub.iloc[:-1]
            if len(sub) >= SMA_LEN + 2 * PIVOT_LEN:
                data[t] = sub
    return data


# ── Indicators ──────────────────────────────────────────────────────────────
def rsi_wilder(close: np.ndarray, length: int = 14) -> np.ndarray:
    """RSI with Wilder smoothing (same as TradingView's ta.rsi)."""
    close = np.asarray(close, float)
    n = len(close)
    out = np.full(n, np.nan)
    if n <= length:
        return out
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    ag, al = gain[:length].mean(), loss[:length].mean()   # RMA seed = SMA of first `length` changes

    def val(g, l):
        return 100.0 if l == 0 else 100 - 100 / (1 + g / l)

    out[length] = val(ag, al)
    for i in range(length, len(delta)):
        ag = (ag * (length - 1) + gain[i]) / length
        al = (al * (length - 1) + loss[i]) / length
        out[i + 1] = val(ag, al)
    return out


# ── Core logic (bar-by-bar port of the Pine script) ─────────────────────────
def analyze(df: pd.DataFrame) -> dict:
    h = df["High"].to_numpy(float)
    lo = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    dates = pd.to_datetime(df.index)
    n = len(c)
    L = PIVOT_LEN
    sma = pd.Series(c).rolling(SMA_LEN).mean().to_numpy()
    sma50 = pd.Series(c).rolling(50).mean().to_numpy()
    rsi = rsi_wilder(c, RSI_LEN)

    last_ph = last_ph_bar = last_pl = last_pl_bar = None
    regime = 0
    leg_hi = leg_lo = None
    leg_hi_bar = leg_lo_bar = None
    prev_zone = 0
    prev_leg = (None, None)
    trig_bar = trig_price = None
    trig_zone = 0

    z1 = z2 = z3 = None
    zone = 0
    entered = False
    leg_valid = False
    bull_ok = bear_ok = False
    dist = None

    for i in range(n):
        ph = pl = None
        if i >= 2 * L:
            ci = i - L
            wh = h[i - 2 * L:i + 1]
            wl = lo[i - 2 * L:i + 1]
            if h[ci] >= wh.max() and h[ci] > h[i - 2 * L:ci].max():
                ph = h[ci]
            if lo[ci] <= wl.min() and lo[ci] < lo[i - 2 * L:ci].min():
                pl = lo[ci]

        # Confirmed swing low: in a bearish regime, a lower low updates the down leg
        if pl is not None:
            pl_bar = i - L
            if regime == -1 and leg_lo is not None and pl < leg_lo and last_pl_bar is not None:
                seg = h[last_pl_bar:pl_bar + 1]
                hi_val = seg.max()
                hi_bar = last_pl_bar + int(seg.argmax())
                if (hi_val - pl) / hi_val >= MIN_MOVE:
                    leg_hi, leg_hi_bar = hi_val, hi_bar
                leg_lo, leg_lo_bar = pl, pl_bar
            last_pl, last_pl_bar = pl, pl_bar

        # Confirmed swing high: higher high = bullish leg; first lower high = bearish
        if ph is not None:
            ph_bar = i - L
            if last_ph is not None:
                if ph > last_ph:
                    seg = lo[last_ph_bar:ph_bar + 1]
                    lo_val = seg.min()
                    lo_bar = last_ph_bar + int(seg.argmin())
                    new_leg = regime != 1 or (lo_val > 0 and (ph - lo_val) / lo_val >= MIN_MOVE)
                    if new_leg:
                        leg_lo, leg_lo_bar = lo_val, lo_bar
                    leg_hi, leg_hi_bar = ph, ph_bar
                    regime = 1
                elif ph < last_ph and regime != -1:
                    seg = lo[last_ph_bar:ph_bar + 1]
                    leg_hi, leg_hi_bar = last_ph, last_ph_bar
                    leg_lo, leg_lo_bar = seg.min(), last_ph_bar + int(seg.argmin())
                    regime = -1
            last_ph, last_ph_bar = ph, ph_bar

        # Structure break invalidates the leg
        if regime == 1 and c[i] < leg_lo:
            regime = 0
        if regime == -1 and c[i] > leg_hi:
            regime = 0

        # Zones
        leg_valid = False
        if regime == 1 and leg_lo and leg_lo > 0:
            leg_valid = (leg_hi - leg_lo) / leg_lo >= MIN_MOVE
        elif regime == -1 and leg_hi and leg_hi > 0:
            leg_valid = (leg_hi - leg_lo) / leg_hi >= MIN_MOVE

        z1 = z2 = z3 = None
        if regime in (1, -1):
            rng = leg_hi - leg_lo
            if regime == 1:
                z1, z2, z3 = leg_hi - rng * FIB_A, leg_hi - rng * FIB_B, leg_hi - rng * FIB_C
            else:
                z1, z2, z3 = leg_lo + rng * FIB_A, leg_lo + rng * FIB_B, leg_lo + rng * FIB_C

        sma_ok_bull = (not USE_SMA) or (not np.isnan(sma[i]) and c[i] > sma[i])
        sma_ok_bear = (not USE_SMA) or (not np.isnan(sma[i]) and c[i] < sma[i])
        bull_ok = regime == 1 and leg_valid and sma_ok_bull
        bear_ok = regime == -1 and leg_valid and sma_ok_bear

        p_bull = lo[i] if TRIGGER == "wick" else c[i]
        p_bear = h[i] if TRIGGER == "wick" else c[i]
        zone = 0
        if bull_ok:
            if z2 <= p_bull <= z1:
                zone = 1
            elif z3 <= p_bull < z2:
                zone = 2
        elif bear_ok:
            if z1 <= p_bear <= z2:
                zone = -1
            elif z2 < p_bear <= z3:
                zone = -2

        dist = None
        if bull_ok:
            dist = (c[i] - z1) / c[i] * 100
        elif bear_ok:
            dist = (z1 - c[i]) / c[i] * 100

        entered = zone != 0 and zone != prev_zone

        # Trigger tracking (resets when a new swing is set)
        if (leg_hi, leg_lo) != prev_leg:
            trig_bar = trig_price = None
            trig_zone = 0
        if entered:
            trig_bar, trig_price, trig_zone = i, c[i], zone
        prev_leg = (leg_hi, leg_lo)
        prev_zone = zone

    def d(idx):
        return dates[idx].strftime("%Y-%m-%d") if idx is not None else ""

    trend = {1: "Bullish", -1: "Bearish", 0: "No active swing"}[regime]
    if regime != 0 and not leg_valid:
        trend += " (swing <15%)"
    if regime == 1 and leg_valid and not bull_ok:
        trend += " (below 200 SMA)"
    if regime == -1 and leg_valid and not bear_ok:
        trend += " (above 200 SMA)"

    return {
        "date": d(n - 1),
        "close": c[-1],
        "rsi14": rsi[-1],
        "sma50": sma50[-1],
        "pct_vs_sma50": (c[-1] / sma50[-1] - 1) * 100 if sma50[-1] == sma50[-1] else None,
        "sma200": sma[-1],
        "pct_vs_sma200": (c[-1] / sma[-1] - 1) * 100 if sma[-1] == sma[-1] else None,
        "trend": trend,
        "regime": regime,
        "bull_ok": bull_ok,
        "zone_code": zone,
        "zone": ZONE_NAMES[zone],
        "entered_today": bool(entered),
        "trigger_zone": ZONE_NAMES[trig_zone],
        "trigger_date": d(trig_bar),
        "trigger_price": trig_price,
        "swing_low": leg_lo,
        "swing_low_date": d(leg_lo_bar),
        "swing_high": leg_hi,
        "swing_high_date": d(leg_hi_bar),
        "fib_618": z1,
        "fib_786": z2,
        "fib_826": z3,
        "dist_to_zone_pct": dist,
        # internal (used for charts; dropped from the CSV)
        "_lo_bar": leg_lo_bar,
        "_hi_bar": leg_hi_bar,
        "_trig_bar": trig_bar,
    }




# ── Charts ──────────────────────────────────────────────────────────────────
# Palette: price in ink; 50D/200D = validated categorical slots 1-2 (blue/orange);
# RSI in its own pane; Fib zones use the same teal/green as the TradingView indicator.
C = {
    "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#8e8d89", "grid": "#ecebe7",
    "sma50": "#2a78d6", "sma200": "#eb6834", "rsi": "#6250d6",
    "ibv": "#089981", "sm": "#2e7d32", "ibs": "#e34948", "sms": "#9c27b0",
}


def make_chart(t: str, df: pd.DataFrame, r: dict, meta: dict) -> bytes:
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    n = len(df)
    c_all = df["Close"].to_numpy(float)
    start = n - CHART_BARS
    def toint(v):
        return int(v) if isnum(v) else None
    lo_b, hi_b, tr_b = toint(r.get("_lo_bar")), toint(r.get("_hi_bar")), toint(r.get("_trig_bar"))
    if lo_b is not None and hi_b is not None:
        start = min(start, min(lo_b, hi_b) - 15)   # always show the full swing
    start = max(0, max(start, n - 750))

    x = pd.to_datetime(df.index[start:])
    close = c_all[start:]
    sma50 = pd.Series(c_all).rolling(50).mean().to_numpy()[start:]
    sma200 = pd.Series(c_all).rolling(SMA_LEN).mean().to_numpy()[start:]
    rsi = rsi_wilder(c_all, RSI_LEN)[start:]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5})
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.4, 3.7), dpi=200, sharex=True,
                                  gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.08})
    for a in (ax, ax2):
        a.set_facecolor("white")
        a.grid(axis="y", color=C["grid"], linewidth=0.6)
        a.grid(axis="x", visible=False)
        for s in ("top", "left", "right"):
            a.spines[s].set_visible(False)
        a.spines["bottom"].set_color(C["muted"])
        a.yaxis.tick_right()
        a.tick_params(colors=C["ink2"], length=0, labelsize=7)

    # Fib zones (shaded from the start of the swing to the right edge)
    regime = r.get("regime", 0)
    z1, z2, z3 = r.get("fib_618"), r.get("fib_786"), r.get("fib_826")
    if z1 is not None and lo_b is not None and hi_b is not None:
        zs = max(min(lo_b, hi_b), start) - start
        xz = x[zs:]
        ca, cb = (C["ibv"], C["sm"]) if regime == 1 else (C["ibs"], C["sms"])
        na_, nb_ = ("IBV", "SM") if regime == 1 else ("IBS", "SMS")
        ax.fill_between(xz, min(z1, z2), max(z1, z2), color=ca, alpha=0.14, linewidth=0, label=f"{na_} 0.618–0.786")
        ax.fill_between(xz, min(z2, z3), max(z2, z3), color=cb, alpha=0.26, linewidth=0, label=f"{nb_} 0.786–0.826")
        for lvl in (r["swing_high"], r["swing_low"]):
            ax.hlines(lvl, xz[0], xz[-1], colors=C["muted"], linewidth=0.7, linestyles=(0, (3, 3)))
        # dashed swing line low -> high
        if start <= lo_b and start <= hi_b:
            ax.plot([x[lo_b - start], x[hi_b - start]], [r["swing_low"], r["swing_high"]],
                    color=C["muted"], linewidth=0.7, linestyle=(0, (2, 2)))

    # Price + moving averages
    ax.plot(x, close, color=C["ink"], linewidth=1.1, label="Close", zorder=4)
    ax.plot(x, sma50, color=C["sma50"], linewidth=1.2, label="50D MA", zorder=3)
    ax.plot(x, sma200, color=C["sma200"], linewidth=1.2, label="200D MA", zorder=3)

    # Right-edge level labels (nudged apart so they never collide)
    ymin, ymax = ax.get_ylim()
    labels = []
    if z1 is not None:
        labels += [(r["swing_high"], f"0  {fmt(r['swing_high'])}"),
                   (z1, f".618  {fmt(z1)}"), (z2, f".786  {fmt(z2)}"), (z3, f".826  {fmt(z3)}"),
                   (r["swing_low"], f"1  {fmt(r['swing_low'])}")]
    labels = [(y, s) for y, s in labels if ymin <= y <= ymax]
    labels.sort(key=lambda p: p[0])
    gap = (ymax - ymin) * 0.06
    placed = []
    for y, s in labels:
        if placed and y - placed[-1] < gap:
            y = placed[-1] + gap
        placed.append(y)
        ax.annotate(s, xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(30, 0),
                    textcoords="offset points", va="center", ha="left", fontsize=6.5, color=C["ink2"])

    # Zone entry marker
    if tr_b is not None and tr_b >= start:
        xi = x[tr_b - start]
        yi = c_all[tr_b]
        zc = C["ibv"] if r.get("trigger_zone") == "IBV" else C["sm"] if r.get("trigger_zone") == "SM" else C["ink2"]
        ax.scatter([xi], [yi], s=34, marker="^", color=zc, edgecolor="white", linewidth=1.2, zorder=6)
        ax.annotate(f"{r.get('trigger_zone')} entry {pd.Timestamp(xi):%m/%d}", xy=(xi, yi), xytext=(0, -13),
                    textcoords="offset points", ha="center", fontsize=6.5, color=C["ink2"])

    ax.legend(loc="lower left", ncol=5, frameon=False, fontsize=6.5, handlelength=1.6,
              columnspacing=1.0, labelcolor=C["ink2"], bbox_to_anchor=(0, 1.0), borderaxespad=0.2)

    # RSI pane
    ax2.axhspan(30, 70, color=C["grid"], alpha=0.45, linewidth=0)
    for lv in (30, 70):
        ax2.axhline(lv, color=C["muted"], linewidth=0.6, linestyle=(0, (3, 3)))
    ax2.plot(x, rsi, color=C["rsi"], linewidth=1.0)
    ax2.set_ylim(0, 100)
    ax2.set_yticks([30, 50, 70])
    last_rsi = rsi[-1]
    ax2.text(0.0, 0.97, f"RSI 14   {last_rsi:.0f}" if last_rsi == last_rsi else "RSI 14",
             transform=ax2.transAxes, va="top", fontsize=6.8, color=C["ink2"])
    loc = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax2.xaxis.set_major_locator(loc)
    ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

    # Title block
    name = (meta.get("name") or "")[:42]
    sub = "  ·  ".join(p for p in (name, meta.get("industry") or "") if p)
    fig.subplots_adjust(left=0.02, right=0.84, top=0.84, bottom=0.08)
    tt = fig.text(0.015, 0.975, t, fontsize=10, fontweight="bold", color=C["ink"], va="top")
    fig.canvas.draw()
    x_end = fig.transFigure.inverted().transform(tt.get_window_extent())[1][0]
    fig.text(x_end + 0.012, 0.97, sub, fontsize=7.5, color=C["ink2"], va="top")
    zone_txt = r.get("zone") or "–"
    rsi_txt = f"RSI {last_rsi:.0f}" if last_rsi == last_rsi else "RSI –"
    fig.text(0.015, 0.915, f"{zone_txt}  ·  Close {fmt(r['close'])}  ·  50D {fmt(r['sma50'])}  ·  "
             f"200D {fmt(r['sma200'])}  ·  {rsi_txt}", fontsize=7.2, color=C["ink2"], va="top")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


# ── Report ──────────────────────────────────────────────────────────────────
def tv_link(t: str) -> str:
    sym = t.replace("-USD", "USD") if t.endswith("-USD") else t.replace("-", ".")
    return f"https://www.tradingview.com/chart/?symbol={sym}"


def isnum(x) -> bool:
    return x is not None and not (isinstance(x, float) and np.isnan(x))


def fmt(x, nd=2):
    if not isnum(x):
        return "–"
    return f"{x:,.{nd}f}"


CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;color:#1d1d1f;max-width:980px}
h2{margin:24px 0 4px;font-size:17px} .sub{color:#6e6e73;font-size:12px;margin:0 0 8px}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th{background:#f2f2f4;text-align:left;padding:6px 8px;border-bottom:1px solid #d2d2d7;font-weight:600;white-space:nowrap}
th.r,td.r{text-align:right}
td{padding:6px 8px;border-bottom:1px solid #ececf0;vertical-align:top;white-space:nowrap}
td.id{white-space:normal;min-width:150px;max-width:230px}
.co{color:#3a3a3c;font-size:11.5px} .ind{color:#8e8e93;font-size:11px}
.pct{color:#8e8e93;font-size:11px} .tag{color:#8e8e93;font-size:10.5px}
.ibv{background:#089981;color:#fff;border-radius:4px;padding:1px 6px;font-weight:600}
.sm{background:#2e7d32;color:#fff;border-radius:4px;padding:1px 6px;font-weight:600}
.sell{background:#e34948;color:#fff;border-radius:4px;padding:1px 6px;font-weight:600}
.muted{color:#8e8e93} a{color:#0a66c2;text-decoration:none}
.chart{margin:14px 0 4px;border:1px solid #ececf0;border-radius:6px;max-width:100%}
"""


def badge(z: str) -> str:
    if not z:
        return '<span class="muted">–</span>'
    cls = {"IBV": "ibv", "SM": "sm"}.get(z, "sell")
    return f'<span class="{cls}">{z}</span>'


def id_cell(t: str, r) -> str:
    name = html.escape(str(r.get("name") or "")[:40])
    ind = html.escape(str(r.get("industry") or r.get("sector") or ""))
    return (f'<td class="id"><a href="{tv_link(t)}"><b>{html.escape(t)}</b></a>'
            f'{"<br><span class=co>" + name + "</span>" if name else ""}'
            f'{"<br><span class=ind>" + ind + "</span>" if ind else ""}</td>')


def rsi_cell(v) -> str:
    if not isnum(v):
        return '<td class="r muted">–</td>'
    tag = " <span class=tag>oversold</span>" if v <= 30 else " <span class=tag>overbought</span>" if v >= 70 else ""
    return f'<td class="r">{v:.0f}{tag}</td>'


def ma_cell(val, pct) -> str:
    if not isnum(val):
        return '<td class="r muted">–</td>'
    p = f'<br><span class="pct">{pct:+.1f}%</span>' if isnum(pct) else ""
    return f'<td class="r">{fmt(val)}{p}</td>'


def fib_cell(r) -> str:
    if not isnum(r["fib_618"]):
        return '<td class="muted">–</td>'
    return f"<td>{fmt(r['fib_618'])}<br>{fmt(r['fib_786'])}<br>{fmt(r['fib_826'])}</td>"


SETUP_HEAD = ("<tr><th>Ticker / Company</th><th>Zone</th><th class=r>Close</th><th class=r>RSI 14</th>"
              "<th class=r>50D MA</th><th class=r>200D MA</th><th>Fib .618 / .786 / .826</th>"
              "<th>Swing low → high</th><th>Entered</th><th class=r>To .618</th></tr>")


def setup_table(df: pd.DataFrame) -> str:
    if df.empty:
        return '<p class="muted">None today.</p>'
    rows = []
    for t, r in df.iterrows():
        rows.append(
            "<tr>" + id_cell(t, r)
            + f"<td>{badge(r['zone'])}</td>"
            + f'<td class="r"><b>{fmt(r["close"])}</b></td>'
            + rsi_cell(r["rsi14"])
            + ma_cell(r["sma50"], r["pct_vs_sma50"])
            + ma_cell(r["sma200"], r["pct_vs_sma200"])
            + fib_cell(r)
            + f"<td>{fmt(r['swing_low'])} → {fmt(r['swing_high'])}<br>"
              f"<span class=pct>{html.escape(r['swing_low_date'])} → {html.escape(r['swing_high_date'])}</span></td>"
            + f"<td>{html.escape(r['trigger_date'] or '–')}</td>"
            + f'<td class="r">{fmt(r["dist_to_zone_pct"], 1)}%</td>'
            + "</tr>"
        )
    return f"<table>{SETUP_HEAD}{''.join(rows)}</table>"


def watch_table(df: pd.DataFrame, watch: list[str], meta: dict) -> str:
    head = ("<tr><th>Ticker / Company</th><th class=r>Close</th><th>Trend</th><th>Zone</th>"
            "<th class=r>RSI 14</th><th class=r>50D MA</th><th class=r>200D MA</th>"
            "<th>Fib .618 / .786 / .826</th><th class=r>To .618</th><th>Last trigger</th></tr>")
    rows = []
    for t in watch:
        if t not in df.index:
            rows.append("<tr>" + id_cell(t, meta.get(t, {}))
                        + '<td colspan="9" class="muted">No data from Yahoo – check the ticker</td></tr>')
            continue
        r = df.loc[t]
        rows.append(
            "<tr>" + id_cell(t, r)
            + f'<td class="r"><b>{fmt(r["close"])}</b></td>'
            + f"<td>{html.escape(r['trend'])}</td>"
            + f"<td>{badge(r['zone'])}</td>"
            + rsi_cell(r["rsi14"])
            + ma_cell(r["sma50"], r["pct_vs_sma50"])
            + ma_cell(r["sma200"], r["pct_vs_sma200"])
            + fib_cell(r)
            + f'<td class="r">{fmt(r["dist_to_zone_pct"], 1)}%</td>'
            + f"<td>{html.escape((r['trigger_zone'] + ' ' + r['trigger_date']).strip() or '–')}</td>"
            + "</tr>"
        )
    return f"<table>{head}{''.join(rows)}</table>"


def build_report(res: pd.DataFrame, data: dict, watch: list[str], notes: list[str],
                 missing: list[str], meta: dict) -> tuple[str, str, list[tuple[str, bytes]]]:
    bull = res[res["bull_ok"]]
    in_zone = bull[bull["zone_code"].isin([1, 2])]
    new_today = in_zone[in_zone["entered_today"]]
    in_zone_old = in_zone[~in_zone["entered_today"]]
    near = bull[(bull["zone_code"] == 0) & (bull["dist_to_zone_pct"] > 0) & (bull["dist_to_zone_pct"] <= NEAR_PCT)]

    order = {"SM": 0, "IBV": 1}
    sort = lambda d: d.assign(_o=d["zone"].map(order)).sort_values(["_o", "dist_to_zone_pct"]).drop(columns="_o")
    new_today, in_zone_old = sort(new_today), sort(in_zone_old)
    near = near.sort_values("dist_to_zone_pct")
    asof = res["date"].max() if not res.empty else dt.date.today().isoformat()

    # Charts: new entries first, then existing in-zone, then in-zone watchlist names
    chart_list = list(new_today.index) + list(in_zone_old.index)
    chart_list += [t for t in watch if t in res.index and res.loc[t, "zone_code"] != 0 and t not in chart_list]
    images, chart_html = [], []
    for t in chart_list[:MAX_CHARTS]:
        try:
            png = make_chart(t, data[t], res.loc[t].to_dict(), meta.get(t, {}))
        except Exception as e:  # noqa: BLE001
            print(f"chart error {t}: {e}", file=sys.stderr)
            continue
        cid = f"chart-{re.sub(r'[^A-Za-z0-9]', '', t)}"
        images.append((cid, png))
        chart_html.append(f'<a href="{tv_link(t)}"><img class="chart" src="cid:{cid}" width="640" alt="{html.escape(t)} chart"></a>')
    extra = len(chart_list) - len(images)

    subject = (f"IBV/SM Scan {asof}: {len(new_today)} new, {len(in_zone)} in zone, "
               f"{len(near)} approaching")

    body = f"""<html><head><style>{CSS}</style></head><body>
<h1 style="font-size:20px;margin-bottom:2px">IBV / SM Nightly Scan</h1>
<p class="sub">Data as of {asof} close · {len(res)} symbols scanned · bullish structure, close above 200D MA, swing ≥15% ·
RSI 14 (Wilder) · MA % = close vs. the average</p>

<h2>🆕 New today ({len(new_today)})</h2>
<p class="sub">Closed inside IBV or SM for the first time on the latest bar.</p>
{setup_table(new_today)}

<h2>📍 Already in zone ({len(in_zone_old)})</h2>
<p class="sub">Still inside IBV / SM; "Entered" = first close in the zone on this swing.</p>
{setup_table(in_zone_old)}

<h2>👀 Approaching ({len(near)})</h2>
<p class="sub">Bullish and within {NEAR_PCT:.0f}% above the 0.618 level.</p>
{setup_table(near)}

<h2>⭐ Your watchlist ({len(watch)})</h2>
{watch_table(res, watch, meta)}

<h2>📈 Setup charts ({len(images)})</h2>
<p class="sub">Price with 50D (blue) and 200D (orange) moving averages; shaded IBV / SM zones from the swing;
RSI 14 in the lower pane with 30 / 70 guides. Click a chart to open it in TradingView.
{f"{extra} more setups not charted (limit {MAX_CHARTS})." if extra > 0 else ""}</p>
{''.join(chart_html) or '<p class="muted">No setups to chart today.</p>'}

<p class="sub" style="margin-top:22px">
Settings: pivot {PIVOT_LEN} bars · min swing {MIN_MOVE:.0%} · Fib {FIB_A}/{FIB_B}/{FIB_C} · SMA {SMA_LEN} · trigger {TRIGGER}<br>
Universe: {html.escape(' · '.join(notes))}{' · Missing data: ' + html.escape(', '.join(missing[:40])) if missing else ''}<br>
Ticker links open TradingView. Not investment advice.</p>
</body></html>"""
    return subject, body, images


def send_email(subject: str, body: str, csv_bytes: bytes, images: list[tuple[str, bytes]]) -> None:
    # Strip ALL whitespace (incl. non-breaking spaces / newlines picked up when pasting)
    user = re.sub(r"\s", "", os.environ["EMAIL_USER"])
    pwd = re.sub(r"\s", "", os.environ["EMAIL_APP_PASSWORD"])
    to = re.sub(r"\s", "", os.environ.get("EMAIL_TO") or "") or user
    if len(pwd) != 16 or not pwd.isascii():
        print(f"WARNING: EMAIL_APP_PASSWORD has {len(pwd)} characters "
              f"(expected 16 letters){'' if pwd.isascii() else ' and contains non-standard characters'}.",
              file=sys.stderr)
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to
    msg.set_content("HTML report attached. Open in an email client that supports HTML.")
    msg.add_alternative(body, subtype="html")
    html_part = msg.get_payload()[1]
    for cid, png in images:
        html_part.add_related(png, maintype="image", subtype="png", cid=f"<{cid}>")
    msg.add_attachment(csv_bytes, maintype="text", subtype="csv",
                       filename=f"ibv_scan_{dt.date.today().isoformat()}.csv")
    ctx = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx, timeout=60) as s:
            s.login(user, pwd)
            s.send_message(msg)
    except smtplib.SMTPAuthenticationError:
        raise SystemExit("Gmail rejected the login: recreate the App Password and update the "
                         "EMAIL_APP_PASSWORD secret; confirm EMAIL_USER is the full Gmail address.")
    except (smtplib.SMTPServerDisconnected, OSError) as e:
        print(f"Port 465 failed ({e.__class__.__name__}); retrying on 587 with STARTTLS", file=sys.stderr)
        try:
            with smtplib.SMTP("smtp.gmail.com", 587, timeout=60) as s:
                s.starttls(context=ctx)
                s.login(user, pwd)
                s.send_message(msg)
        except (smtplib.SMTPAuthenticationError, smtplib.SMTPServerDisconnected) as e2:
            raise SystemExit(f"Gmail login failed on both ports ({e2.__class__.__name__}). "
                             "Most likely cause: EMAIL_USER / EMAIL_APP_PASSWORD secrets are wrong. "
                             "Recreate the App Password and paste it again (16 letters).")


# ── Main ────────────────────────────────────────────────────────────────────
def run(data: dict[str, pd.DataFrame], universe: list[str], watch: list[str], notes: list[str],
        meta: dict[str, dict]):
    results = {}
    for t, df in data.items():
        try:
            results[t] = analyze(df)
        except Exception as e:  # noqa: BLE001
            print(f"analyze error {t}: {e}", file=sys.stderr)
    res = pd.DataFrame.from_dict(results, orient="index")
    info = pd.DataFrame({k: [meta.get(t, {}).get(k, "") for t in res.index]
                         for k in ("name", "sector", "industry")}, index=res.index)
    res = pd.concat([info, res], axis=1)
    missing = [t for t in universe if t not in results]
    subject, body, images = build_report(res, data, watch, notes, missing, meta)
    return res, subject, body, images


def main() -> int:
    universe, watch, notes, meta = build_universe()
    print(f"Universe: {len(universe)} symbols ({'; '.join(notes)})")
    data = download(universe)
    print(f"Downloaded: {len(data)} symbols")
    need = [t for t in data if not meta.get(t, {}).get("name")][:80]
    if need:
        meta.update(fetch_meta_yahoo(need))
        print(f"Company info from Yahoo: {len(need)} symbols")
    res, subject, body, images = run(data, universe, watch, notes, meta)
    csv_bytes = res.drop(columns=[c for c in res.columns if str(c).startswith("_")]).round(4) \
                   .to_csv(index_label="ticker").encode()

    dry = os.environ.get("DRY_RUN") == "1" or not os.environ.get("EMAIL_USER")
    (BASE / "results.csv").write_bytes(csv_bytes)
    if dry:
        for cid, png in images:
            (BASE / f"{cid}.png").write_bytes(png)
            body = body.replace(f"cid:{cid}", f"{cid}.png")
        (BASE / "report.html").write_text(body)
        print(f"DRY RUN: {subject}\nWrote report.html, charts and results.csv")
    else:
        send_email(subject, body, csv_bytes, images)
        print(f"Emailed: {subject} ({len(images)} charts)")
    return 0 if len(data) else 1


if __name__ == "__main__":
    sys.exit(main())
