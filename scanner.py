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


def fetch_index(url: str, cols: list[str]) -> list[str]:
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; IBV-SM-Scanner/1.0)"}, timeout=30)
    r.raise_for_status()
    for table in pd.read_html(StringIO(r.text)):
        for c in cols:
            if c in table.columns and len(table) >= 90:
                return [normalize(str(x)) for x in table[c].dropna()]
    raise ValueError("ticker table not found")


def build_universe() -> tuple[list[str], list[str], list[str]]:
    """Returns (all_tickers, watchlist, notes)."""
    notes, index = [], []
    for name, (url, cols) in INDEX_SOURCES.items():
        try:
            tick = fetch_index(url, cols)
            index += tick
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
    return universe, watch, notes


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


# ── Core logic (bar-by-bar port of the Pine script) ─────────────────────────
def analyze(df: pd.DataFrame) -> dict:
    h = df["High"].to_numpy(float)
    lo = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    dates = pd.to_datetime(df.index)
    n = len(c)
    L = PIVOT_LEN
    sma = pd.Series(c).rolling(SMA_LEN).mean().to_numpy()

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
        "sma200": sma[-1],
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
    }


# ── Report ──────────────────────────────────────────────────────────────────
def tv_link(t: str) -> str:
    sym = t.replace("-USD", "USD") if t.endswith("-USD") else t.replace("-", ".")
    return f"https://www.tradingview.com/chart/?symbol={sym}"


def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:,.{nd}f}"


CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;color:#1d1d1f;max-width:900px}
h2{margin:22px 0 6px;font-size:17px} .sub{color:#6e6e73;font-size:12px;margin:0 0 8px}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th{background:#f2f2f4;text-align:left;padding:6px 8px;border-bottom:1px solid #d2d2d7}
td{padding:5px 8px;border-bottom:1px solid #ececf0;white-space:nowrap}
.ibv{background:#089981;color:#fff;border-radius:4px;padding:1px 6px;font-weight:600}
.sm{background:#2e7d32;color:#fff;border-radius:4px;padding:1px 6px;font-weight:600}
.sell{background:#f23645;color:#fff;border-radius:4px;padding:1px 6px;font-weight:600}
.muted{color:#8e8e93} a{color:#0a66c2;text-decoration:none}
"""


def badge(z: str) -> str:
    if not z:
        return '<span class="muted">–</span>'
    cls = {"IBV": "ibv", "SM": "sm"}.get(z, "sell")
    return f'<span class="{cls}">{z}</span>'


def setup_table(df: pd.DataFrame, cols: str = "zone") -> str:
    if df.empty:
        return '<p class="muted">None today.</p>'
    rows = []
    for t, r in df.iterrows():
        rows.append(
            "<tr>"
            f'<td><a href="{tv_link(t)}"><b>{html.escape(t)}</b></a></td>'
            f"<td>{badge(r['zone'])}</td>"
            f"<td>{fmt(r['close'])}</td>"
            f"<td>{fmt(r['fib_618'])} – {fmt(r['fib_786'])}</td>"
            f"<td>{fmt(r['fib_786'])} – {fmt(r['fib_826'])}</td>"
            f"<td>{fmt(r['swing_low'])} → {fmt(r['swing_high'])}</td>"
            f"<td>{html.escape(r['trigger_date'] or '–')}</td>"
            f"<td>{fmt(r['dist_to_zone_pct'], 1)}%</td>"
            "</tr>"
        )
    head = ("<tr><th>Ticker</th><th>Zone</th><th>Close</th><th>IBV (0.618–0.786)</th>"
            "<th>SM (0.786–0.826)</th><th>Swing low → high</th><th>Entered</th><th>Dist to 0.618</th></tr>")
    return f"<table>{head}{''.join(rows)}</table>"


def watch_table(df: pd.DataFrame, watch: list[str]) -> str:
    rows = []
    for t in watch:
        if t not in df.index:
            rows.append(f'<tr><td><b>{html.escape(t)}</b></td><td colspan="6" class="muted">No data from Yahoo</td></tr>')
            continue
        r = df.loc[t]
        rows.append(
            "<tr>"
            f'<td><a href="{tv_link(t)}"><b>{html.escape(t)}</b></a></td>'
            f"<td>{fmt(r['close'])}</td>"
            f"<td>{html.escape(r['trend'])}</td>"
            f"<td>{badge(r['zone'])}</td>"
            f"<td>{fmt(r['fib_618'])} / {fmt(r['fib_786'])} / {fmt(r['fib_826'])}</td>"
            f"<td>{fmt(r['dist_to_zone_pct'], 1)}%</td>"
            f"<td>{html.escape((r['trigger_zone'] + ' ' + r['trigger_date']).strip() or '–')}</td>"
            "</tr>"
        )
    head = ("<tr><th>Ticker</th><th>Close</th><th>Trend</th><th>Zone</th>"
            "<th>0.618 / 0.786 / 0.826</th><th>Dist</th><th>Last trigger</th></tr>")
    return f"<table>{head}{''.join(rows)}</table>"


def build_report(res: pd.DataFrame, watch: list[str], notes: list[str], missing: list[str]) -> tuple[str, str]:
    bull = res[res["bull_ok"]]
    in_zone = bull[bull["zone_code"].isin([1, 2])]
    new_today = in_zone[in_zone["entered_today"]]
    in_zone_old = in_zone[~in_zone["entered_today"]]
    near = bull[(bull["zone_code"] == 0) & (bull["dist_to_zone_pct"] > 0) & (bull["dist_to_zone_pct"] <= NEAR_PCT)]

    order = {"SM": 0, "IBV": 1}
    sort = lambda d: d.assign(_o=d["zone"].map(order)).sort_values(["_o", "dist_to_zone_pct"]).drop(columns="_o")
    asof = res["date"].max() if not res.empty else dt.date.today().isoformat()

    subject = (f"IBV/SM Scan {asof}: {len(new_today)} new, {len(in_zone)} in zone, "
               f"{len(near)} approaching")

    body = f"""<html><head><style>{CSS}</style></head><body>
<h1 style="font-size:20px;margin-bottom:2px">IBV / SM Nightly Scan</h1>
<p class="sub">Data as of {asof} close · {len(res)} symbols scanned · bullish structure, close above 200 SMA, swing ≥15%</p>

<h2>🆕 New today ({len(new_today)})</h2>
<p class="sub">Closed inside IBV or SM for the first time on the latest bar.</p>
{setup_table(sort(new_today))}

<h2>📍 Already in zone ({len(in_zone_old)})</h2>
<p class="sub">Still inside IBV / SM; "Entered" is the date it first closed in the zone on this swing.</p>
{setup_table(sort(in_zone_old))}

<h2>👀 Approaching ({len(near)})</h2>
<p class="sub">Bullish and within {NEAR_PCT:.0f}% above the 0.618 level.</p>
{setup_table(near.sort_values("dist_to_zone_pct"))}

<h2>⭐ Your watchlist ({len(watch)})</h2>
{watch_table(res, watch)}

<p class="sub" style="margin-top:22px">
Settings: pivot {PIVOT_LEN} bars · min swing {MIN_MOVE:.0%} · Fib {FIB_A}/{FIB_B}/{FIB_C} · SMA {SMA_LEN} · trigger {TRIGGER}<br>
Universe: {html.escape(' · '.join(notes))}{' · Missing data: ' + html.escape(', '.join(missing[:40])) if missing else ''}<br>
Ticker links open TradingView. Not investment advice.</p>
</body></html>"""
    return subject, body


def send_email(subject: str, body: str, csv_bytes: bytes) -> None:
    user = os.environ["EMAIL_USER"]
    pwd = os.environ["EMAIL_APP_PASSWORD"].replace(" ", "")
    to = os.environ.get("EMAIL_TO") or user
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to
    msg.set_content("HTML report attached. Open in an email client that supports HTML.")
    msg.add_alternative(body, subtype="html")
    msg.add_attachment(csv_bytes, maintype="text", subtype="csv",
                       filename=f"ibv_scan_{dt.date.today().isoformat()}.csv")
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context()) as s:
        s.login(user, pwd)
        s.send_message(msg)


# ── Main ────────────────────────────────────────────────────────────────────
def run(data: dict[str, pd.DataFrame], universe: list[str], watch: list[str], notes: list[str]):
    results = {}
    for t, df in data.items():
        try:
            results[t] = analyze(df)
        except Exception as e:  # noqa: BLE001
            print(f"analyze error {t}: {e}", file=sys.stderr)
    res = pd.DataFrame.from_dict(results, orient="index")
    missing = [t for t in universe if t not in results]
    subject, body = build_report(res, watch, notes, missing)
    return res, subject, body


def main() -> int:
    universe, watch, notes = build_universe()
    print(f"Universe: {len(universe)} symbols ({'; '.join(notes)})")
    data = download(universe)
    print(f"Downloaded: {len(data)} symbols")
    res, subject, body = run(data, universe, watch, notes)
    csv_bytes = res.round(4).to_csv(index_label="ticker").encode()

    dry = os.environ.get("DRY_RUN") == "1" or not os.environ.get("EMAIL_USER")
    if dry:
        (BASE / "report.html").write_text(body)
        (BASE / "results.csv").write_bytes(csv_bytes)
        print(f"DRY RUN: {subject}\nWrote report.html and results.csv")
    else:
        (BASE / "results.csv").write_bytes(csv_bytes)
        send_email(subject, body, csv_bytes)
        print(f"Emailed: {subject}")
    return 0 if len(data) else 1


if __name__ == "__main__":
    sys.exit(main())
