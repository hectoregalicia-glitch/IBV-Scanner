# IBV / SM Nightly Scanner

Emails a nightly list of stocks in the **IBV (0.618–0.786)** and **SM (0.786–0.826)** buy zones with bullish structure, plus stocks approaching the zone and the status of every name on your watchlist. Same logic as the TradingView "IBV / SM Zone Scanner" indicator.

Universe: S&P 500 + Nasdaq-100 (pulled fresh from Wikipedia each run) + `my_watchlist.txt`.
Schedule: weekdays 22:30 UTC (2:30 PM Pacific in winter, 3:30 PM in summer).

## Files
| File | Purpose |
|---|---|
| `scanner.py` | Scanner + email report |
| `my_watchlist.txt` | Your extra tickers — edit anytime, one per line |
| `requirements.txt` | Python libraries |
| `.github/workflows/scan.yml` | GitHub schedule |

## One-time setup (~20 min)
1. **Gmail App Password**
   Google Account → Security → turn on **2-Step Verification** → search "App passwords" → create one named `IBV Scanner` → copy the 16-character code.
2. **GitHub repository**
   github.com → sign up (free) → **New repository** → name `ibv-scanner` → **Private** → Create.
3. **Upload files**
   **Add file → Upload files** → drag in `scanner.py`, `my_watchlist.txt`, `requirements.txt`, `README.md` → Commit.
4. **Create the workflow file**
   **Add file → Create new file** → name it exactly `.github/workflows/scan.yml` → paste the contents of `scan.yml` → Commit.
5. **Add secrets**
   Repo **Settings → Secrets and variables → Actions → New repository secret**:
   - `EMAIL_USER` = your Gmail address
   - `EMAIL_APP_PASSWORD` = the 16-character App Password
   - `EMAIL_TO` = (optional) where to send it; defaults to `EMAIL_USER`
6. **Test**
   **Actions** tab → enable workflows if prompted → **IBV-SM Nightly Scan** → **Run workflow**. Takes ~3–6 minutes; check your inbox.

## Changing things
- **Add/remove tickers:** edit `my_watchlist.txt` on GitHub (pencil icon) → Commit. Crypto = `BTC-USD`.
- **Settings:** top of `scanner.py` (`PIVOT_LEN`, `MIN_MOVE`, `FIB_A/B/C`, `SMA_LEN`, `USE_SMA`, `TRIGGER`, `NEAR_PCT`).
- **Schedule:** the `cron` line in `scan.yml` (UTC).

## Notes
- Data comes from Yahoo Finance via `yfinance` (free, unofficial). A run occasionally misses a few tickers; they're listed at the bottom of the email.
- GitHub may start scheduled runs 5–30 minutes late.
- Every run also saves `results.csv` (all symbols) under the run's **Artifacts** in the Actions tab, and attaches it to the email.
