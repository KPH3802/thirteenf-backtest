#!/usr/bin/env python3
"""
13F Path B Scoring Analysis — Does initiator count predict return magnitude?

Steps:
  1. Add return columns to initiation_signals table (safe ALTER TABLE)
  2. Populate returns via yfinance batch download (3-source fallback)
  3. Run scoring analysis: bucket by initiator count, compare returns
  4. If significant: update signal_benchmarks + thirteenf_scanner.py scoring
  5. Print clear verdict

Usage:
    python3 thirteenf_path_b.py
"""

import sys
import sqlite3
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings('ignore', category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
DATA_DB = SCRIPT_DIR / "thirteenf_data.db"
POSITIONS_DB = Path("/Users/kevinheaney/gmc_data/positions.db")

# FMP key from ib_execution config
sys.path.insert(0, str(SCRIPT_DIR.parent / "ib_execution"))
try:
    import config as ib_config
    FMP_API_KEY = getattr(ib_config, 'FMP_API_KEY', '')
except ImportError:
    FMP_API_KEY = ''


# ---------------------------------------------------------------------------
# Step 1: Add columns to initiation_signals
# ---------------------------------------------------------------------------
NEW_COLUMNS = [
    ("entry_date",        "TEXT"),
    ("entry_price",       "REAL"),
    ("exit_date",         "TEXT"),
    ("exit_price",        "REAL"),
    ("ret_91d",           "REAL"),
    ("spy_ret_91d",       "REAL"),
    ("alpha_91d",         "REAL"),
    ("initiator_bucket",  "TEXT"),
]


def add_columns(conn):
    """Add new columns if they don't exist."""
    c = conn.cursor()
    c.execute("PRAGMA table_info(initiation_signals)")
    existing = {row[1] for row in c.fetchall()}
    for col_name, col_type in NEW_COLUMNS:
        if col_name not in existing:
            c.execute(f"ALTER TABLE initiation_signals ADD COLUMN {col_name} {col_type}")
            print(f"  Added column: {col_name} {col_type}")
        else:
            print(f"  Column exists: {col_name}")
    conn.commit()


# ---------------------------------------------------------------------------
# Step 2: Populate returns
# ---------------------------------------------------------------------------
def add_business_days(start_date, n_days):
    """Add n business days to a date (skip weekends)."""
    current = start_date
    added = 0
    while added < n_days:
        current += timedelta(days=1)
        if current.weekday() < 5:  # Mon-Fri
            added += 1
    return current


def fetch_prices_batch(tickers, start_date, end_date):
    """Fetch prices for multiple tickers via yfinance batch download.
    Returns dict {ticker: DataFrame with Date index and Close column}.
    """
    if not tickers:
        return {}

    result = {}
    # Batch download
    try:
        # Add buffer days for weekends/holidays
        start_buf = (pd.Timestamp(start_date) - pd.Timedelta(days=5)).strftime('%Y-%m-%d')
        end_buf = (pd.Timestamp(end_date) + pd.Timedelta(days=5)).strftime('%Y-%m-%d')

        ticker_str = " ".join(tickers)
        data = yf.download(ticker_str, start=start_buf, end=end_buf,
                           progress=False, auto_adjust=True, threads=True)

        if data.empty:
            return result

        # Handle MultiIndex columns from multi-ticker download
        if isinstance(data.columns, pd.MultiIndex):
            for t in tickers:
                try:
                    close = data[('Close', t)].dropna()
                    if not close.empty:
                        result[t] = close
                except (KeyError, TypeError):
                    pass
        else:
            # Single ticker case
            if len(tickers) == 1 and 'Close' in data.columns:
                result[tickers[0]] = data['Close'].dropna()

    except Exception as e:
        print(f"  Batch download failed: {e}")

    return result


def get_price_on_date(price_series, target_date):
    """Get the closest price on or after target_date (within 5 days)."""
    if price_series is None or price_series.empty:
        return None
    target = pd.Timestamp(target_date)
    # Look for exact or next available date within 5 business days
    mask = price_series.index >= target
    available = price_series[mask]
    if available.empty:
        return None
    first_date = available.index[0]
    if (first_date - target).days > 7:
        return None
    return float(available.iloc[0])


def fetch_price_fmp(ticker, target_date):
    """Fallback: fetch single price from FMP."""
    if not FMP_API_KEY:
        return None
    try:
        import requests
        r = requests.get(
            f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}"
            f"?from={target_date}&to={target_date}&apikey={FMP_API_KEY}",
            timeout=10
        )
        data = r.json()
        if 'historical' in data and data['historical']:
            return float(data['historical'][0]['close'])
    except Exception:
        pass
    return None


def populate_returns(conn):
    """Populate return data for all rows with 3+ initiators."""
    c = conn.cursor()

    # Get rows needing population (3+ initiators, no return yet)
    c.execute("""
        SELECT ticker, quarter_end, filing_date, new_initiations
        FROM initiation_signals
        WHERE new_initiations >= 3 AND ret_91d IS NULL
        ORDER BY filing_date
    """)
    rows = c.fetchall()
    total = len(rows)
    print(f"\n  {total} rows need return data")

    if total == 0:
        return

    # Calculate entry/exit dates for all rows
    entries = []
    for ticker, qe, fd, ni in rows:
        filing = date.fromisoformat(fd)
        entry_dt = add_business_days(filing, 2)
        exit_dt = entry_dt + timedelta(days=91)
        bucket = '3' if ni == 3 else ('4' if ni == 4 else '5+')
        entries.append({
            'ticker': ticker, 'quarter_end': qe, 'filing_date': fd,
            'new_initiations': ni,
            'entry_date': entry_dt, 'exit_date': exit_dt,
            'bucket': bucket,
        })

    # Collect all unique tickers + SPY
    all_tickers = list(set(e['ticker'] for e in entries)) + ['SPY']
    min_date = min(e['entry_date'] for e in entries).isoformat()
    max_date = max(e['exit_date'] for e in entries).isoformat()

    print(f"  Fetching prices for {len(all_tickers)} tickers ({min_date} to {max_date})...")

    # Batch download in chunks of 50 tickers
    price_cache = {}
    chunk_size = 50
    for i in range(0, len(all_tickers), chunk_size):
        chunk = all_tickers[i:i + chunk_size]
        print(f"    Batch {i // chunk_size + 1}: {len(chunk)} tickers...")
        batch = fetch_prices_batch(chunk, min_date, max_date)
        price_cache.update(batch)

    print(f"  Got price data for {len(price_cache)} of {len(all_tickers)} tickers")

    spy_prices = price_cache.get('SPY')

    # Populate each row
    populated = 0
    skipped = 0
    for e in entries:
        ticker = e['ticker']
        entry_date_str = e['entry_date'].isoformat()
        exit_date_str = e['exit_date'].isoformat()

        # Get prices
        ticker_prices = price_cache.get(ticker)
        entry_price = get_price_on_date(ticker_prices, e['entry_date'])
        exit_price = get_price_on_date(ticker_prices, e['exit_date'])

        # FMP fallback for missing prices
        if entry_price is None:
            entry_price = fetch_price_fmp(ticker, entry_date_str)
        if exit_price is None:
            exit_price = fetch_price_fmp(ticker, exit_date_str)

        if entry_price is None or exit_price is None or entry_price <= 0:
            # Write dates and bucket but leave returns NULL
            c.execute("""
                UPDATE initiation_signals
                SET entry_date = ?, exit_date = ?, initiator_bucket = ?
                WHERE ticker = ? AND quarter_end = ?
            """, (entry_date_str, exit_date_str, e['bucket'], ticker, e['quarter_end']))
            skipped += 1
            continue

        ret_91d = ((exit_price - entry_price) / entry_price) * 100

        # SPY return over same period
        spy_entry = get_price_on_date(spy_prices, e['entry_date'])
        spy_exit = get_price_on_date(spy_prices, e['exit_date'])
        spy_ret_91d = None
        alpha_91d = None
        if spy_entry and spy_exit and spy_entry > 0:
            spy_ret_91d = ((spy_exit - spy_entry) / spy_entry) * 100
            alpha_91d = ret_91d - spy_ret_91d

        c.execute("""
            UPDATE initiation_signals
            SET entry_date = ?, entry_price = ?, exit_date = ?, exit_price = ?,
                ret_91d = ?, spy_ret_91d = ?, alpha_91d = ?,
                initiator_bucket = ?
            WHERE ticker = ? AND quarter_end = ?
        """, (entry_date_str, round(entry_price, 4), exit_date_str, round(exit_price, 4),
              round(ret_91d, 4), round(spy_ret_91d, 4) if spy_ret_91d is not None else None,
              round(alpha_91d, 4) if alpha_91d is not None else None,
              e['bucket'], ticker, e['quarter_end']))
        populated += 1

    conn.commit()
    print(f"  Populated: {populated} | Skipped (no price): {skipped}")


# ---------------------------------------------------------------------------
# Step 3: Scoring analysis
# ---------------------------------------------------------------------------
def run_analysis(conn):
    """Run the scoring analysis and print results."""
    c = conn.cursor()

    # Verify population
    c.execute("SELECT COUNT(*) FROM initiation_signals WHERE new_initiations >= 3 AND ret_91d IS NOT NULL")
    filled = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM initiation_signals WHERE new_initiations >= 3")
    total = c.fetchone()[0]
    print(f"\n  Data coverage: {filled}/{total} signals have returns ({filled/total*100:.1f}%)")

    if filled < 50:
        print("  ERROR: Insufficient data for analysis")
        return None

    # Load data
    df = pd.read_sql_query("""
        SELECT ticker, quarter_end, filing_date, new_initiations,
               entry_date, entry_price, exit_date, exit_price,
               ret_91d, spy_ret_91d, alpha_91d, initiator_bucket
        FROM initiation_signals
        WHERE new_initiations >= 3 AND ret_91d IS NOT NULL
    """, conn)

    # Ensure buckets
    df['initiator_bucket'] = df['new_initiations'].apply(
        lambda x: '3' if x == 3 else ('4' if x == 4 else '5+'))

    # Extract year for era filtering
    df['year'] = pd.to_datetime(df['filing_date']).dt.year

    print("\n" + "=" * 75)
    print("=== 13F PATH B SCORING ANALYSIS ===")
    print("=" * 75)

    # --- Full dataset ---
    print("\nFULL DATASET (all years, 3+ initiators):")
    _print_bucket_table(df)

    # --- Recent era ---
    df_recent = df[df['year'] >= 2020]
    if len(df_recent) > 30:
        print("\nRECENT ERA (2020-2025 only):")
        _print_bucket_table(df_recent)

    # --- Statistical significance ---
    print("\nSTATISTICAL SIGNIFICANCE (t-test, two-tailed):")
    b3 = df[df['initiator_bucket'] == '3']['ret_91d'].dropna()
    b4 = df[df['initiator_bucket'] == '4']['ret_91d'].dropna()
    b5 = df[df['initiator_bucket'] == '5+']['ret_91d'].dropna()

    sig_4 = False
    sig_5 = False

    if len(b4) >= 10 and len(b3) >= 10:
        t4, p4 = stats.ttest_ind(b4, b3, equal_var=False)
        sig_4 = p4 < 0.10
        label_4 = "SIGNIFICANT at 10%" if sig_4 else "not significant"
        print(f"  4 vs 3:  t={t4:.2f}, p={p4:.3f}  [{label_4}]")
    else:
        print(f"  4 vs 3:  insufficient data (n4={len(b4)}, n3={len(b3)})")

    if len(b5) >= 10 and len(b3) >= 10:
        t5, p5 = stats.ttest_ind(b5, b3, equal_var=False)
        sig_5 = p5 < 0.10
        label_5 = "SIGNIFICANT at 10%" if sig_5 else "not significant"
        print(f"  5+ vs 3: t={t5:.2f}, p={p5:.3f}  [{label_5}]")
    else:
        print(f"  5+ vs 3: insufficient data (n5={len(b5)}, n3={len(b3)})")

    # --- Alpha analysis ---
    if 'alpha_91d' in df.columns and df['alpha_91d'].notna().sum() > 30:
        print("\nALPHA ANALYSIS (vs SPY):")
        for bucket in ['3', '4', '5+']:
            bk = df[df['initiator_bucket'] == bucket]['alpha_91d'].dropna()
            if len(bk) > 0:
                t_alpha, p_alpha = stats.ttest_1samp(bk, 0)
                sig_label = "***" if p_alpha < 0.01 else ("**" if p_alpha < 0.05 else ("*" if p_alpha < 0.10 else ""))
                print(f"  Bucket {bucket:>2s}: alpha={bk.mean():+.2f}%  t={t_alpha:.2f}  p={p_alpha:.3f} {sig_label}  (n={len(bk)})")

    # --- Recommendation ---
    print("\n" + "-" * 75)
    print("SCORING RECOMMENDATION:")

    b3_mean = b3.mean() if len(b3) > 0 else 0
    b4_mean = b4.mean() if len(b4) > 0 else 0
    b5_mean = b5.mean() if len(b5) > 0 else 0

    if sig_4 or sig_5:
        print(f"  Score 3 (5% size):  3 initiators -> avg {b3_mean:+.2f}%")
        if sig_4:
            print(f"  Score 4 (8% size):  4 initiators -> avg {b4_mean:+.2f}%  [SIGNIFICANT]")
        else:
            print(f"  Score 3 (5% size):  4 initiators -> avg {b4_mean:+.2f}%  [not significant, keep at Score 3]")
        if sig_5:
            print(f"  Score 5 (8% size):  5+ initiators -> avg {b5_mean:+.2f}%  [SIGNIFICANT]")
        else:
            print(f"  Score 3 (5% size):  5+ initiators -> avg {b5_mean:+.2f}%  [not significant, keep at Score 3]")
        print("\n  VERDICT: Implement tiered scoring based on significant buckets.")
        return {'sig_4': sig_4, 'sig_5': sig_5,
                'b3_mean': b3_mean, 'b4_mean': b4_mean, 'b5_mean': b5_mean}
    else:
        print(f"  Score 3 (5% size):  3 initiators -> avg {b3_mean:+.2f}%")
        print(f"  Score 3 (5% size):  4 initiators -> avg {b4_mean:+.2f}%  [not significant]")
        print(f"  Score 3 (5% size):  5+ initiators -> avg {b5_mean:+.2f}%  [not significant]")
        print("\n  VERDICT: Keep flat Score 3. No initiator gradient detected.")
        return None


def _print_bucket_table(df):
    """Print the bucket analysis table."""
    baseline_ret = None
    header = f"{'Bucket':>8s} | {'Trades':>6s} | {'Win Rate':>8s} | {'Avg Return':>10s} | {'Avg Alpha':>10s} | {'vs 3-init baseline':>20s}"
    print(f"  {header}")
    print(f"  {'-' * len(header)}")

    for bucket in ['3', '4', '5+']:
        bk = df[df['initiator_bucket'] == bucket]
        n = len(bk)
        if n == 0:
            continue
        wins = (bk['ret_91d'] > 0).sum()
        wr = wins / n * 100
        avg_ret = bk['ret_91d'].mean()
        avg_alpha = bk['alpha_91d'].mean() if bk['alpha_91d'].notna().sum() > 0 else float('nan')

        if baseline_ret is None:
            baseline_ret = avg_ret
            delta_str = "baseline"
        else:
            delta = avg_ret - baseline_ret
            delta_str = f"{delta:+.2f}% delta"

        alpha_str = f"{avg_alpha:+.2f}%" if not np.isnan(avg_alpha) else "N/A"
        print(f"  {bucket:>8s} | {n:>6d} | {wr:>7.1f}% | {avg_ret:>+9.2f}% | {alpha_str:>10s} | {delta_str:>20s}")


# ---------------------------------------------------------------------------
# Step 4: Update signal_benchmarks (if significant)
# ---------------------------------------------------------------------------
def update_benchmarks(result):
    """Update signal_benchmarks in positions.db if gradient is significant."""
    if not POSITIONS_DB.exists():
        print(f"\n  WARNING: positions.db not found at {POSITIONS_DB}")
        print("  Run this on the Mac Studio where positions.db lives.")
        print("  SQL statements to run manually:")
        _print_benchmark_sql(result)
        return

    conn = sqlite3.connect(str(POSITIONS_DB))
    c = conn.cursor()

    # Check if signal_benchmarks table exists
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='signal_benchmarks'")
    if not c.fetchone():
        print("\n  WARNING: signal_benchmarks table not found. Run migrate_positions_db.py first.")
        _print_benchmark_sql(result)
        conn.close()
        return

    stmts = []
    if result['sig_4']:
        sql = ("INSERT OR REPLACE INTO signal_benchmarks "
               "(source, direction, expected_return_pct, expected_hold_days) "
               f"VALUES ('THIRTEENF_BULL_S4', 'BUY', {result['b4_mean']:.2f}, 91)")
        stmts.append(sql)
    if result['sig_5']:
        sql = ("INSERT OR REPLACE INTO signal_benchmarks "
               "(source, direction, expected_return_pct, expected_hold_days) "
               f"VALUES ('THIRTEENF_BULL_S5', 'BUY', {result['b5_mean']:.2f}, 91)")
        stmts.append(sql)

    # Also update the base THIRTEENF_BULL benchmark with bucket-3 mean
    sql_base = ("INSERT OR REPLACE INTO signal_benchmarks "
                "(source, direction, expected_return_pct, expected_hold_days) "
                f"VALUES ('THIRTEENF_BULL', 'BUY', {result['b3_mean']:.2f}, 91)")
    stmts.append(sql_base)

    print("\n  Updating signal_benchmarks in positions.db:")
    for sql in stmts:
        print(f"    {sql}")
        c.execute(sql)
    conn.commit()
    conn.close()
    print("  Done.")


def _print_benchmark_sql(result):
    """Print SQL for manual execution."""
    if result['sig_4']:
        print(f"    INSERT OR REPLACE INTO signal_benchmarks (source, direction, expected_return_pct, expected_hold_days) "
              f"VALUES ('THIRTEENF_BULL_S4', 'BUY', {result['b4_mean']:.2f}, 91);")
    if result['sig_5']:
        print(f"    INSERT OR REPLACE INTO signal_benchmarks (source, direction, expected_return_pct, expected_hold_days) "
              f"VALUES ('THIRTEENF_BULL_S5', 'BUY', {result['b5_mean']:.2f}, 91);")


# ---------------------------------------------------------------------------
# Step 5: Update thirteenf_scanner.py (if significant)
# ---------------------------------------------------------------------------
def update_scanner(result):
    """Show the changes needed to thirteenf_scanner.py for tiered scoring."""
    scanner_path = SCRIPT_DIR.parent / "thirteenf_scanner" / "thirteenf_scanner.py"
    if not scanner_path.exists():
        print(f"\n  Scanner not found at {scanner_path}")
        return

    print("\n" + "=" * 75)
    print("SCANNER UPDATE: thirteenf_scanner.py")
    print("=" * 75)
    print("\n  The scanner currently fires all signals with hardcoded score=3.")
    print("  The email subject '13F BULL: TICK1, TICK2' does NOT include initiator count.")
    print()
    print("  ISSUE: The autotrader parses tickers from the subject line only.")
    print("  It does not know the initiator count per ticker.")
    print()
    print("  TO IMPLEMENT TIERED SCORING, two changes are needed:")
    print()
    print("  1. thirteenf_scanner.py: change subject format to include initiator counts:")
    print('     OLD: "13F BULL: AAPL, MSFT, GOOG"')
    print('     NEW: "13F BULL: AAPL(5), MSFT(4), GOOG(3)"')
    print()
    print("  2. ib_autotrader.py query_13f_signals_from_email(): parse counts from subject:")
    print('     Parse TICK(N) format -> set score=3 if N==3, score=4 if N==4, score=5 if N>=5')
    print()

    # Show the scanner email subject change
    if result['sig_4'] or result['sig_5']:
        print("  SCANNER CHANGE (thirteenf_scanner.py line 578):")
        print("  OLD:")
        print("    subject = '13F BULL: {}'.format(ticker_str)")
        print("  NEW:")
        print("    ticker_parts = ['{t}({n})'.format(t=s['ticker'], n=s['initiators']) for s in signals]")
        print("    ticker_str = ', '.join(ticker_parts[:8])")
        print("    if len(signals) > 8:")
        print("        ticker_str += ' +{} more'.format(len(signals) - 8)")
        print("    subject = '13F BULL: {}'.format(ticker_str)")
        print()
        print("  AUTOTRADER CHANGE (ib_autotrader.py query_13f_signals_from_email line 766):")
        print("  OLD:")
        print("    'score': 3,")
        print("  NEW:")
        print("    # Parse initiator count from TICK(N) format")
        print("    m = re.match(r'^([A-Z]{1,5})\\((\\d+)\\)$', ticker.strip())")
        print("    if m:")
        print("        tick, n_init = m.group(1), int(m.group(2))")
        print("    else:")
        print("        tick, n_init = ticker.strip(), 3  # backward compat")
        score_logic = "    score = 5 if n_init >= 5 else (4 if n_init == 4 else 3)"
        print(f"    {score_logic}")
        print()
        print("  NOTE: These changes require coordinated deploy on PA (scanner) + Mac Studio (autotrader).")
        print("  Recommend: deploy scanner first (new format), then autotrader (parser).")
        print("  Backward compat: autotrader fallback handles old 'TICK' format as score=3.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 75)
    print("13F PATH B SCORING ANALYSIS")
    print(f"Database: {DATA_DB}")
    print("=" * 75)

    conn = sqlite3.connect(str(DATA_DB))

    # Step 1: Add columns
    print("\nStep 1: Schema migration...")
    add_columns(conn)

    # Step 2: Populate returns
    print("\nStep 2: Populating returns...")
    populate_returns(conn)

    # Step 3: Analysis
    print("\nStep 3: Running analysis...")
    result = run_analysis(conn)

    # Step 4: Update benchmarks if significant
    if result:
        print("\nStep 4: Updating signal benchmarks...")
        update_benchmarks(result)

        # Step 5: Scanner update guidance
        print("\nStep 5: Scanner update guidance...")
        update_scanner(result)
    else:
        print("\nStep 4: No benchmark updates needed.")
        print("Step 5: No scanner changes needed.")

    # Final verification
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM initiation_signals WHERE new_initiations >= 3 AND ret_91d IS NOT NULL")
    filled = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM initiation_signals WHERE new_initiations >= 3")
    total = c.fetchone()[0]

    print("\n" + "=" * 75)
    print("VERIFICATION:")
    print(f"  Signals with returns: {filled}/{total} ({filled/total*100:.1f}%)")
    print(f"  Target: 400+ of {total}")
    print(f"  Status: {'PASS' if filled >= 400 else 'BELOW TARGET'}")
    print("=" * 75)

    conn.close()


if __name__ == "__main__":
    main()
