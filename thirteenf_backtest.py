#!/usr/bin/env python3
"""
13F Institutional Holdings Backtest
=====================================
Tests whether stocks that appear as NEW positions (initiations) in
multiple 13F filings in the same quarter outperform over the following
4-26 weeks.

Hypothesis: When multiple institutional investors initiate new positions
in the same stock in the same quarter, this reflects coordinated smart-money
conviction and predicts outperformance.

Data source: SEC EDGAR 13F filings (free, same pipeline as 8-K/Form 4)

Two-step process:
  1. python3 collect_13f.py    -- collect 13F data from SEC EDGAR (~2-4 hours)
  2. python3 thirteenf_backtest.py  -- run the backtest

Usage:
  python3 thirteenf_backtest.py              # Full backtest
  python3 thirteenf_backtest.py --min-init 3 # Stricter: 3+ initiators
  python3 thirteenf_backtest.py --export     # Save CSV
"""

import sys
import argparse
import sqlite3
import warnings
from pathlib import Path
from datetime import datetime, timedelta
import os
import contextlib

import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / cfg.RESULTS_DIR
DB_PATH = SCRIPT_DIR / cfg.THIRTEENF_DB


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_signals(min_initiations, start, end):
    """Load all qualifying 13F initiation signals from DB."""
    conn = sqlite3.connect(str(DB_PATH))

    conditions = [f'new_initiations >= {min_initiations}']
    if start:
        conditions.append(f'filing_date >= \"{start}\"')
    if end:
        conditions.append(f'filing_date <= \"{end}\"')

    query = f"""
        SELECT ticker, filing_date, quarter_end, new_initiations,
               total_holders, total_value_usd
        FROM initiation_signals
        WHERE {' AND '.join(conditions)}
        ORDER BY filing_date, ticker
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    df['filing_date'] = pd.to_datetime(df['filing_date'])
    return df


# ---------------------------------------------------------------------------
# Price fetching
# ---------------------------------------------------------------------------

def fetch_prices(ticker, start, end=None):
    end = end or datetime.today().strftime('%Y-%m-%d')
    try:
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if data is None or data.empty:
            return None
        if 'Close' not in data.columns:
            return None
        s = data['Close'].squeeze().dropna()
        s.index = pd.to_datetime(s.index)
        return s if not s.empty else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Forward return computation
# ---------------------------------------------------------------------------

def compute_forward_returns(signals_df, hold_periods, entry_delay_days):
    """Long entry after filing date, measure forward returns."""
    tickers = signals_df['ticker'].unique().tolist()
    total = len(tickers)
    price_cache = {}

    price_start = (pd.to_datetime(cfg.BACKTEST_START) - timedelta(weeks=4)).strftime('%Y-%m-%d')
    price_end = cfg.BACKTEST_END or datetime.today().strftime('%Y-%m-%d')

    print(f'  Fetching prices for {total} unique tickers...')
    for i, ticker in enumerate(tickers, 1):
        if i % 100 == 0 or i == total:
            print(f'    {i}/{total} ({len(price_cache)} retrieved)...', flush=True)
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stderr(devnull):
                prices = fetch_prices(ticker, price_start, price_end)
        if prices is not None:
            price_cache[ticker] = prices

    print(f'  Got prices for {len(price_cache)} of {total} tickers')

    trades = []
    for _, row in signals_df.iterrows():
        ticker = row['ticker']
        filing_date = row['filing_date']

        if ticker not in price_cache:
            continue

        prices = price_cache[ticker]
        entry_from = filing_date + timedelta(days=entry_delay_days)
        future = prices[prices.index >= entry_from]
        if future.empty:
            continue

        entry_date = future.index[0]
        entry_price = future.iloc[0]

        trade = {
            'ticker': ticker,
            'filing_date': filing_date,
            'entry_date': entry_date,
            'new_initiations': row['new_initiations'],
            'total_holders': row.get('total_holders'),
            'total_value_usd': row.get('total_value_usd'),
            'entry_price': entry_price,
        }

        for weeks in hold_periods:
            target_date = entry_date + timedelta(weeks=weeks)
            future_exit = prices[prices.index >= target_date]
            if future_exit.empty:
                trade[f'ret_{weeks}w'] = np.nan
                continue
            exit_price = future_exit.iloc[0]
            trade[f'ret_{weeks}w'] = (exit_price - entry_price) / entry_price

        trades.append(trade)

    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# Statistics & reporting
# ---------------------------------------------------------------------------

def compute_stats(rets):
    rets = rets.dropna()
    if len(rets) < 10:
        return dict(n=len(rets), alpha=np.nan, win_rate=np.nan, t_stat=np.nan, p_value=np.nan)
    t, p = stats.ttest_1samp(rets, 0)
    return dict(n=len(rets), alpha=rets.mean(), win_rate=(rets > 0).mean(), t_stat=t, p_value=p)


def sig_stars(p):
    if pd.isna(p): return ''
    if p < 0.01: return '***'
    if p < 0.05: return '**'
    if p < 0.10: return '*'
    return ''


def print_sep(char='=', w=70):
    print(char * w)


def print_stats_block(label, trades, hold_periods):
    print(f'  {label}  ({len(trades)} trades)')
    print(f'  {"Horizon":<10} {"N":>5} {"Alpha":>9} {"Win%":>8} {"t-stat":>8} {"p-val":>8}  Sig')
    print('  ' + '-' * 58)
    for weeks in hold_periods:
        col = f'ret_{weeks}w'
        if col not in trades.columns:
            continue
        s = compute_stats(trades[col])
        if pd.isna(s['alpha']):
            print(f'  {weeks}w{"":<8} {int(s["n"]):>5} {"--":>9}')
        else:
            stars = sig_stars(s['p_value'])
            print(
                f'  {weeks}w{"":<8} {int(s["n"]):>5} '
                f'{s["alpha"]:>+8.2%} {s["win_rate"]:>8.1%} '
                f'{s["t_stat"]:>8.2f} {s["p_value"]:>8.4f}  {stars}'
            )
    print()


def print_initiation_breakdown(trades, col_weeks=8):
    """Break results by number of new initiators."""
    col = f'ret_{col_weeks}w'
    if col not in trades.columns or trades.empty:
        return

    print(f'  INITIATION COUNT BREAKDOWN ({col_weeks}w return)')
    print(f'  {"# Initiators":<16} {"N":>5} {"Alpha":>9} {"Win%":>8} {"t-stat":>8}  Sig')
    print('  ' + '-' * 54)

    for n_init in sorted(trades['new_initiations'].unique()):
        subset = trades[trades['new_initiations'] == n_init]
        s = compute_stats(subset[col])
        label = f'{int(n_init)} initiators'
        if pd.isna(s['alpha']):
            print(f'  {label:<16} {int(s["n"]):>5} {"--":>9}')
        else:
            stars = sig_stars(s['p_value'])
            print(f'  {label:<16} {int(s["n"]):>5} {s["alpha"]:>+8.2%} {s["win_rate"]:>8.1%} {s["t_stat"]:>8.2f}  {stars}')
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='13F Institutional Initiation Backtest')
    parser.add_argument('--min-init', type=int, default=cfg.MIN_NEW_INITIATIONS,
                        help=f'Minimum new initiators (default: {cfg.MIN_NEW_INITIATIONS})')
    parser.add_argument('--export', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    if not DB_PATH.exists():
        print(f'ERROR: {DB_PATH} not found. Run collect_13f.py first.')
        sys.exit(1)

    print()
    print_sep()
    print('  13F INSTITUTIONAL INITIATION BACKTEST')
    print(f'  Run date:       {datetime.today().strftime("%Y-%m-%d")}')
    print(f'  DB:             {DB_PATH}')
    print(f'  Signal:         >= {args.min_init} new institutional initiations in same quarter')
    print(f'  Entry:          {cfg.ENTRY_DELAY_DAYS} day(s) after filing date')
    print(f'  Horizons:       {cfg.HOLD_PERIODS} weeks')
    print_sep()

    print(f'\n  Loading signals...', end=' ', flush=True)
    signals = load_signals(args.min_init, cfg.BACKTEST_START, cfg.BACKTEST_END)
    print(f'{len(signals):,} signals')

    if signals.empty:
        print('  No signals found. Run collect_13f.py first.')
        return

    print(f'  Date range: {signals["filing_date"].min().date()} to {signals["filing_date"].max().date()}')
    print(f'  Unique tickers: {signals["ticker"].nunique():,}')

    trades = compute_forward_returns(signals, cfg.HOLD_PERIODS, cfg.ENTRY_DELAY_DAYS)
    print(f'  Trades with price data: {len(trades):,}')

    if trades.empty:
        print('  No trades computed.')
        return

    print()
    print_sep()
    print('  RESULTS')
    print_sep()
    print()

    print_stats_block('ALL signals (long only)', trades, cfg.HOLD_PERIODS)
    print_sep('-')
    print_initiation_breakdown(trades, col_weeks=8)
    print('  * p<0.10  ** p<0.05  *** p<0.01')

    if args.export:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.today().strftime('%Y%m%d_%H%M%S')
        out_path = RESULTS_DIR / f'thirteenf_backtest_{ts}.csv'
        trades.to_csv(out_path, index=False)
        print(f'\n  Exported to: {out_path}')


if __name__ == '__main__':
    main()
