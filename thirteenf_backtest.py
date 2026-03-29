#!/usr/bin/env python3
"""
13F Institutional Holdings Backtest — v2
==========================================
Tests whether stocks newly initiated by multiple hedge funds in the same
quarter outperform over the following 4–26 weeks.

Signal definition:
  A stock appears as a NEW position (initiation) in 2+ hedge fund 13F filings
  in the same quarter.  Only Common Stock — ETFs and funds excluded via
  OpenFIGI security_type filter applied during collection.

Hypothesis:
  Coordinated new initiations by smart-money funds reflect genuine conviction
  and predict short-to-medium-term outperformance — especially in small/mid caps
  where institutional interest is less well-known.

Two-step process:
  1. python3 collect_13f.py --reset   (collect + map CUSIPs + build signals)
  2. python3 thirteenf_backtest.py    (run this file)

Usage:
  python3 thirteenf_backtest.py              # Full backtest
  python3 thirteenf_backtest.py --min-init 3 # Stricter: 3+ initiators required
  python3 thirteenf_backtest.py --export     # Save results CSV
"""

import sys
import argparse
import sqlite3
import warnings
import os
import contextlib
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

SCRIPT_DIR  = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / cfg.RESULTS_DIR
DB_PATH     = SCRIPT_DIR / cfg.THIRTEENF_DB


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_signals(min_initiations, start, end):
    """Load qualifying 13F initiation signals from DB."""
    conn = sqlite3.connect(str(DB_PATH))

    conditions = [f'new_initiations >= {min_initiations}']
    if start:
        conditions.append(f'filing_date >= "{start}"')
    if end:
        conditions.append(f'filing_date <= "{end}"')

    query = f"""
        SELECT ticker, filing_date, quarter_end,
               new_initiations, total_holders, total_value_usd
        FROM   initiation_signals
        WHERE  {' AND '.join(conditions)}
        ORDER  BY filing_date, ticker
    """

    df = pd.read_sql_query(query, conn)
    conn.close()
    df['filing_date'] = pd.to_datetime(df['filing_date'])
    return df


def load_holdings_detail():
    """Load filer-level detail from holdings table for L2 filer analysis."""
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql_query("""
        SELECT h.ticker, h.quarter_end, h.filing_date, h.filer_name,
               h.value_usd, h.is_new
        FROM   holdings h
        WHERE  h.is_new = 1
          AND  h.security_type = 'Common Stock'
          AND  h.ticker IS NOT NULL AND h.ticker != ''
    """, conn)
    conn.close()
    return df


# ---------------------------------------------------------------------------
# Price fetching
# ---------------------------------------------------------------------------

def fetch_prices(ticker, start, end=None):
    end = end or datetime.today().strftime('%Y-%m-%d')
    try:
        with open(os.devnull, 'w') as devnull, contextlib.redirect_stderr(devnull):
            data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if data is None or data.empty or 'Close' not in data.columns:
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
    """Long entry after filing date, measure returns at each hold horizon."""
    tickers     = signals_df['ticker'].unique().tolist()
    price_start = (pd.to_datetime(cfg.BACKTEST_START) - timedelta(weeks=4)).strftime('%Y-%m-%d')
    price_end   = cfg.BACKTEST_END or datetime.today().strftime('%Y-%m-%d')

    print(f'  Fetching prices for {len(tickers)} tickers...')
    price_cache = {}
    for i, ticker in enumerate(tickers, 1):
        if i % 100 == 0 or i == len(tickers):
            print(f'    {i}/{len(tickers)} ({len(price_cache)} retrieved)...', flush=True)
        prices = fetch_prices(ticker, price_start, price_end)
        if prices is not None:
            price_cache[ticker] = prices

    print(f'  Price data: {len(price_cache)}/{len(tickers)} tickers resolved')

    trades = []
    for _, row in signals_df.iterrows():
        ticker      = row['ticker']
        filing_date = row['filing_date']

        if ticker not in price_cache:
            continue

        prices     = price_cache[ticker]
        entry_from = filing_date + timedelta(days=entry_delay_days)
        future     = prices[prices.index >= entry_from]
        if future.empty:
            continue

        entry_date  = future.index[0]
        entry_price = future.iloc[0]

        trade = {
            'ticker':          ticker,
            'filing_date':     filing_date,
            'entry_date':      entry_date,
            'new_initiations': row['new_initiations'],
            'total_holders':   row.get('total_holders'),
            'total_value_usd': row.get('total_value_usd'),
            'entry_price':     entry_price,
        }

        for weeks in hold_periods:
            target_date = entry_date + timedelta(weeks=weeks)
            future_exit = prices[prices.index >= target_date]
            if future_exit.empty:
                trade[f'ret_{weeks}w'] = np.nan
                continue
            trade[f'ret_{weeks}w'] = (future_exit.iloc[0] - entry_price) / entry_price

        trades.append(trade)

    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def compute_stats(rets):
    rets = rets.dropna()
    if len(rets) < 10:
        return dict(n=len(rets), alpha=np.nan, win_rate=np.nan, t_stat=np.nan, p_value=np.nan)
    t, p = stats.ttest_1samp(rets, 0)
    return dict(n=len(rets), alpha=rets.mean(), win_rate=(rets > 0).mean(), t_stat=t, p_value=p)


def sig_stars(p):
    if pd.isna(p): return ''
    if p < 0.01:   return '***'
    if p < 0.05:   return '**'
    if p < 0.10:   return '*'
    return ''


def sep(char='=', w=70):
    print(char * w)


# ---------------------------------------------------------------------------
# Reporting blocks
# ---------------------------------------------------------------------------

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
    """L2: does signal strengthen with more initiators?"""
    col = f'ret_{col_weeks}w'
    if col not in trades.columns or trades.empty:
        return

    print(f'  INITIATOR COUNT BREAKDOWN ({col_weeks}w)')
    print(f'  {"# Initiators":<16} {"N":>5} {"Alpha":>9} {"Win%":>8} {"t-stat":>8}  Sig')
    print('  ' + '-' * 54)

    for n_init in sorted(trades['new_initiations'].unique()):
        subset = trades[trades['new_initiations'] == n_init]
        s      = compute_stats(subset[col])
        label  = f'{int(n_init)} initiators'
        if pd.isna(s['alpha']):
            print(f'  {label:<16} {int(s["n"]):>5} {"--":>9}')
        else:
            stars = sig_stars(s['p_value'])
            print(f'  {label:<16} {int(s["n"]):>5} {s["alpha"]:>+8.2%} '
                  f'{s["win_rate"]:>8.1%} {s["t_stat"]:>8.2f}  {stars}')
    print()


def print_year_breakdown(trades, col_weeks=8):
    """L2: year-by-year stability check."""
    col = f'ret_{col_weeks}w'
    if col not in trades.columns or trades.empty:
        return

    trades = trades.copy()
    trades['year'] = pd.to_datetime(trades['filing_date']).dt.year

    print(f'  YEAR-BY-YEAR BREAKDOWN ({col_weeks}w)')
    print(f'  {"Year":<8} {"N":>5} {"Alpha":>9} {"Win%":>8} {"t-stat":>8}  Sig')
    print('  ' + '-' * 50)

    for yr in sorted(trades['year'].unique()):
        subset = trades[trades['year'] == yr]
        s      = compute_stats(subset[col])
        if pd.isna(s['alpha']):
            print(f'  {yr:<8} {int(s["n"]):>5} {"--":>9}')
        else:
            stars = sig_stars(s['p_value'])
            print(f'  {yr:<8} {int(s["n"]):>5} {s["alpha"]:>+8.2%} '
                  f'{s["win_rate"]:>8.1%} {s["t_stat"]:>8.2f}  {stars}')
    print()


def print_filer_contribution(trades, col_weeks=8):
    """
    L2: which filers contribute the most high-conviction initiations?
    Loads filer-level detail from holdings table and joins to trade returns.
    """
    col = f'ret_{col_weeks}w'
    if col not in trades.columns or trades.empty:
        return

    try:
        detail = load_holdings_detail()
    except Exception:
        return

    if detail.empty:
        return

    # Join trade returns to filer detail on ticker + quarter_end
    trades_copy = trades.copy()
    trades_copy['quarter_end'] = trades_copy['filing_date'].apply(
        lambda d: (d - pd.offsets.QuarterEnd(1)).strftime('%Y-%m-%d')
        if not pd.isna(d) else None
    )

    merged = detail.merge(
        trades_copy[['ticker', 'filing_date', col]],
        on='ticker', how='inner'
    )
    if merged.empty:
        return

    print(f'  FILER CONTRIBUTION ({col_weeks}w) — top initiators by trade count')
    print(f'  {"Filer":<32} {"Initiations":>11} {"Avg Alpha":>10} {"Win%":>7}')
    print('  ' + '-' * 64)

    for filer, grp in merged.groupby('filer_name'):
        rets = grp[col].dropna()
        if len(rets) < 3:
            continue
        print(f'  {filer:<32} {len(rets):>11} {rets.mean():>+9.2%} {(rets>0).mean():>7.1%}')

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='13F Initiation Backtest v2')
    p.add_argument('--min-init', type=int, default=cfg.MIN_NEW_INITIATIONS,
                   help=f'Minimum new initiators (default: {cfg.MIN_NEW_INITIATIONS})')
    p.add_argument('--export',   action='store_true',
                   help='Save trades CSV to results/')
    return p.parse_args()


def main():
    args = parse_args()

    if not DB_PATH.exists():
        print(f'ERROR: {DB_PATH} not found.')
        print('Run:  python3 collect_13f.py --reset')
        sys.exit(1)

    print()
    sep()
    print('  13F INSTITUTIONAL INITIATION BACKTEST  v2')
    print(f'  Run date:      {datetime.today().strftime("%Y-%m-%d")}')
    print(f'  Signal:        >= {args.min_init} new hedge fund initiations in same quarter')
    print(f'  Universe:      Common Stock only (ETFs/funds excluded via OpenFIGI)')
    print(f'  Entry:         {cfg.ENTRY_DELAY_DAYS} day(s) after filing date')
    print(f'  Horizons:      {cfg.HOLD_PERIODS} weeks')
    print(f'  Backtest:      {cfg.BACKTEST_START} to {cfg.BACKTEST_END or "present"}')
    sep()

    print(f'\n  Loading signals...', end=' ', flush=True)
    signals = load_signals(args.min_init, cfg.BACKTEST_START, cfg.BACKTEST_END)
    print(f'{len(signals):,} signals')

    if signals.empty:
        print('  No signals found. Run: python3 collect_13f.py --reset')
        return

    print(f'  Date range:     {signals["filing_date"].min().date()} to {signals["filing_date"].max().date()}')
    print(f'  Unique tickers: {signals["ticker"].nunique():,}')

    trades = compute_forward_returns(signals, cfg.HOLD_PERIODS, cfg.ENTRY_DELAY_DAYS)
    print(f'  Trades with price data: {len(trades):,}')

    if trades.empty:
        print('  No trades computed — check that tickers are populated in DB.')
        print('  Run:  python3 collect_13f.py --map-only')
        return

    print()
    sep()
    print('  L1 RESULTS')
    sep()
    print()

    print_stats_block('ALL signals — long only', trades, cfg.HOLD_PERIODS)

    sep('-')
    print()
    print('  L2 BREAKDOWNS')
    sep('-')
    print()

    ref_weeks = cfg.HOLD_PERIODS[1] if len(cfg.HOLD_PERIODS) > 1 else cfg.HOLD_PERIODS[0]
    print_initiation_breakdown(trades, col_weeks=ref_weeks)
    print_year_breakdown(trades, col_weeks=ref_weeks)
    print_filer_contribution(trades, col_weeks=ref_weeks)

    print('  * p<0.10  ** p<0.05  *** p<0.01')

    if args.export:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts       = datetime.today().strftime('%Y%m%d_%H%M%S')
        out_path = RESULTS_DIR / f'thirteenf_backtest_{ts}.csv'
        trades.to_csv(out_path, index=False)
        print(f'\n  Exported: {out_path}')


if __name__ == '__main__':
    main()
