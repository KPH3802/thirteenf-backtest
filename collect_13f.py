#!/usr/bin/env python3
"""
13F Data Collector — v2 (Redesigned)
======================================
Downloads 13F institutional holdings from SEC EDGAR for a curated list
of quality hedge funds (active stock-pickers only — no index managers).

Key improvements over v1:
  - Hedge fund filer list only (Vanguard/BlackRock/State Street/mutual funds removed)
  - OpenFIGI CUSIP → ticker mapping (replaces broken stub from v1)
  - ETF/fund filter: only 'Common Stock' holdings enter the backtest
  - Two-phase collection: (1) raw holdings → (2) CUSIP map → (3) rebuild signals
  - --reset flag wipes DB and rebuilds from scratch

Usage:
  python3 collect_13f.py --reset             # Wipe DB and rebuild (START HERE)
  python3 collect_13f.py --map-only          # Re-run CUSIP mapping on existing data
  python3 collect_13f.py --validate          # Show DB summary only
  python3 collect_13f.py --quarters 4        # Last 4 quarters only (fast test)
"""

import sys
import time
import json
import sqlite3
import argparse
import urllib.request
import urllib.error
import re
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

SCRIPT_DIR = Path(__file__).parent
DB_PATH    = SCRIPT_DIR / cfg.THIRTEENF_DB

# ---------------------------------------------------------------------------
# Filer universe — hedge funds and active managers ONLY
# CIKs verified via SEC EDGAR submissions API.
# To add more funds: look up CIK at https://www.sec.gov/cgi-bin/browse-edgar
# ---------------------------------------------------------------------------

# CIKs verified via SEC EDGAR in March 2026.
# Quant/multi-strategy funds excluded (RenTech/Citadel/AQR/Millennium/D.E.Shaw/Balyasny):
# their 'new initiations' are algorithmic rotation noise, not conviction signals.
# Only concentrated stock-pickers with typically <300 holdings per quarter.
HEDGE_FUND_FILERS = [
    # Concentrated long-only and activist funds
    ('0001067983', 'Berkshire Hathaway'),           # ~120 holdings
    ('0001336528', 'Pershing Square Capital Mgmt'), # ~11 holdings (confirmed)
    ('0001103804', 'Viking Global Investors'),       # ~40-60 holdings
    ('0001135730', 'Coatue Management'),             # ~60-80 holdings
    ('0001061165', 'Lone Pine Capital'),             # ~50-60 holdings
    ('0001040273', 'Third Point LLC'),               # ~15-25 holdings; stopped filing after Q1 2024
    ('0001656456', 'Appaloosa LP'),                  # ~10-20 holdings
    ('0000934639', 'Maverick Capital'),              # ~60-80 holdings
    ('0001079114', 'Greenlight Capital'),            # ~30-40 holdings
    # Balyasny removed: 1,250-3,526 positions/quarter (pod shop, not conviction-based)
    ('0001061768', 'Baupost Group'),                 # ~20-30 holdings
    ('0001138995', 'Glenview Capital Management'),   # ~50-70 holdings
    ('0000909661', 'Farallon Capital Management'),   # ~50-80 holdings
    ('0000921669', 'Carl C. Icahn / Icahn Associates'), # ~15-25 holdings; corrected CIK (was 0001412093)
    ('0001388838', 'Tiger Global Management'),       # ~20-30 holdings
    ('0001045810', 'Soros Fund Management'),         # ~5-10 holdings
    ('0001543160', 'Point72 Asset Management'),      # ~15-40 holdings
    ('0001159159', 'Paulson & Co'),                  # ~10-50 holdings
]


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

def create_database(reset=False):
    if reset and DB_PATH.exists():
        DB_PATH.unlink()
        print(f'  Wiped existing DB: {DB_PATH.name}')

    conn = sqlite3.connect(str(DB_PATH))
    cur  = conn.cursor()

    # Raw holdings — one row per filer/quarter/cusip
    cur.execute("""
        CREATE TABLE IF NOT EXISTS holdings (
            filer_cik       TEXT NOT NULL,
            filer_name      TEXT,
            quarter_end     TEXT NOT NULL,
            filing_date     TEXT NOT NULL,
            cusip           TEXT NOT NULL,
            company_name    TEXT,
            ticker          TEXT,
            security_type   TEXT,
            value_usd       REAL,
            shares          INTEGER,
            is_new          INTEGER DEFAULT 0,
            PRIMARY KEY (filer_cik, quarter_end, cusip)
        )
    """)

    # CUSIP lookup cache — persists across runs, never re-fetched
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cusip_map (
            cusip           TEXT PRIMARY KEY,
            ticker          TEXT,
            security_type   TEXT,
            figi_name       TEXT,
            mapped_at       TEXT
        )
    """)

    # Aggregated initiation signals — rebuilt after every collection+mapping run
    cur.execute("""
        CREATE TABLE IF NOT EXISTS initiation_signals (
            ticker              TEXT NOT NULL,
            quarter_end         TEXT NOT NULL,
            filing_date         TEXT NOT NULL,
            new_initiations     INTEGER,
            total_holders       INTEGER,
            total_value_usd     REAL,
            PRIMARY KEY (ticker, quarter_end)
        )
    """)

    cur.execute('CREATE INDEX IF NOT EXISTS idx_holdings_ticker   ON holdings(ticker)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_holdings_quarter  ON holdings(quarter_end)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_holdings_filer    ON holdings(filer_cik)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_holdings_is_new   ON holdings(is_new)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_signals_date      ON initiation_signals(filing_date)')

    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# SEC EDGAR helpers
# ---------------------------------------------------------------------------

def sec_get(url, retries=3):
    headers = {'User-Agent': cfg.SEC_USER_AGENT, 'Accept': 'application/json'}
    for attempt in range(retries):
        try:
            req  = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req, timeout=30)
            time.sleep(cfg.SEC_REQUEST_DELAY)
            return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                print('  Rate limited, sleeping 60s...')
                time.sleep(60)
            elif e.code == 404:
                return None
            else:
                time.sleep(2 * (attempt + 1))
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def get_quarterly_13f_filings(cik, num_quarters):
    """Return the most recent N 13F-HR filings for a given CIK."""
    url         = f'https://data.sec.gov/submissions/CIK{cik}.json'
    submissions = sec_get(url)
    if not submissions:
        return []

    recent         = submissions.get('filings', {}).get('recent', {})
    form_types     = recent.get('form', [])
    filing_dates   = recent.get('filingDate', [])
    accession_nums = recent.get('accessionNumber', [])
    period_reports = recent.get('reportDate', [])

    filings = []
    for i, form in enumerate(form_types):
        if form in ('13F-HR', '13F-HR/A'):
            filings.append({
                'form':        form,
                'filing_date': filing_dates[i]                         if i < len(filing_dates)   else '',
                'accession':   accession_nums[i].replace('-', '')      if i < len(accession_nums) else '',
                'period':      period_reports[i]                       if i < len(period_reports) else '',
            })

    filings.sort(key=lambda x: x['filing_date'], reverse=True)
    return filings[:num_quarters]


def parse_13f_xml(cik, accession):
    """Fetch and parse the infotable XML from a 13F-HR filing."""
    index_url  = f'https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/index.json'
    index_data = sec_get(index_url)
    if not index_data:
        return []

    xml_file = None
    for item in index_data.get('directory', {}).get('item', []):
        name = item.get('name', '')
        if name.endswith('.xml') and 'infotable' in name.lower():
            xml_file = name
            break
        elif name.endswith('.xml') and name != 'primary_doc.xml':
            xml_file = name

    if not xml_file:
        return []

    xml_url = f'https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{xml_file}'
    headers = {'User-Agent': cfg.SEC_USER_AGENT}
    try:
        req         = urllib.request.Request(xml_url, headers=headers)
        resp        = urllib.request.urlopen(req, timeout=30)
        xml_content = resp.read().decode('utf-8', errors='ignore')
        time.sleep(cfg.SEC_REQUEST_DELAY)
    except Exception:
        return []

    holdings = []
    entries  = re.findall(r'<infoTable>(.*?)</infoTable>', xml_content, re.DOTALL)

    for entry in entries:
        def get_val(tag, text=entry):
            m = re.search(fr'<{tag}[^>]*>([^<]*)</{tag}>', text)
            return m.group(1).strip() if m else ''

        cusip          = get_val('cusip')
        name_of_issuer = get_val('nameOfIssuer')
        value_str      = get_val('value')
        shares_m       = re.search(r'<sshPrnamt[^>]*>([^<]*)</sshPrnamt>', entry)

        try:
            # XML values are in $thousands — multiply to get actual dollars
            value_usd = float(value_str.replace(',', '')) * 1000 if value_str else 0
        except ValueError:
            value_usd = 0

        try:
            shares = int(shares_m.group(1).replace(',', '')) if shares_m else 0
        except (ValueError, AttributeError):
            shares = 0

        if cusip and value_usd >= cfg.MIN_POSITION_VALUE:
            holdings.append({
                'cusip':        cusip,
                'company_name': name_of_issuer,
                'value_usd':    value_usd,
                'shares':       shares,
            })

    return holdings


# ---------------------------------------------------------------------------
# OpenFIGI CUSIP -> ticker + security type mapping
# ---------------------------------------------------------------------------

def openfigi_batch(cusips):
    """
    Map up to 10 CUSIPs to tickers via OpenFIGI free API.
    Returns: {cusip: {'ticker': str|None, 'security_type': str, 'name': str|None}}

    API docs: https://www.openfigi.com/api
    Rate limits: 25 req/min (no key) / 250 req/min (with free API key)
    No key required for basic use. Set OPENFIGI_API_KEY in config for higher limits.
    """
    url     = 'https://api.openfigi.com/v3/mapping'
    payload = json.dumps([{'idType': 'ID_CUSIP', 'idValue': c} for c in cusips]).encode()
    headers = {'Content-Type': 'application/json'}
    if cfg.OPENFIGI_API_KEY:
        headers['X-OPENFIGI-APIKEY'] = cfg.OPENFIGI_API_KEY

    try:
        req     = urllib.request.Request(url, data=payload, headers=headers, method='POST')
        resp    = urllib.request.urlopen(req, timeout=30)
        results = json.loads(resp.read())
    except Exception as e:
        print(f'    OpenFIGI error: {e}')
        return {}

    # US equity exchange codes
    us_exchanges = {'US', 'UN', 'UW', 'UA', 'UR', 'UF'}

    mapping = {}
    for i, result in enumerate(results):
        if i >= len(cusips):
            break
        cusip     = cusips[i]
        data_list = result.get('data', [])

        if not data_list:
            mapping[cusip] = {
                'ticker':        None,
                'security_type': result.get('warning', 'No match'),
                'name':          None,
            }
            continue

        # Prefer US-listed Common Stock; fall back to first result
        chosen = None
        for item in data_list:
            if (item.get('securityType') == 'Common Stock'
                    and item.get('exchCode', '') in us_exchanges):
                chosen = item
                break
        if not chosen:
            chosen = data_list[0]

        mapping[cusip] = {
            'ticker':        chosen.get('ticker'),
            'security_type': chosen.get('securityType'),
            'name':          chosen.get('name'),
        }

    return mapping


def build_cusip_map(conn):
    """
    Look up all unmapped CUSIPs in holdings via OpenFIGI.
    Results cached in cusip_map table — idempotent, skips already-mapped CUSIPs.
    After mapping, backfills ticker + security_type columns in holdings.
    """
    cur = conn.cursor()

    cur.execute("""
        SELECT DISTINCT h.cusip
        FROM   holdings h
        LEFT   JOIN cusip_map m ON h.cusip = m.cusip
        WHERE  m.cusip IS NULL
    """)
    unmapped = [r[0] for r in cur.fetchall()]

    if not unmapped:
        print('  All CUSIPs already mapped.')
        _backfill_tickers(conn)
        return

    total      = len(unmapped)
    batch_size = 10
    delay      = cfg.OPENFIGI_REQUEST_DELAY
    batches    = [unmapped[i:i + batch_size] for i in range(0, total, batch_size)]

    api_note = 'with API key' if cfg.OPENFIGI_API_KEY else 'no API key — 25 req/min'
    print(f'\n  Phase 2: CUSIP mapping ({api_note})')
    print(f'  {total:,} unmapped CUSIPs  |  {len(batches)} requests  |  ~{len(batches) * delay / 60:.0f} min estimated')

    mapped = 0
    for i, batch in enumerate(batches, 1):
        if i % 100 == 0 or i == len(batches):
            print(f'    Batch {i}/{len(batches)} — {mapped} tickers resolved so far...', flush=True)

        result = openfigi_batch(batch)
        now    = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

        for cusip, info in result.items():
            cur.execute("""
                INSERT OR REPLACE INTO cusip_map
                    (cusip, ticker, security_type, figi_name, mapped_at)
                VALUES (?, ?, ?, ?, ?)
            """, (cusip, info['ticker'], info['security_type'], info['name'], now))
            if info['ticker']:
                mapped += 1

        conn.commit()
        time.sleep(delay)

    print(f'  Mapping complete: {mapped}/{total} CUSIPs resolved to tickers.')
    _backfill_tickers(conn)


def _backfill_tickers(conn):
    """Backfill ticker + security_type into holdings from cusip_map."""
    cur = conn.cursor()
    cur.execute("""
        UPDATE holdings
        SET    ticker        = (SELECT ticker        FROM cusip_map WHERE cusip_map.cusip = holdings.cusip),
               security_type = (SELECT security_type FROM cusip_map WHERE cusip_map.cusip = holdings.cusip)
        WHERE  (ticker IS NULL OR ticker = '')
    """)
    conn.commit()
    cur.execute("SELECT COUNT(*) FROM holdings WHERE ticker IS NOT NULL AND ticker != ''")
    n = cur.fetchone()[0]
    print(f'  Holdings backfilled: {n:,} rows now have a ticker.')


# ---------------------------------------------------------------------------
# Collection logic
# ---------------------------------------------------------------------------

def collect_quarter(conn, cik, filer_name, filing):
    """Collect and store all holdings for one filer, one quarter."""
    accession   = filing['accession']
    period      = filing['period']
    filing_date = filing['filing_date']

    if not period or not filing_date or not accession:
        return 0

    cur = conn.cursor()
    cur.execute(
        'SELECT COUNT(*) FROM holdings WHERE filer_cik=? AND quarter_end=?',
        (cik, period)
    )
    if cur.fetchone()[0] > 0:
        return 0  # Already collected this quarter

    holdings = parse_13f_xml(cik, accession)
    if not holdings:
        return 0

    # New initiation = CUSIP not seen in any prior quarter for this filer
    cur.execute("""
        SELECT DISTINCT cusip FROM holdings
        WHERE filer_cik = ? AND quarter_end < ?
    """, (cik, period))
    prev_cusips = {r[0] for r in cur.fetchall()}

    inserted = 0
    for h in holdings:
        cusip  = h['cusip']
        is_new = 1 if cusip not in prev_cusips else 0
        cur.execute("""
            INSERT OR REPLACE INTO holdings
                (filer_cik, filer_name, quarter_end, filing_date, cusip, company_name,
                 ticker, security_type, value_usd, shares, is_new)
            VALUES (?, ?, ?, ?, ?, ?, '', '', ?, ?, ?)
        """, (cik, filer_name, period, filing_date, cusip,
              h['company_name'], h['value_usd'], h['shares'], is_new))
        inserted += 1

    conn.commit()
    return inserted


# ---------------------------------------------------------------------------
# Signal rebuild
# ---------------------------------------------------------------------------

def rebuild_signals(conn):
    """
    Rebuild initiation_signals from holdings.
    Only 'Common Stock' securities included — ETFs, mutual funds, preferreds excluded
    via the security_type field populated by OpenFIGI.
    """
    print('\n  Phase 3: Rebuilding initiation signals (Common Stock only)...')
    cur = conn.cursor()
    cur.execute('DELETE FROM initiation_signals')

    cur.execute("""
        SELECT   ticker,
                 quarter_end,
                 MAX(filing_date)          AS filing_date,
                 SUM(is_new)               AS new_initiations,
                 COUNT(DISTINCT filer_cik) AS total_holders,
                 SUM(value_usd)            AS total_value_usd
        FROM     holdings
        WHERE    is_new = 1
          AND    ticker IS NOT NULL
          AND    ticker != ''
          AND    security_type = 'Common Stock'
        GROUP BY ticker, quarter_end
        HAVING   new_initiations >= ?
    """, (cfg.MIN_NEW_INITIATIONS,))

    rows = cur.fetchall()
    for row in rows:
        ticker, quarter_end, filing_date, new_init, total_holders, total_value = row
        cur.execute("""
            INSERT OR REPLACE INTO initiation_signals
                (ticker, quarter_end, filing_date, new_initiations, total_holders, total_value_usd)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (ticker, quarter_end, filing_date, new_init, total_holders, total_value))

    conn.commit()
    print(f'  {len(rows)} initiation signals built.')


# ---------------------------------------------------------------------------
# DB validation summary
# ---------------------------------------------------------------------------

def validate_db(conn):
    cur = conn.cursor()

    cur.execute('SELECT COUNT(*) FROM holdings')
    total = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM holdings WHERE security_type = 'Common Stock'")
    common = cur.fetchone()[0]

    cur.execute("""
        SELECT COUNT(*) FROM holdings
        WHERE  security_type NOT IN ('Common Stock', '')
          AND  security_type IS NOT NULL
    """)
    filtered = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM holdings WHERE ticker = '' OR ticker IS NULL")
    unmapped_count = cur.fetchone()[0]

    cur.execute('SELECT COUNT(DISTINCT filer_cik) FROM holdings')
    filers = cur.fetchone()[0]

    cur.execute('SELECT COUNT(DISTINCT quarter_end) FROM holdings')
    quarters = cur.fetchone()[0]

    cur.execute('SELECT MIN(quarter_end), MAX(quarter_end) FROM holdings')
    min_q, max_q = cur.fetchone()

    cur.execute('SELECT COUNT(*) FROM cusip_map')
    cusip_count = cur.fetchone()[0]

    cur.execute('SELECT COUNT(*) FROM initiation_signals')
    signals = cur.fetchone()[0]

    print()
    print('=' * 62)
    print('  13F DATABASE SUMMARY (v2)')
    print('=' * 62)
    print(f'  Holdings total:           {total:>8,}')
    print(f'  Common Stock (tradeable): {common:>8,}')
    print(f'  ETF/fund filtered out:    {filtered:>8,}')
    print(f'  Unmapped (no ticker):     {unmapped_count:>8,}')
    print(f'  Unique filers:            {filers:>8}')
    print(f'  Quarters covered:         {quarters:>8}  ({min_q} to {max_q})')
    print(f'  CUSIP map entries:        {cusip_count:>8,}')
    print(f'  Initiation signals:       {signals:>8,}')
    print('=' * 62)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='13F Data Collector v2')
    p.add_argument('--reset',    action='store_true',
                   help='Wipe DB and rebuild from scratch (required for first run)')
    p.add_argument('--map-only', action='store_true',
                   help='Skip collection; re-run CUSIP mapping on existing holdings')
    p.add_argument('--validate', action='store_true',
                   help='Show DB summary and exit')
    p.add_argument('--quarters', type=int, default=cfg.COLLECT_QUARTERS,
                   help=f'Quarters per filer (default: {cfg.COLLECT_QUARTERS})')
    return p.parse_args()


def main():
    args = parse_args()
    conn = create_database(reset=args.reset)

    if args.validate:
        validate_db(conn)
        conn.close()
        return

    if args.map_only:
        print('\n  Skipping collection — running CUSIP mapping only.')
        build_cusip_map(conn)
        rebuild_signals(conn)
        validate_db(conn)
        conn.close()
        return

    # ---- Phase 1: Collect raw holdings ----
    print(f'\n  Phase 1: Collecting 13F holdings')
    print(f'  Filers:             {len(HEDGE_FUND_FILERS)} hedge funds (no index managers)')
    print(f'  Quarters per filer: {args.quarters}')
    print(f'  Min position value: ${cfg.MIN_POSITION_VALUE:,.0f}')
    print(f'  Estimated runtime:  2-4 hours (run overnight)')
    print()

    total_records = 0
    for i, (cik, name) in enumerate(HEDGE_FUND_FILERS, 1):
        print(f'  [{i}/{len(HEDGE_FUND_FILERS)}] {name}  (CIK: {cik})')
        filings = get_quarterly_13f_filings(cik, args.quarters)
        if not filings:
            print(f'    No 13F filings found')
            continue
        for filing in filings:
            n = collect_quarter(conn, cik, name, filing)
            if n > 0:
                print(f'    {filing["period"]}: {n} holdings')
                total_records += n

    print(f'\n  Collection complete. {total_records:,} raw holdings records.')

    # ---- Phase 2: CUSIP -> ticker mapping ----
    build_cusip_map(conn)

    # ---- Phase 3: Rebuild signals ----
    rebuild_signals(conn)
    validate_db(conn)
    conn.close()


if __name__ == '__main__':
    main()
