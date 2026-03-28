#!/usr/bin/env python3
"""
13F Data Collector
==================
Downloads 13F institutional holdings filings from SEC EDGAR and builds
a database of new position initiations by quarter.

SEC EDGAR 13F data is freely available via the submissions API.
Same pipeline as 8-K and Form 4 scanners.

This is a long-running script -- collecting 28 quarters of 13F data
from thousands of filers takes 2-4 hours. Run overnight.

Usage:
  python3 collect_13f.py              # Collect all quarters
  python3 collect_13f.py --validate   # Show DB summary
  python3 collect_13f.py --quarters 4 # Last 4 quarters only (fast test)
"""

import sys
import time
import sqlite3
import argparse
import json
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, date
import re

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

SCRIPT_DIR = Path(__file__).parent
DB_PATH = SCRIPT_DIR / cfg.THIRTEENF_DB


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

def create_database():
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    # Raw holdings per filer per quarter
    cur.execute("""
        CREATE TABLE IF NOT EXISTS holdings (
            filer_cik       TEXT NOT NULL,
            quarter_end     TEXT NOT NULL,
            filing_date     TEXT NOT NULL,
            ticker          TEXT,
            cusip           TEXT,
            company_name    TEXT,
            value_usd       REAL,
            shares          INTEGER,
            is_new          INTEGER DEFAULT 0,
            PRIMARY KEY (filer_cik, quarter_end, cusip)
        )
    """)

    # Aggregated initiation signals
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

    cur.execute('CREATE INDEX IF NOT EXISTS idx_holdings_ticker ON holdings(ticker)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_holdings_quarter ON holdings(quarter_end)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_signals_date ON initiation_signals(filing_date)')

    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# SEC EDGAR API
# ---------------------------------------------------------------------------

def sec_get(url, retries=3):
    """SEC EDGAR GET with rate limiting and retry."""
    headers = {'User-Agent': cfg.SEC_USER_AGENT, 'Accept': 'application/json'}
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req, timeout=30)
            time.sleep(cfg.SEC_REQUEST_DELAY)
            return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                print(f'  Rate limited, waiting 60s...')
                time.sleep(60)
            elif e.code == 404:
                return None
            else:
                time.sleep(2 * (attempt + 1))
        except Exception as e:
            time.sleep(2 * (attempt + 1))
    return None


def get_13f_filers():
    """Get list of large institutional 13F filers from SEC EDGAR company search."""
    # Use SEC full-text search to find recent 13F-HR filers
    url = 'https://efts.sec.gov/LATEST/search-index?q=%2213F-HR%22&dateRange=custom&startdt=2025-10-01&enddt=2026-02-01&forms=13F-HR&hits.hits._source=period_of_report,entity_name,file_num&hits.hits.total=true'

    # Alternative: use the company tickers file to get CIKs, then filter for 13F filers
    # More reliable approach: get from SEC submissions
    url2 = 'https://data.sec.gov/submissions/CIK0001067983.json'  # Berkshire as test

    # Best approach: use EDGAR full text search for 13F-HR filings
    search_url = 'https://efts.sec.gov/LATEST/search-index?forms=13F-HR&dateRange=custom&startdt=2024-10-01&enddt=2025-02-28&hits.hits.total=true&hits.hits._source=cik,entity_name,period_of_report,filed_at'

    data = sec_get(search_url)
    if not data:
        return []

    filers = []
    hits = data.get('hits', {}).get('hits', [])
    for hit in hits:
        src = hit.get('_source', {})
        cik = src.get('cik', '')
        name = src.get('entity_name', '')
        if cik:
            filers.append({'cik': cik.zfill(10), 'name': name})

    return filers


def get_filer_submissions(cik):
    """Get all filings for a given CIK."""
    url = f'https://data.sec.gov/submissions/CIK{cik}.json'
    return sec_get(url)


def get_quarterly_13f_filings(cik, num_quarters):
    """Get the most recent N quarterly 13F filings for a CIK."""
    submissions = get_filer_submissions(cik)
    if not submissions:
        return []

    recent = submissions.get('filings', {}).get('recent', {})
    form_types = recent.get('form', [])
    filing_dates = recent.get('filingDate', [])
    accession_nums = recent.get('accessionNumber', [])
    period_of_reports = recent.get('reportDate', [])

    filings = []
    for i, form in enumerate(form_types):
        if form in ('13F-HR', '13F-HR/A') and i < len(accession_nums):
            filings.append({
                'form': form,
                'filing_date': filing_dates[i] if i < len(filing_dates) else '',
                'accession': accession_nums[i].replace('-', ''),
                'period': period_of_reports[i] if i < len(period_of_reports) else '',
            })

    # Sort by filing date, take most recent N
    filings.sort(key=lambda x: x['filing_date'], reverse=True)
    return filings[:num_quarters]


def parse_13f_xml(cik, accession):
    """Parse 13F-HR XML to extract holdings."""
    # Try to get the primary document index
    acc_formatted = f'{accession[:10]}-{accession[10:12]}-{accession[12:]}'
    index_url = f'https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/index.json'

    index_data = sec_get(index_url)
    if not index_data:
        return []

    # Find the XML information table
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
        req = urllib.request.Request(xml_url, headers=headers)
        resp = urllib.request.urlopen(req, timeout=30)
        xml_content = resp.read().decode('utf-8', errors='ignore')
        time.sleep(cfg.SEC_REQUEST_DELAY)
    except Exception:
        return []

    # Parse holdings from XML
    holdings = []
    # Match infoTable entries
    entries = re.findall(r'<infoTable>(.*?)</infoTable>', xml_content, re.DOTALL)
    for entry in entries:
        def get_val(tag, text):
            m = re.search(f'<{tag}[^>]*>([^<]*)</{tag}>', text)
            return m.group(1).strip() if m else ''

        name_of_issuer = get_val('nameOfIssuer', entry)
        cusip = get_val('cusip', entry)
        value_str = get_val('value', entry)
        shares_str = re.search(r'<sshPrnamt[^>]*>([^<]*)</sshPrnamt>', entry)
        ticker = ''  # 13F doesn't include ticker, we'll map from CUSIP later

        try:
            value = float(value_str.replace(',', '')) * 1000 if value_str else 0  # values in thousands
        except ValueError:
            value = 0

        try:
            shares = int(shares_str.group(1).replace(',', '')) if shares_str else 0
        except (ValueError, AttributeError):
            shares = 0

        if cusip and value >= cfg.MIN_POSITION_VALUE / 1000:  # filter small positions
            holdings.append({
                'cusip': cusip,
                'company_name': name_of_issuer,
                'value_usd': value,
                'shares': shares,
            })

    return holdings


# ---------------------------------------------------------------------------
# CUSIP to ticker mapping
# ---------------------------------------------------------------------------

_cusip_cache = {}

def cusip_to_ticker(cusip):
    """Map CUSIP to ticker using SEC company tickers file."""
    global _cusip_cache
    if not _cusip_cache:
        try:
            url = 'https://www.sec.gov/files/company_tickers_exchange.json'
            headers = {'User-Agent': cfg.SEC_USER_AGENT}
            req = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req, timeout=30)
            data = json.loads(resp.read())
            # Build CUSIP -> ticker lookup (partial match on first 8 chars)
            for item in data.get('data', []):
                # Format: [cik, name, ticker, exchange]
                if len(item) >= 3:
                    _cusip_cache[item[1].upper()[:8]] = item[2]
            time.sleep(cfg.SEC_REQUEST_DELAY)
        except Exception:
            pass
    # Try to match by company name prefix (imperfect but useful)
    return None  # Return None -- we'll use company name as fallback


# ---------------------------------------------------------------------------
# Main collection logic
# ---------------------------------------------------------------------------

def get_large_filers_list():
    """Get a list of well-known large institutional filers (CIKs)."""
    # Hardcoded list of major institutional investors -- reliable and fast
    # These are the filers whose 13F data matters most for signal quality
    return [
        ('0001067983', 'Berkshire Hathaway'),
        ('0000102909', 'Vanguard Group'),
        ('0000315066', 'BlackRock'),
        ('0000093751', 'State Street'),
        ('0001336528', 'Citadel Advisors'),
        ('0001423053', 'Renaissance Technologies'),
        ('0000088053', 'Goldman Sachs'),
        ('0000895421', 'Morgan Stanley'),
        ('0000019617', 'JPMorgan Chase'),
        ('0000070858', 'Bank of America'),
        ('0001109357', 'Two Sigma'),
        ('0001582202', 'Bridgewater Associates'),
        ('0001061219', 'D.E. Shaw'),
        ('0000906107', 'Tiger Management'),
        ('0001603459', 'Viking Global'),
        ('0001456655', 'Coatue Management'),
        ('0001502554', 'Lone Pine Capital'),
        ('0001418819', 'Third Point'),
        ('0001037389', 'Pershing Square'),
        ('0001166408', 'Appaloosa Management'),
        ('0001649289', 'Melvin Capital'),
        ('0001336917', 'Maverick Capital'),
        ('0001326380', 'Greenlight Capital'),
        ('0001045810', 'Soros Fund Management'),
        ('0001365135', 'Balyasny Asset Management'),
        ('0001541119', 'Millennium Management'),
        ('0001543160', 'Point72 Asset Management'),
        ('0001099590', 'AQR Capital'),
        ('0001061219', 'DE Shaw'),
        ('0001035520', 'Fidelity'),
        ('0001454939', 'T. Rowe Price'),
        ('0000100885', 'Wellington Management'),
        ('0000049196', 'Putnam Investments'),
        ('0000807985', 'Capital Research'),
        ('0000884905', 'Dodge & Cox'),
        ('0001350487', 'Baupost Group'),
        ('0001336528', 'Citadel'),
        ('0001603869', 'Glenview Capital'),
        ('0001159159', 'Paulson & Co'),
        ('0001037540', 'Farallon Capital'),
    ]


def collect_quarter(conn, cik, filer_name, filing):
    """Collect holdings for one filer, one quarter."""
    accession = filing['accession']
    period = filing['period']
    filing_date = filing['filing_date']

    if not period or not filing_date:
        return 0

    # Check if already collected
    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) FROM holdings WHERE filer_cik=? AND quarter_end=?', (cik, period))
    if cur.fetchone()[0] > 0:
        return 0  # Already have this quarter

    holdings = parse_13f_xml(cik, accession)
    if not holdings:
        return 0

    # Get previous quarter holdings for this filer to identify new positions
    cur.execute("""
        SELECT DISTINCT cusip FROM holdings
        WHERE filer_cik=? AND quarter_end < ?
        ORDER BY quarter_end DESC
        LIMIT 10000
    """, (cik, period))
    prev_cusips = {r[0] for r in cur.fetchall()}

    inserted = 0
    for h in holdings:
        cusip = h['cusip']
        is_new = 1 if cusip not in prev_cusips else 0
        cur.execute("""
            INSERT OR REPLACE INTO holdings
            (filer_cik, quarter_end, filing_date, ticker, cusip, company_name, value_usd, shares, is_new)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (cik, period, filing_date, '', cusip, h['company_name'], h['value_usd'], h['shares'], is_new))
        inserted += 1

    conn.commit()
    return inserted


def rebuild_signals(conn):
    """Rebuild the initiation_signals table from raw holdings."""
    print('  Rebuilding initiation signals...')
    cur = conn.cursor()
    cur.execute('DELETE FROM initiation_signals')

    # Count new initiations per company name per quarter
    cur.execute("""
        SELECT company_name, quarter_end,
               MAX(filing_date) as filing_date,
               SUM(is_new) as new_initiations,
               COUNT(DISTINCT filer_cik) as total_holders,
               SUM(value_usd) as total_value_usd
        FROM holdings
        WHERE is_new = 1
        GROUP BY company_name, quarter_end
        HAVING new_initiations >= ?
    """, (cfg.MIN_NEW_INITIATIONS,))

    rows = cur.fetchall()
    for row in rows:
        company_name, quarter_end, filing_date, new_init, total_holders, total_value = row
        # Use company name as ticker placeholder -- will need CUSIP->ticker mapping
        # For now, try to extract a clean name
        ticker = company_name[:20].strip() if company_name else ''
        cur.execute("""
            INSERT OR REPLACE INTO initiation_signals
            (ticker, quarter_end, filing_date, new_initiations, total_holders, total_value_usd)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (ticker, quarter_end, filing_date, new_init, total_holders, total_value))

    conn.commit()
    print(f'  {len(rows)} initiation signals built')


def validate_db(conn):
    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) FROM holdings')
    total = cur.fetchone()[0]
    cur.execute('SELECT COUNT(DISTINCT filer_cik) FROM holdings')
    filers = cur.fetchone()[0]
    cur.execute('SELECT COUNT(DISTINCT quarter_end) FROM holdings')
    quarters = cur.fetchone()[0]
    cur.execute('SELECT COUNT(*) FROM initiation_signals')
    signals = cur.fetchone()[0]
    cur.execute('SELECT MIN(quarter_end), MAX(quarter_end) FROM holdings')
    min_q, max_q = cur.fetchone()

    print()
    print('=' * 60)
    print('  13F DATABASE SUMMARY')
    print('=' * 60)
    print(f'  Holdings records: {total:,}')
    print(f'  Unique filers:    {filers}')
    print(f'  Quarters covered: {quarters} ({min_q} to {max_q})')
    print(f'  Init signals:     {signals:,}')
    print('=' * 60)


def parse_args():
    parser = argparse.ArgumentParser(description='13F Data Collector')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--quarters', type=int, default=cfg.COLLECT_QUARTERS,
                        help=f'Number of quarters to collect (default: {cfg.COLLECT_QUARTERS})')
    return parser.parse_args()


def main():
    args = parse_args()
    conn = create_database()

    if args.validate:
        validate_db(conn)
        conn.close()
        return

    filers = get_large_filers_list()
    print(f'\n  Collecting 13F data for {len(filers)} institutional filers')
    print(f'  Last {args.quarters} quarters per filer')
    print(f'  This will take 1-3 hours -- run overnight')
    print()

    total_records = 0
    for i, (cik, name) in enumerate(filers, 1):
        print(f'  [{i}/{len(filers)}] {name} (CIK: {cik})')

        filings = get_quarterly_13f_filings(cik, args.quarters)
        if not filings:
            print(f'    No 13F filings found')
            continue

        for filing in filings:
            n = collect_quarter(conn, cik, name, filing)
            if n > 0:
                print(f'    {filing["period"]}: {n} holdings')
                total_records += n

    print(f'\n  Collection complete. {total_records:,} holdings records.')
    rebuild_signals(conn)
    validate_db(conn)
    conn.close()


if __name__ == '__main__':
    main()
