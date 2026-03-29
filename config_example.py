# 13F Institutional Holdings Backtest — Configuration Example
# Copy this file to config.py and fill in your values.
# config.py is gitignored and should never be committed.

# --- SEC EDGAR settings ---
SEC_USER_AGENT    = 'Your Name youremail@example.com'
SEC_REQUEST_DELAY = 0.15

# --- OpenFIGI CUSIP->ticker mapping ---
# Free API at https://www.openfigi.com/api
# Leave empty to run without a key (25 req/min limit, slower)
# Register for a free key for 250 req/min
OPENFIGI_API_KEY       = ''
OPENFIGI_REQUEST_DELAY = 2.5   # seconds; set to 0.25 with an API key

# --- Database ---
THIRTEENF_DB = 'thirteenf_data.db'

# --- Signal definition ---
MIN_NEW_INITIATIONS = 2
MIN_POSITION_VALUE  = 1_000_000   # $1M minimum per holding

# --- Entry timing ---
ENTRY_DELAY_DAYS = 1

# --- Forward return horizons (weeks) ---
HOLD_PERIODS = [4, 8, 13, 26]

# --- Backtest date range ---
BACKTEST_START = '2018-01-01'
BACKTEST_END   = None

# --- Collection ---
COLLECT_QUARTERS = 28

# --- Output ---
RESULTS_DIR = 'results'
