from pathlib import Path

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

SP500_UNIVERSE = "sp500"
DEFAULT_PERIOD = "5y"
DEFAULT_INTERVAL = "1d"

# Single source of truth for risk-free rate across the entire codebase.
# Used in Sharpe/Sortino calculations, risk engine, and backtests.
# As of 2025, the 10-year Treasury yield is ~4.3%, so 5% is conservative.
RISK_FREE_RATE = 0.05
