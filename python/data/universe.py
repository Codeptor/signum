"""Point-in-time S&P 500 universe provider for survivorship-bias-free backtesting.

Addresses Finding #29 from ingestion.py: fetch_sp500_tickers() only returns
current constituents, which introduces survivorship bias in backtests.

Data sources (in priority order):
1. fja05680/sp500 GitHub dataset (1996-present, pre-built snapshots)
2. Wikipedia "Selected changes" table (fallback / delta updates)
3. Local cache for offline operation

Usage::

    provider = SurvivalUniverseProvider()
    tickers_2020 = provider.get_universe(date(2020, 3, 15))
    # Returns the ~500 tickers that were in the S&P 500 on March 15, 2020
"""

import io
import json
import logging
import time
import urllib.request
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from python.data.config import DATA_DIR

logger = logging.getLogger(__name__)

# --- Configuration ---
UNIVERSE_CACHE_DIR = DATA_DIR / "cache" / "universe"
CHANGES_CACHE_PATH = UNIVERSE_CACHE_DIR / "sp500_changes.parquet"
CACHE_META_PATH = UNIVERSE_CACHE_DIR / "universe_meta.json"
CACHE_TTL_DAYS = 7

# fja05680/sp500 raw URL for changes since 2019
GITHUB_CHANGES_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "sp500_changes_since_2019.csv"
)

# Known ticker renames: old -> (new, effective_date)
TICKER_RENAMES: dict[str, tuple[str, str]] = {
    "FB": ("META", "2022-06-09"),
    "DWDP": ("DD", "2019-06-03"),
    "BRK.B": ("BRK-B", "1996-01-01"),
    "BF.B": ("BF-B", "1996-01-01"),
}


class SP500ChangeEvent:
    """A single add/remove event in the S&P 500."""

    __slots__ = ("date", "added", "removed")

    def __init__(self, dt: date, added: list[str], removed: list[str]) -> None:
        self.date = dt
        self.added = added
        self.removed = removed

    def __repr__(self) -> str:
        return f"SP500Change({self.date}, +{self.added}, -{self.removed})"


class SurvivalUniverseProvider:
    """Point-in-time S&P 500 constituent provider.

    Reconstructs the index membership on any historical date by starting
    from the current composition and walking backward through change events.

    Parameters
    ----------
    cache_dir : Path
        Where to store downloaded / processed data.
    auto_download : bool
        If True (default), automatically fetch data on first use.
    """

    def __init__(
        self,
        cache_dir: Path = UNIVERSE_CACHE_DIR,
        auto_download: bool = True,
    ) -> None:
        self._cache_dir = cache_dir
        self._auto_download = auto_download
        self._changes: list[SP500ChangeEvent] | None = None
        self._current_tickers: list[str] | None = None

    def get_universe(self, as_of: date) -> list[str]:
        """Return S&P 500 tickers as of a given date.

        Walks backward from current composition through change events.
        """
        self._ensure_loaded()
        return self._reconstruct_from_changes(as_of)

    def get_universe_range(
        self,
        start: date,
        end: date,
        freq: str = "M",
    ) -> dict[date, list[str]]:
        """Return universes for a range of dates (e.g., monthly rebalance)."""
        dates = pd.date_range(start, end, freq=freq.replace("M", "ME") if freq == "M" else freq)
        return {d.date(): self.get_universe(d.date()) for d in dates}

    def get_additions_removals(
        self,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Return all add/remove events in a date range."""
        self._ensure_loaded()
        rows = []
        for evt in self._changes or []:
            if start <= evt.date <= end:
                for ticker in evt.added:
                    rows.append({"date": evt.date, "ticker": ticker, "action": "added"})
                for ticker in evt.removed:
                    rows.append({"date": evt.date, "ticker": ticker, "action": "removed"})
        return pd.DataFrame(rows)

    def normalize_ticker(self, ticker: str, as_of: date | None = None) -> str:
        """Normalize a ticker for Yahoo Finance compatibility."""
        normalized = ticker.replace(".", "-")
        if as_of is not None and normalized in TICKER_RENAMES:
            new_ticker, effective_str = TICKER_RENAMES[normalized]
            effective = datetime.strptime(effective_str, "%Y-%m-%d").date()
            if as_of >= effective:
                return new_ticker
        return normalized

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._changes is not None:
            return

        if self._is_cache_fresh():
            self._load_from_cache()
        elif self._auto_download:
            self._download_and_cache()
        elif CHANGES_CACHE_PATH.exists():
            logger.warning("Cache stale but auto_download=False; using stale cache")
            self._load_from_cache()
        else:
            raise RuntimeError(
                "No cached universe data and auto_download=False. "
                "Call download_and_cache() manually."
            )

    def _is_cache_fresh(self) -> bool:
        if not CACHE_META_PATH.exists():
            return False
        try:
            meta = json.loads(CACHE_META_PATH.read_text())
            age_days = (time.time() - meta["timestamp"]) / 86400
            return age_days < CACHE_TTL_DAYS
        except Exception:
            return False

    def _load_from_cache(self) -> None:
        if CHANGES_CACHE_PATH.exists():
            df = pd.read_parquet(CHANGES_CACHE_PATH)
            self._changes = []
            for _, row in df.iterrows():
                added = [t for t in row["added"].split(",") if t] if row["added"] else []
                removed = [t for t in row["removed"].split(",") if t] if row["removed"] else []
                self._changes.append(
                    SP500ChangeEvent(row["date"].date(), added, removed)
                )
            self._changes.sort(key=lambda e: e.date)
        else:
            self._changes = []

        logger.info(f"Loaded universe cache: {len(self._changes)} change events")

    def download_and_cache(self) -> None:
        """Public method to force a re-download."""
        self._download_and_cache()

    def _download_and_cache(self) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # 1. Download GitHub changes
        events = self._download_github_changes()

        # 2. Supplement with Wikipedia
        wiki_events = self._scrape_wikipedia_changes()
        events = self._merge_change_events(events, wiki_events)

        # 3. Cache
        self._changes = sorted(events, key=lambda e: e.date)
        changes_df = pd.DataFrame([
            {
                "date": pd.Timestamp(e.date),
                "added": ",".join(e.added),
                "removed": ",".join(e.removed),
            }
            for e in self._changes
        ])
        if not changes_df.empty:
            changes_df.to_parquet(CHANGES_CACHE_PATH)

        # 4. Get current tickers
        self._current_tickers = self._get_current_tickers()

        # 5. Write metadata
        CACHE_META_PATH.write_text(
            json.dumps({"timestamp": time.time(), "n_events": len(self._changes)})
        )

        logger.info(f"Universe cache updated: {len(self._changes)} change events")

    def _download_github_changes(self) -> list[SP500ChangeEvent]:
        """Download sp500_changes_since_2019.csv from fja05680/sp500."""
        events = []
        try:
            req = urllib.request.Request(
                GITHUB_CHANGES_URL,
                headers={"User-Agent": "quant-platform/0.1"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                content = resp.read().decode("utf-8")
            df = pd.read_csv(io.StringIO(content))
            for _, row in df.iterrows():
                dt = pd.to_datetime(row["date"]).date()
                added = (
                    [t.strip().replace(".", "-") for t in str(row["add"]).split(",") if t.strip()]
                    if pd.notna(row.get("add")) and str(row.get("add", "")).strip()
                    else []
                )
                removed = (
                    [t.strip().replace(".", "-") for t in str(row["remove"]).split(",") if t.strip()]
                    if pd.notna(row.get("remove")) and str(row.get("remove", "")).strip()
                    else []
                )
                events.append(SP500ChangeEvent(dt, added, removed))
            logger.info(f"Downloaded {len(events)} change events from GitHub")
        except Exception as e:
            logger.warning(f"Failed to download GitHub changes: {e}")
        return events

    def _scrape_wikipedia_changes(self) -> list[SP500ChangeEvent]:
        """Scrape 'Selected changes' table from Wikipedia."""
        events = []
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            if len(tables) < 2:
                return events

            changes = tables[1]

            # Flatten MultiIndex columns
            if isinstance(changes.columns, pd.MultiIndex):
                changes.columns = [
                    "_".join(str(c) for c in col).strip() for col in changes.columns
                ]

            date_col = _find_column(changes, ["date"])
            added_col = _find_column(changes, ["added", "ticker"])
            removed_col = _find_column(changes, ["removed", "ticker"])

            if not all([date_col, added_col, removed_col]):
                logger.warning(f"Could not identify Wikipedia columns: {list(changes.columns)}")
                return events

            for _, row in changes.iterrows():
                try:
                    dt = pd.to_datetime(row[date_col]).date()
                except Exception:
                    continue

                added_raw = str(row[added_col]).strip()
                removed_raw = str(row[removed_col]).strip()

                added = (
                    [added_raw.replace(".", "-")]
                    if added_raw and added_raw.lower() != "nan"
                    else []
                )
                removed = (
                    [removed_raw.replace(".", "-")]
                    if removed_raw and removed_raw.lower() != "nan"
                    else []
                )

                if added or removed:
                    events.append(SP500ChangeEvent(dt, added, removed))

            logger.info(f"Scraped {len(events)} change events from Wikipedia")
        except Exception as e:
            logger.warning(f"Wikipedia changes scrape failed: {e}")
        return events

    def _get_current_tickers(self) -> list[str]:
        """Get current S&P 500 tickers (reuse existing ingestion logic)."""
        try:
            from python.data.ingestion import fetch_sp500_tickers

            return fetch_sp500_tickers()
        except Exception as e:
            logger.warning(f"Could not fetch current tickers: {e}")
            return []

    # ------------------------------------------------------------------
    # Universe reconstruction
    # ------------------------------------------------------------------

    def _reconstruct_from_changes(self, as_of: date) -> list[str]:
        """Reconstruct S&P 500 on as_of by walking changes backward from today.

        Algorithm:
        1. Start with current membership.
        2. Walk backward through change events from today to as_of.
        3. For each event after as_of: REVERSE it.
        """
        if not self._current_tickers:
            self._current_tickers = self._get_current_tickers()

        universe = set(self._current_tickers)
        today = date.today()

        if not self._changes:
            logger.warning("No change events loaded; returning current universe")
            return sorted(universe)

        for evt in reversed(self._changes):
            if evt.date <= as_of:
                break
            if evt.date > today:
                continue
            # Reverse: un-add = remove, un-remove = add
            for ticker in evt.added:
                universe.discard(ticker)
            for ticker in evt.removed:
                universe.add(ticker)

        return sorted(universe)

    # ------------------------------------------------------------------
    # Merge logic
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_change_events(
        primary: list[SP500ChangeEvent],
        secondary: list[SP500ChangeEvent],
    ) -> list[SP500ChangeEvent]:
        """Merge two lists of change events, deduplicating by (date, ticker)."""
        seen: set[tuple[date, str]] = set()
        merged = []

        for evt in primary:
            key_set = {(evt.date, t) for t in evt.added + evt.removed}
            if not key_set & seen:
                merged.append(evt)
                seen |= key_set

        for evt in secondary:
            key_set = {(evt.date, t) for t in evt.added + evt.removed}
            if not key_set & seen:
                merged.append(evt)
                seen |= key_set

        return sorted(merged, key=lambda e: e.date)


def _find_column(df: pd.DataFrame, keywords: list[str]) -> str | None:
    """Find a column whose name contains ALL of the given keywords."""
    for col in df.columns:
        col_lower = str(col).lower()
        if all(kw in col_lower for kw in keywords):
            return col
    return None
