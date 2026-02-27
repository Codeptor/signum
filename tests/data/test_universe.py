"""Tests for point-in-time S&P 500 universe provider."""

from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

from python.data.universe import (
    SP500ChangeEvent,
    SurvivalUniverseProvider,
    _find_column,
)


# ---------------------------------------------------------------------------
# SP500ChangeEvent
# ---------------------------------------------------------------------------


class TestChangeEvent:
    def test_repr(self):
        evt = SP500ChangeEvent(date(2024, 1, 1), ["AAPL"], ["MSFT"])
        assert "2024-01-01" in repr(evt)
        assert "AAPL" in repr(evt)

    def test_slots(self):
        evt = SP500ChangeEvent(date(2024, 1, 1), ["A"], ["B"])
        assert evt.date == date(2024, 1, 1)
        assert evt.added == ["A"]
        assert evt.removed == ["B"]


# ---------------------------------------------------------------------------
# SurvivalUniverseProvider — unit tests (no network)
# ---------------------------------------------------------------------------


class TestProviderOffline:
    """Tests that don't require network access."""

    def _make_provider(self, changes, current_tickers):
        """Create a provider pre-loaded with mock data."""
        p = SurvivalUniverseProvider(auto_download=False)
        p._changes = sorted(changes, key=lambda e: e.date)
        p._current_tickers = current_tickers
        return p

    def test_no_changes_returns_current(self):
        provider = self._make_provider(
            changes=[],
            current_tickers=["AAPL", "MSFT", "GOOG"],
        )
        result = provider.get_universe(date(2020, 1, 1))
        assert result == ["AAPL", "GOOG", "MSFT"]

    def test_reconstruct_reverses_addition(self):
        """If TSLA was added on 2020-12-21, it shouldn't be in universe before that."""
        provider = self._make_provider(
            changes=[
                SP500ChangeEvent(date(2020, 12, 21), ["TSLA"], []),
            ],
            current_tickers=["AAPL", "MSFT", "TSLA"],
        )
        # Before TSLA was added
        result = provider.get_universe(date(2020, 6, 1))
        assert "TSLA" not in result
        assert "AAPL" in result

    def test_reconstruct_reverses_removal(self):
        """If XOM was removed on 2020-08-31, it should be in universe before that."""
        provider = self._make_provider(
            changes=[
                SP500ChangeEvent(date(2020, 8, 31), [], ["XOM"]),
            ],
            current_tickers=["AAPL", "MSFT"],
        )
        # Before XOM was removed
        result = provider.get_universe(date(2020, 1, 1))
        assert "XOM" in result

    def test_multiple_changes(self):
        provider = self._make_provider(
            changes=[
                SP500ChangeEvent(date(2020, 6, 1), ["A"], ["B"]),
                SP500ChangeEvent(date(2021, 1, 1), ["C"], ["D"]),
            ],
            current_tickers=["A", "C", "E"],
        )
        # Before any changes: undo C add, undo A add; restore B, D
        result = provider.get_universe(date(2019, 1, 1))
        assert "A" not in result
        assert "C" not in result
        assert "B" in result
        assert "D" in result
        assert "E" in result

    def test_universe_on_change_date_includes_addition(self):
        """On the date of addition, the ticker should be present."""
        provider = self._make_provider(
            changes=[
                SP500ChangeEvent(date(2020, 12, 21), ["TSLA"], []),
            ],
            current_tickers=["AAPL", "TSLA"],
        )
        result = provider.get_universe(date(2020, 12, 21))
        assert "TSLA" in result

    def test_get_additions_removals(self):
        provider = self._make_provider(
            changes=[
                SP500ChangeEvent(date(2020, 1, 15), ["A"], ["B"]),
                SP500ChangeEvent(date(2020, 6, 1), ["C"], []),
                SP500ChangeEvent(date(2021, 1, 1), ["D"], ["E"]),
            ],
            current_tickers=["A", "C", "D"],
        )
        df = provider.get_additions_removals(date(2020, 1, 1), date(2020, 12, 31))
        assert len(df) == 3  # A added, B removed, C added
        assert set(df["ticker"]) == {"A", "B", "C"}

    def test_get_universe_range(self):
        provider = self._make_provider(
            changes=[],
            current_tickers=["AAPL", "MSFT"],
        )
        result = provider.get_universe_range(date(2024, 1, 1), date(2024, 3, 31), freq="M")
        assert len(result) == 3  # Jan, Feb, Mar month-ends


# ---------------------------------------------------------------------------
# Normalize ticker
# ---------------------------------------------------------------------------


class TestNormalizeTicker:
    def test_dot_to_hyphen(self):
        provider = SurvivalUniverseProvider(auto_download=False)
        assert provider.normalize_ticker("BRK.B") == "BRK-B"

    def test_known_rename_after_effective(self):
        provider = SurvivalUniverseProvider(auto_download=False)
        assert provider.normalize_ticker("FB", as_of=date(2023, 1, 1)) == "META"

    def test_known_rename_before_effective(self):
        provider = SurvivalUniverseProvider(auto_download=False)
        assert provider.normalize_ticker("FB", as_of=date(2020, 1, 1)) == "FB"


# ---------------------------------------------------------------------------
# Merge change events
# ---------------------------------------------------------------------------


class TestMergeEvents:
    def test_deduplicates(self):
        evt1 = SP500ChangeEvent(date(2020, 1, 1), ["A"], [])
        evt2 = SP500ChangeEvent(date(2020, 1, 1), ["A"], [])
        merged = SurvivalUniverseProvider._merge_change_events([evt1], [evt2])
        assert len(merged) == 1

    def test_keeps_both_different_dates(self):
        evt1 = SP500ChangeEvent(date(2020, 1, 1), ["A"], [])
        evt2 = SP500ChangeEvent(date(2020, 6, 1), ["B"], [])
        merged = SurvivalUniverseProvider._merge_change_events([evt1], [evt2])
        assert len(merged) == 2

    def test_sorted_by_date(self):
        evt1 = SP500ChangeEvent(date(2021, 1, 1), ["B"], [])
        evt2 = SP500ChangeEvent(date(2020, 1, 1), ["A"], [])
        merged = SurvivalUniverseProvider._merge_change_events([evt1], [evt2])
        assert merged[0].date < merged[1].date


# ---------------------------------------------------------------------------
# Helper: _find_column
# ---------------------------------------------------------------------------


class TestFindColumn:
    def test_finds_matching_column(self):
        df = pd.DataFrame(columns=["Date_Date", "Added_Ticker", "Removed_Ticker"])
        assert _find_column(df, ["added", "ticker"]) == "Added_Ticker"

    def test_returns_none_when_not_found(self):
        df = pd.DataFrame(columns=["foo", "bar"])
        assert _find_column(df, ["missing"]) is None
