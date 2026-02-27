"""Tests for Transaction Cost Analysis (TCA) module."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from python.portfolio.tca import TradeRecord, TransactionCostAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts(hour=10):
    """Create a UTC timestamp."""
    return datetime(2024, 6, 15, hour, 0, 0, tzinfo=timezone.utc)


def _make_trade(
    symbol="AAPL",
    side="BUY",
    order_qty=100,
    fill_qty=100,
    fill_price=150.0,
    decision_price=149.5,
    commission=1.50,
    vwap=149.8,
    hour=10,
) -> TradeRecord:
    return TradeRecord(
        symbol=symbol,
        side=side,
        order_qty=order_qty,
        fill_qty=fill_qty,
        fill_price=fill_price,
        decision_price=decision_price,
        timestamp=_ts(hour),
        commission=commission,
        vwap=vwap,
    )


# ---------------------------------------------------------------------------
# Tests: Per-trade metric calculations
# ---------------------------------------------------------------------------


class TestPerTradeMetrics:
    def test_implementation_shortfall_buy(self):
        """Buy: paid more than decision → positive IS."""
        is_bps = TransactionCostAnalyzer.implementation_shortfall(150.5, 150.0, "BUY")
        expected = (150.5 - 150.0) / 150.0 * 10_000  # ~33.3 bps
        assert is_bps == pytest.approx(expected, rel=1e-6)

    def test_implementation_shortfall_sell(self):
        """Sell: received less than decision → positive IS."""
        is_bps = TransactionCostAnalyzer.implementation_shortfall(149.5, 150.0, "SELL")
        # raw = (149.5 - 150.0) / 150.0 * 10000 = -33.3, negate for sell
        expected = -((149.5 - 150.0) / 150.0 * 10_000)  # ~33.3 bps
        assert is_bps == pytest.approx(expected, rel=1e-6)

    def test_implementation_shortfall_zero_decision(self):
        """Zero decision price should return 0.0."""
        assert TransactionCostAnalyzer.implementation_shortfall(100.0, 0.0, "BUY") == 0.0

    def test_vwap_slippage_positive(self):
        """Fill above VWAP → positive slippage."""
        slip = TransactionCostAnalyzer.vwap_slippage_bps(150.3, 150.0)
        expected = (150.3 - 150.0) / 150.0 * 10_000  # 20 bps
        assert slip == pytest.approx(expected, rel=1e-6)

    def test_vwap_slippage_negative(self):
        """Fill below VWAP → negative slippage."""
        slip = TransactionCostAnalyzer.vwap_slippage_bps(149.7, 150.0)
        assert slip < 0

    def test_vwap_slippage_zero_vwap(self):
        """Zero VWAP should return 0.0."""
        assert TransactionCostAnalyzer.vwap_slippage_bps(100.0, 0.0) == 0.0

    def test_fill_rate_full(self):
        assert TransactionCostAnalyzer.fill_rate(100, 100) == 1.0

    def test_fill_rate_partial(self):
        assert TransactionCostAnalyzer.fill_rate(75, 100) == pytest.approx(0.75)

    def test_fill_rate_zero_order(self):
        assert TransactionCostAnalyzer.fill_rate(50, 0) == 0.0

    def test_fill_rate_capped_at_one(self):
        """Overfill should cap at 1.0."""
        assert TransactionCostAnalyzer.fill_rate(110, 100) == 1.0


# ---------------------------------------------------------------------------
# Tests: Analyzer initialization
# ---------------------------------------------------------------------------


class TestAnalyzerInit:
    def test_empty_init(self):
        tca = TransactionCostAnalyzer()
        assert tca.n_trades == 0

    def test_init_with_trades(self):
        trades = [_make_trade(), _make_trade(symbol="MSFT")]
        tca = TransactionCostAnalyzer(trades=trades)
        assert tca.n_trades == 2

    def test_add_trade(self):
        tca = TransactionCostAnalyzer()
        tca.add_trade(_make_trade())
        assert tca.n_trades == 1

    def test_add_trades_from_df(self):
        df = pd.DataFrame([
            {
                "symbol": "AAPL", "side": "BUY", "order_qty": 100,
                "fill_qty": 100, "fill_price": 150.0,
                "decision_price": 149.5, "timestamp": _ts(),
                "commission": 1.5, "vwap": 149.8,
            },
            {
                "symbol": "MSFT", "side": "SELL", "order_qty": 50,
                "fill_qty": 50, "fill_price": 400.0,
                "decision_price": 401.0, "timestamp": _ts(11),
                "commission": 2.0, "vwap": 400.5,
            },
        ])
        tca = TransactionCostAnalyzer()
        tca.add_trades_from_df(df)
        assert tca.n_trades == 2

    def test_add_trades_from_df_missing_columns(self):
        df = pd.DataFrame({"symbol": ["AAPL"], "price": [100]})
        tca = TransactionCostAnalyzer()
        with pytest.raises(ValueError, match="Missing columns"):
            tca.add_trades_from_df(df)


# ---------------------------------------------------------------------------
# Tests: Aggregate analysis
# ---------------------------------------------------------------------------


class TestAnalyze:
    def test_analyze_empty(self):
        tca = TransactionCostAnalyzer()
        df = tca.analyze()
        assert df.empty

    def test_analyze_columns(self):
        tca = TransactionCostAnalyzer(trades=[_make_trade()])
        df = tca.analyze()
        expected_cols = {
            "symbol", "side", "fill_price", "decision_price",
            "fill_qty", "order_qty", "timestamp", "commission",
            "is_bps", "vwap_slip_bps", "fill_rate", "trade_value",
            "commission_bps", "total_cost_bps",
        }
        assert set(df.columns) == expected_cols

    def test_analyze_trade_value(self):
        trade = _make_trade(fill_price=150.0, fill_qty=100)
        tca = TransactionCostAnalyzer(trades=[trade])
        df = tca.analyze()
        assert df.iloc[0]["trade_value"] == pytest.approx(15000.0)

    def test_analyze_commission_bps(self):
        trade = _make_trade(fill_price=100.0, fill_qty=100, commission=10.0)
        tca = TransactionCostAnalyzer(trades=[trade])
        df = tca.analyze()
        # commission_bps = 10 / (100*100) * 10000 = 10 bps
        assert df.iloc[0]["commission_bps"] == pytest.approx(10.0)

    def test_analyze_no_vwap(self):
        trade = _make_trade(vwap=None)
        tca = TransactionCostAnalyzer(trades=[trade])
        df = tca.analyze()
        assert np.isnan(df.iloc[0]["vwap_slip_bps"])


# ---------------------------------------------------------------------------
# Tests: Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_empty(self):
        tca = TransactionCostAnalyzer()
        s = tca.summary()
        assert s["n_trades"] == 0

    def test_summary_basic(self):
        trades = [
            _make_trade(symbol="AAPL", side="BUY"),
            _make_trade(symbol="MSFT", side="SELL", fill_price=400.0,
                        decision_price=401.0, vwap=400.5),
        ]
        tca = TransactionCostAnalyzer(trades=trades)
        s = tca.summary()
        assert s["n_trades"] == 2
        assert s["total_volume"] > 0
        assert "mean_is_bps" in s
        assert "mean_fill_rate" in s

    def test_summary_side_breakdown(self):
        trades = [
            _make_trade(side="BUY"),
            _make_trade(side="SELL", fill_price=149.0, decision_price=150.0),
        ]
        tca = TransactionCostAnalyzer(trades=trades)
        s = tca.summary()
        assert "buy_mean_is_bps" in s
        assert "sell_mean_is_bps" in s
        assert s["buy_count"] == 1
        assert s["sell_count"] == 1


# ---------------------------------------------------------------------------
# Tests: By-symbol breakdown
# ---------------------------------------------------------------------------


class TestBySymbol:
    def test_by_symbol_empty(self):
        tca = TransactionCostAnalyzer()
        assert tca.by_symbol().empty

    def test_by_symbol_groups(self):
        trades = [
            _make_trade(symbol="AAPL"),
            _make_trade(symbol="AAPL", hour=11),
            _make_trade(symbol="MSFT"),
        ]
        tca = TransactionCostAnalyzer(trades=trades)
        df = tca.by_symbol()
        assert len(df) == 2
        assert df.loc["AAPL"]["n_trades"] == 2
        assert df.loc["MSFT"]["n_trades"] == 1

    def test_by_symbol_sorted_by_volume(self):
        trades = [
            _make_trade(symbol="SMALL", fill_price=10.0, fill_qty=10),
            _make_trade(symbol="BIG", fill_price=500.0, fill_qty=1000),
        ]
        tca = TransactionCostAnalyzer(trades=trades)
        df = tca.by_symbol()
        assert df.index[0] == "BIG"


# ---------------------------------------------------------------------------
# Tests: Capacity analysis
# ---------------------------------------------------------------------------


class TestCapacityAnalysis:
    def test_capacity_empty(self):
        tca = TransactionCostAnalyzer()
        assert tca.capacity_analysis(pd.Series(dtype=float)).empty

    def test_capacity_flags_high_participation(self):
        trades = [
            _make_trade(symbol="AAPL", fill_qty=50_000),
            _make_trade(symbol="MSFT", fill_qty=100),
        ]
        adv = pd.Series({"AAPL": 100_000, "MSFT": 10_000_000})
        tca = TransactionCostAnalyzer(trades=trades)
        flagged = tca.capacity_analysis(adv, threshold_pct=10.0)
        # AAPL: 50000/100000 = 50% → flagged
        # MSFT: 100/10M = 0.001% → not flagged
        assert len(flagged) == 1
        assert flagged.iloc[0]["symbol"] == "AAPL"
        assert flagged.iloc[0]["participation_pct"] == pytest.approx(50.0)

    def test_capacity_threshold(self):
        trades = [_make_trade(symbol="X", fill_qty=1000)]
        adv = pd.Series({"X": 5000})  # 20% participation
        tca = TransactionCostAnalyzer(trades=trades)

        # At 25% threshold: not flagged
        assert tca.capacity_analysis(adv, threshold_pct=25.0).empty
        # At 15% threshold: flagged
        assert len(tca.capacity_analysis(adv, threshold_pct=15.0)) == 1


# ---------------------------------------------------------------------------
# Tests: JSON export
# ---------------------------------------------------------------------------


class TestToJson:
    def test_to_json_empty(self):
        tca = TransactionCostAnalyzer()
        result = tca.to_json()
        assert result["summary"]["n_trades"] == 0
        assert result["by_symbol"] == []
        assert result["recent_trades"] == []

    def test_to_json_structure(self):
        tca = TransactionCostAnalyzer(trades=[_make_trade()])
        result = tca.to_json()
        assert "summary" in result
        assert "by_symbol" in result
        assert "recent_trades" in result
        assert len(result["recent_trades"]) == 1

    def test_to_json_serializable(self):
        """Output should be JSON-serializable (no numpy/pandas types)."""
        import json

        tca = TransactionCostAnalyzer(trades=[_make_trade()])
        result = tca.to_json()
        # Should not raise
        json.dumps(result, default=str)
