"""Transaction Cost Analysis (TCA) for post-trade execution quality.

Measures how well trades were executed by comparing fill prices against
benchmarks (arrival price, VWAP). Key metrics:

  - **Implementation Shortfall (IS)**: (fill - decision) / decision
  - **VWAP Slippage**: (fill - VWAP) / VWAP in basis points
  - **Fill Rate**: filled_qty / order_qty
  - **Capacity Analysis**: participation rate vs ADV

References:
  - Kissell, 2013 — "The Science of Algorithmic Trading and Portfolio Management"
  - Almgren & Chriss, 2001 — "Optimal Execution of Portfolio Transactions"
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Single trade execution record for TCA analysis."""

    symbol: str
    side: str  # 'BUY' or 'SELL'
    order_qty: float
    fill_qty: float
    fill_price: float
    decision_price: float  # price when signal was generated
    timestamp: datetime
    commission: float = 0.0
    vwap: Optional[float] = None  # VWAP over execution window


class TransactionCostAnalyzer:
    """Post-trade transaction cost analysis.

    Accepts trade records and computes execution quality metrics
    per trade, per symbol, and aggregated across the portfolio.
    """

    def __init__(self, trades: Optional[list[TradeRecord]] = None):
        self._trades: list[TradeRecord] = trades or []

    def add_trade(self, trade: TradeRecord) -> None:
        """Record a new trade for analysis."""
        self._trades.append(trade)

    def add_trades_from_df(self, df: pd.DataFrame) -> None:
        """Bulk-load trades from a DataFrame.

        Expected columns: symbol, side, order_qty, fill_qty, fill_price,
        decision_price, timestamp. Optional: commission, vwap.
        """
        required = {"symbol", "side", "order_qty", "fill_qty", "fill_price",
                     "decision_price", "timestamp"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        for _, row in df.iterrows():
            self._trades.append(TradeRecord(
                symbol=row["symbol"],
                side=row["side"],
                order_qty=row["order_qty"],
                fill_qty=row["fill_qty"],
                fill_price=row["fill_price"],
                decision_price=row["decision_price"],
                timestamp=row["timestamp"],
                commission=row.get("commission", 0.0),
                vwap=row.get("vwap", None),
            ))

    @property
    def n_trades(self) -> int:
        return len(self._trades)

    # ------------------------------------------------------------------
    # Per-trade metrics
    # ------------------------------------------------------------------

    @staticmethod
    def implementation_shortfall(
        fill_price: float, decision_price: float, side: str
    ) -> float:
        """Implementation shortfall in basis points.

        For buys, positive IS means you paid more than the decision price.
        For sells, positive IS means you received less than the decision price.
        """
        if decision_price <= 0:
            return 0.0
        raw = (fill_price - decision_price) / decision_price * 10_000
        return raw if side.upper() == "BUY" else -raw

    @staticmethod
    def vwap_slippage_bps(fill_price: float, vwap: float) -> float:
        """Slippage vs VWAP in basis points."""
        if vwap <= 0:
            return 0.0
        return (fill_price - vwap) / vwap * 10_000

    @staticmethod
    def fill_rate(fill_qty: float, order_qty: float) -> float:
        """Fraction of order that was filled (0.0 to 1.0)."""
        if order_qty <= 0:
            return 0.0
        return min(fill_qty / order_qty, 1.0)

    # ------------------------------------------------------------------
    # Aggregate analysis
    # ------------------------------------------------------------------

    def analyze(self) -> pd.DataFrame:
        """Compute per-trade TCA metrics for all recorded trades.

        Returns DataFrame with one row per trade, columns:
            symbol, side, fill_price, decision_price, fill_qty, order_qty,
            timestamp, commission, is_bps, vwap_slip_bps, fill_rate,
            trade_value, total_cost_bps
        """
        if not self._trades:
            return pd.DataFrame()

        rows = []
        for t in self._trades:
            is_bps = self.implementation_shortfall(
                t.fill_price, t.decision_price, t.side
            )
            vwap_slip = (
                self.vwap_slippage_bps(t.fill_price, t.vwap)
                if t.vwap is not None and t.vwap > 0
                else np.nan
            )
            fr = self.fill_rate(t.fill_qty, t.order_qty)
            trade_value = t.fill_price * t.fill_qty
            commission_bps = (
                t.commission / trade_value * 10_000 if trade_value > 0 else 0.0
            )

            rows.append({
                "symbol": t.symbol,
                "side": t.side,
                "fill_price": t.fill_price,
                "decision_price": t.decision_price,
                "fill_qty": t.fill_qty,
                "order_qty": t.order_qty,
                "timestamp": t.timestamp,
                "commission": t.commission,
                "is_bps": is_bps,
                "vwap_slip_bps": vwap_slip,
                "fill_rate": fr,
                "trade_value": trade_value,
                "commission_bps": commission_bps,
                "total_cost_bps": abs(is_bps) + commission_bps,
            })

        return pd.DataFrame(rows)

    def summary(self) -> dict:
        """Aggregate TCA summary across all trades.

        Returns dict with:
            n_trades, total_volume, mean_is_bps, median_is_bps,
            mean_vwap_slip_bps, mean_fill_rate, mean_commission_bps,
            mean_total_cost_bps, cost_by_side
        """
        df = self.analyze()
        if df.empty:
            return {"n_trades": 0}

        vwap_valid = df["vwap_slip_bps"].dropna()

        result = {
            "n_trades": len(df),
            "total_volume": float(df["trade_value"].sum()),
            "mean_is_bps": float(df["is_bps"].mean()),
            "median_is_bps": float(df["is_bps"].median()),
            "std_is_bps": float(df["is_bps"].std()) if len(df) > 1 else 0.0,
            "mean_vwap_slip_bps": float(vwap_valid.mean()) if len(vwap_valid) > 0 else None,
            "mean_fill_rate": float(df["fill_rate"].mean()),
            "mean_commission_bps": float(df["commission_bps"].mean()),
            "mean_total_cost_bps": float(df["total_cost_bps"].mean()),
        }

        # Cost breakdown by side
        for side in ["BUY", "SELL"]:
            side_df = df[df["side"].str.upper() == side]
            if not side_df.empty:
                result[f"{side.lower()}_mean_is_bps"] = float(side_df["is_bps"].mean())
                result[f"{side.lower()}_count"] = len(side_df)

        return result

    def by_symbol(self) -> pd.DataFrame:
        """Per-symbol TCA breakdown.

        Returns DataFrame indexed by symbol with aggregated metrics.
        """
        df = self.analyze()
        if df.empty:
            return pd.DataFrame()

        grouped = df.groupby("symbol").agg(
            n_trades=("is_bps", "count"),
            total_volume=("trade_value", "sum"),
            mean_is_bps=("is_bps", "mean"),
            median_is_bps=("is_bps", "median"),
            mean_fill_rate=("fill_rate", "mean"),
            mean_total_cost_bps=("total_cost_bps", "mean"),
        )
        return grouped.sort_values("total_volume", ascending=False)

    def capacity_analysis(
        self,
        adv: pd.Series,
        threshold_pct: float = 10.0,
    ) -> pd.DataFrame:
        """Flag trades where participation exceeds ADV threshold.

        Args:
            adv: Series indexed by symbol with average daily volume.
            threshold_pct: Participation rate threshold (default 10%).

        Returns:
            DataFrame of flagged trades with participation_pct column.
        """
        df = self.analyze()
        if df.empty:
            return pd.DataFrame()

        df["adv"] = df["symbol"].map(adv)
        df["participation_pct"] = np.where(
            df["adv"] > 0,
            df["fill_qty"] / df["adv"] * 100,
            np.nan,
        )

        flagged = df[df["participation_pct"] > threshold_pct].copy()
        if not flagged.empty:
            logger.warning(
                f"TCA: {len(flagged)} trades exceed {threshold_pct}% ADV participation"
            )
        return flagged[["symbol", "side", "fill_qty", "adv", "participation_pct",
                         "timestamp"]].sort_values("participation_pct", ascending=False)

    def to_json(self) -> dict:
        """Export full TCA report as JSON-serializable dict (for dashboard API)."""
        summary = self.summary()
        by_sym = self.by_symbol()

        result = {
            "summary": summary,
            "by_symbol": by_sym.reset_index().to_dict(orient="records") if not by_sym.empty else [],
        }

        # Add per-trade detail (last 50 trades for dashboard)
        df = self.analyze()
        if not df.empty:
            recent = df.sort_values("timestamp", ascending=False).head(50)
            recent["timestamp"] = recent["timestamp"].astype(str)
            result["recent_trades"] = recent.to_dict(orient="records")
        else:
            result["recent_trades"] = []

        return result
