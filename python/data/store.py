"""Data storage layer. Uses TimescaleDB in production, SQLite for testing."""

import logging
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from python.data.models import Base, OHLCVBar

logger = logging.getLogger(__name__)


class DataStore:
    def __init__(self, connection_string: str = "sqlite:///data/quant.db"):
        self.engine = create_engine(connection_string)

    def init_db(self):
        """Create tables if they don't exist."""
        Base.metadata.create_all(self.engine)

    def upsert_ohlcv(self, df: pd.DataFrame) -> int:
        """Insert or update OHLCV bars from a DataFrame.

        Expects columns: ticker, open, high, low, close, volume
        with a DatetimeIndex.
        """
        records = []
        for ts, row in df.iterrows():
            records.append(
                OHLCVBar(
                    ticker=row["ticker"],
                    timestamp=ts,
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row["volume"],
                )
            )

        with Session(self.engine) as session:
            for record in records:
                existing = session.execute(
                    select(OHLCVBar).where(
                        OHLCVBar.ticker == record.ticker,
                        OHLCVBar.timestamp == record.timestamp,
                    )
                ).scalar_one_or_none()
                if existing:
                    existing.open = record.open
                    existing.high = record.high
                    existing.low = record.low
                    existing.close = record.close
                    existing.volume = record.volume
                else:
                    session.add(record)
            session.commit()

        logger.info(f"Upserted {len(records)} OHLCV bars")
        return len(records)

    def get_ohlcv(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Retrieve OHLCV bars for a ticker within a date range."""
        with Session(self.engine) as session:
            stmt = (
                select(OHLCVBar)
                .where(
                    OHLCVBar.ticker == ticker,
                    OHLCVBar.timestamp >= datetime.fromisoformat(start_date),
                    OHLCVBar.timestamp <= datetime.fromisoformat(end_date),
                )
                .order_by(OHLCVBar.timestamp)
            )
            rows = session.execute(stmt).scalars().all()

        return pd.DataFrame(
            [
                {
                    "timestamp": r.timestamp,
                    "ticker": r.ticker,
                    "open": r.open,
                    "high": r.high,
                    "low": r.low,
                    "close": r.close,
                    "volume": r.volume,
                }
                for r in rows
            ]
        )
