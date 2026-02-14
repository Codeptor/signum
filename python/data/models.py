"""Database table definitions."""

from sqlalchemy import Column, DateTime, Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class OHLCVBar(Base):
    __tablename__ = "ohlcv_bars"
    __table_args__ = (UniqueConstraint("ticker", "timestamp", name="uq_ticker_timestamp"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
