"""Broker factory for creating broker instances."""

import logging
from typing import Dict, Type

from python.brokers.base import BaseBroker

logger = logging.getLogger(__name__)


class BrokerFactory:
    """Factory for creating broker instances."""

    _brokers: Dict[str, Type[BaseBroker]] = {}

    @classmethod
    def register(cls, name: str, broker_class: Type[BaseBroker]) -> None:
        """Register a broker class."""
        cls._brokers[name.lower()] = broker_class
        logger.info(f"Registered broker: {name}")

    @classmethod
    def create(
        cls,
        name: str,
        paper_trading: bool = True,
        **kwargs,
    ) -> BaseBroker:
        """
        Create a broker instance.

        Args:
            name: Broker name (alpaca, etc.)
            paper_trading: If True, use paper trading
            **kwargs: Additional broker-specific arguments

        Returns:
            Broker instance
        """
        name = name.lower()

        if name not in cls._brokers:
            raise ValueError(f"Unknown broker: {name}. Available: {list(cls._brokers.keys())}")

        broker_class = cls._brokers[name]
        return broker_class(paper_trading=paper_trading, **kwargs)

    @classmethod
    def list_brokers(cls) -> list:
        """List available broker names."""
        return list(cls._brokers.keys())


# Auto-register available brokers
def _register_brokers():
    """Auto-register broker implementations."""
    try:
        from python.brokers.alpaca_broker import AlpacaBroker

        BrokerFactory.register("alpaca", AlpacaBroker)
    except ImportError:
        logger.warning("alpaca-trade-api not installed. Alpaca broker unavailable.")


# Register on import
_register_brokers()
