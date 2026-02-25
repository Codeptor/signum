"""Tests for broker module."""

import pytest

from python.brokers.base import (
    BrokerAccount,
    BrokerOrder,
    BrokerPosition,
)
from python.brokers.factory import BrokerFactory


class TestBrokerOrder:
    """Test BrokerOrder dataclass."""

    def test_order_creation(self):
        """Test order creation."""
        order = BrokerOrder(
            symbol="AAPL",
            side="buy",
            qty=100.0,
            order_type="market",
        )

        assert order.symbol == "AAPL"
        assert order.side == "buy"
        assert order.qty == 100.0
        assert order.order_type == "market"
        assert order.time_in_force == "day"


class TestBrokerPosition:
    """Test BrokerPosition dataclass."""

    def test_position_creation(self):
        """Test position creation."""
        position = BrokerPosition(
            symbol="AAPL",
            qty=100.0,
            avg_entry_price=150.0,
            market_value=16000.0,
            unrealized_pl=1000.0,
            unrealized_plpc=0.0667,
        )

        assert position.symbol == "AAPL"
        assert position.qty == 100.0
        assert position.avg_entry_price == 150.0


class TestBrokerAccount:
    """Test BrokerAccount dataclass."""

    def test_account_creation(self):
        """Test account creation."""
        account = BrokerAccount(
            account_id="test-123",
            cash=50000.0,
            portfolio_value=100000.0,
            buying_power=100000.0,
            equity=100000.0,
            status="ACTIVE",
        )

        assert account.account_id == "test-123"
        assert account.cash == 50000.0
        assert account.status == "ACTIVE"


class TestBrokerFactory:
    """Test BrokerFactory."""

    def test_list_brokers(self):
        """Test listing available brokers."""
        brokers = BrokerFactory.list_brokers()

        # Should be empty if alpaca-trade-api not installed
        assert isinstance(brokers, list)

    def test_create_unknown_broker(self):
        """Test creating unknown broker raises error."""
        with pytest.raises(ValueError, match="Unknown broker"):
            BrokerFactory.create("unknown_broker")

    def test_register_broker(self):
        """Test registering a broker."""
        from python.brokers.base import BaseBroker

        class MockBroker(BaseBroker):
            def connect(self):
                return True

            def disconnect(self):
                pass

            def is_connected(self):
                return True

            def get_account(self):
                pass

            def submit_order(self, order):
                pass

            def cancel_order(self, order_id):
                return True

            def get_order(self, order_id):
                pass

            def list_orders(self, status="open"):
                return []

            def get_position(self, symbol):
                pass

            def list_positions(self):
                return []

            def get_latest_price(self, symbol):
                return 100.0

            def get_clock(self):
                return {}

        BrokerFactory.register("mock", MockBroker)

        assert "mock" in BrokerFactory.list_brokers()

        broker = BrokerFactory.create("mock", paper_trading=True)
        assert isinstance(broker, MockBroker)
