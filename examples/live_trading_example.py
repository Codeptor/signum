"""
Example: Live Trading with Alpaca Paper Trading

This example demonstrates how to:
1. Connect to Alpaca paper trading
2. Execute trades based on strategy signals
3. Monitor positions and P&L

Prerequisites:
1. Sign up for Alpaca account: https://alpaca.markets
2. Get API keys from: https://app.alpaca.markets/paper/dashboard/overview
3. Set environment variables:
   export ALPACA_API_KEY="your_key"
   export ALPACA_API_SECRET="your_secret"
4. Install broker dependencies:
   pip install alpaca-trade-api

Usage:
    python examples/live_trading_example.py
"""

import logging
import os
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Run live trading example."""
    # Check for API keys
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")

    if not api_key or not api_secret:
        print("Please set ALPACA_API_KEY and ALPACA_API_SECRET environment variables")
        print("Get your keys from: https://app.alpaca.markets/paper/dashboard/overview")
        return

    try:
        from python.brokers.alpaca_broker import AlpacaPaperTrading  # noqa: F401
        from python.portfolio.risk_manager import RiskLimits, RiskManager  # noqa: F401
    except ImportError as e:
        print(f"Missing dependencies: {e}")
        print("Install with: pip install alpaca-trade-api")
        return

    # Initialize broker (paper trading)
    logger.info("Initializing Alpaca paper trading...")
    broker = AlpacaPaperTrading(api_key=api_key, api_secret=api_secret)

    # Connect to broker
    if not broker.connect():
        logger.error("Failed to connect to Alpaca")
        return

    try:
        # Get account info
        account = broker.get_account()
        logger.info(f"Account ID: {account.account_id}")
        logger.info(f"Cash: ${account.cash:,.2f}")
        logger.info(f"Portfolio Value: ${account.portfolio_value:,.2f}")
        logger.info(f"Buying Power: ${account.buying_power:,.2f}")

        # Get current positions
        positions = broker.list_positions()
        logger.info(f"Current Positions: {len(positions)}")
        for pos in positions:
            logger.info(
                f"  {pos.symbol}: {pos.qty} shares @ ${pos.avg_entry_price:.2f} "
                f"(P&L: ${pos.unrealized_pl:.2f})"
            )

        # Get market clock
        clock = broker.get_clock()
        logger.info(f"Market Open: {clock['is_open']}")
        if clock["is_open"]:
            logger.info(f"Next Close: {clock['next_close']}")
        else:
            logger.info(f"Next Open: {clock['next_open']}")

        # Example: Get current price for AAPL
        try:
            aapl_price = broker.get_latest_price("AAPL")
            logger.info(f"AAPL Current Price: ${aapl_price:.2f}")
        except Exception as e:
            logger.warning(f"Could not get AAPL price: {e}")

        # Example: Submit a small paper trade (uncomment to test)
        # from python.brokers.base import BrokerOrder
        # order = BrokerOrder(
        #     symbol="AAPL",
        #     side="buy",
        #     qty=1,  # Just 1 share for testing
        #     order_type="market",
        # )
        # order_id = broker.submit_order(order)
        # logger.info(f"Order submitted: {order_id}")

        # Wait a bit and check positions again
        logger.info("Waiting 5 seconds...")
        time.sleep(5)

        positions = broker.list_positions()
        logger.info(f"Updated Positions: {len(positions)}")

    except Exception as e:
        logger.error(f"Error during trading: {e}")

    finally:
        # Always disconnect
        broker.disconnect()
        logger.info("Disconnected from Alpaca")


if __name__ == "__main__":
    main()
