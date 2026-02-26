"""
Live Automated Trading Bot

This script represents the continuous loop required to run the strategy automatically.
It should be scheduled to run daily (e.g., 15 minutes before market close) or
run continuously in a while loop checking for market hours.
"""

import os
import time
import logging
from datetime import datetime
import pandas as pd

from python.brokers.alpaca_broker import AlpacaBroker
from python.bridge.execution import ExecutionBridge
from python.portfolio.risk_manager import RiskManager, RiskLimits

# Note: In a full setup, you would import your trained ML model and data fetchers
# from python.alpha.predict import get_model_predictions
# from python.data.fetcher import get_latest_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("LiveBot")


def run_trading_cycle(broker: AlpacaBroker, risk_manager: RiskManager):
    """The core daily trading logic."""
    logger.info("Starting daily trading cycle...")

    # 1. Fetch Latest Data
    logger.info("Fetching latest market data...")
    # data = get_latest_data(symbols=["AAPL", "NVDA", "TSLA", "MSFT", "AMZN"])

    # 2. Generate Signals using ML Model
    logger.info("Running ML model to generate alpha signals...")
    # predictions = get_model_predictions(model, data)

    # 3. Portfolio Optimization
    logger.info("Optimizing portfolio weights...")
    # target_weights = optimizer.optimize(predictions, risk_constraints)

    # For demonstration, we use hardcoded target weights representing the ML output
    target_weights = {
        "AAPL": 0.25,
        "NVDA": 0.25,
        "TSLA": 0.20,
        "MSFT": 0.20,
        "AMZN": 0.10,
    }

    # 4. Get Current Prices to calculate order quantities
    prices = {}
    for sym in target_weights.keys():
        try:
            prices[sym] = broker.get_latest_price(sym)
        except Exception as e:
            logger.warning(f"Could not fetch price for {sym}: {e}")
            prices[sym] = 100.0  # Fallback just for this example script

    # 5. Execution
    logger.info("Passing target weights to Execution Bridge...")
    account = broker.get_account()

    bridge = ExecutionBridge(risk_manager=risk_manager, initial_capital=account.equity)

    # Sync current real positions from broker into the bridge
    for pos in broker.list_positions():
        bridge_pos = bridge.get_position(pos.symbol)
        bridge_pos.quantity = pos.qty
        bridge_pos.avg_cost = pos.avg_entry_price

    # Reconcile portfolio (this automatically buys/sells to reach target weights)
    fills = bridge.reconcile_target_weights(
        target_weights=target_weights,
        prices=prices,
        current_date=datetime.now().strftime("%Y-%m-%d"),
    )

    if not fills:
        logger.info("Portfolio is already at target weights. No trades needed.")
    else:
        logger.info(f"Executed {len(fills)} trades successfully.")


def main():
    logger.info("Starting Live Trading Bot Daemon...")

    # Set to False to trade with REAL MONEY
    # WARNING: Only set to False after thorough paper testing!
    PAPER_TRADING = True

    broker = AlpacaBroker(
        paper_trading=PAPER_TRADING,
        api_key=os.getenv("ALPACA_API_KEY"),
        api_secret=os.getenv("ALPACA_API_SECRET"),
    )

    if not broker.connect():
        logger.error("Failed to connect to Broker. Exiting.")
        return

    # Setup strict risk limits for the live bot
    risk_limits = RiskLimits(
        max_position_weight=0.30,  # Never hold more than 30% in one stock
        max_portfolio_var_95=0.06,  # Stop trading if risk is too high
        max_drawdown_limit=0.15,  # Stop trading if we lose 15% from peak
    )
    risk_manager = RiskManager(limits=risk_limits)

    try:
        while True:
            clock = broker.get_clock()
            is_open = clock["is_open"]

            if is_open:
                # If the market is open, run the trading cycle
                # Usually you run this once per day (e.g., at open or right before close)
                run_trading_cycle(broker, risk_manager)

                # Sleep until the next day to prevent infinite trading loops
                logger.info("Trading cycle complete. Sleeping until tomorrow...")
                time.sleep(60 * 60 * 12)  # Sleep for 12 hours
            else:
                # If market is closed, sleep and check again
                logger.info("Market is currently closed. Sleeping for 1 hour...")
                time.sleep(60 * 60)

    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
    except Exception as e:
        logger.error(f"Fatal error in bot loop: {e}")
    finally:
        broker.disconnect()


if __name__ == "__main__":
    main()
