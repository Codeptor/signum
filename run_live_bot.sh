#!/bin/bash
# Live Trading Bot Launcher with Crash Recovery
#
# This script runs the live trading bot with automatic restart on crashes.
# It implements exponential backoff to avoid tight restart loops.
#
# Usage:
#   export ALPACA_API_KEY="your_key"
#   export ALPACA_API_SECRET="your_secret"
#   ./run_live_bot.sh
#
# For production deployment, consider using systemd instead of this script.

set -euo pipefail

# Configuration
MAX_RETRIES=${MAX_RETRIES:-10}
INITIAL_BACKOFF_SECS=${INITIAL_BACKOFF_SECS:-5}
MAX_BACKOFF_SECS=${MAX_BACKOFF_SECS:-300}  # 5 minutes
LOG_DIR=${LOG_DIR:-"logs"}
BOT_SCRIPT="examples/live_bot.py"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Validate environment
if [[ -z "${ALPACA_API_KEY:-}" ]] || [[ "$ALPACA_API_KEY" == "your_api_key" ]]; then
    echo "ERROR: ALPACA_API_KEY must be set to a valid API key (not the placeholder)"
    exit 1
fi

if [[ -z "${ALPACA_API_SECRET:-}" ]] || [[ "$ALPACA_API_SECRET" == "your_secret_key" ]]; then
    echo "ERROR: ALPACA_API_SECRET must be set to a valid API secret (not the placeholder)"
    exit 1
fi

# Export for the Python script
export ALPACA_API_KEY
export ALPACA_API_SECRET

# Function to calculate next backoff with exponential growth and jitter
calculate_backoff() {
    local attempt=$1
    local backoff=$((INITIAL_BACKOFF_SECS * (2 ** (attempt - 1))))
    if (( backoff > MAX_BACKOFF_SECS )); then
        backoff=$MAX_BACKOFF_SECS
    fi
    # Add random jitter (±20%) to prevent thundering herd
    local jitter=$((backoff / 5))
    if (( jitter > 0 )); then
        backoff=$((backoff + (RANDOM % jitter) - (jitter / 2)))
    fi
    echo "$backoff"
}

# Main restart loop
echo "========================================"
echo "Live Trading Bot Launcher"
echo "Max retries: $MAX_RETRIES"
echo "Log directory: $LOG_DIR"
echo "========================================"

attempt=0
while true; do
    attempt=$((attempt + 1))
    timestamp=$(date +"%Y%m%d_%H%M%S")
    log_file="$LOG_DIR/bot_${timestamp}_attempt${attempt}.log"

    echo ""
    echo "[$timestamp] Starting bot (attempt $attempt/$MAX_RETRIES)..."
    echo "Logging to: $log_file"

    # Run the bot and capture exit code
    set +e
    uv run python "$BOT_SCRIPT" 2>&1 | tee "$log_file"
    exit_code=${PIPESTATUS[0]}
    set -e

    # Check exit code
    if [[ $exit_code -eq 0 ]]; then
        echo "[$timestamp] Bot exited cleanly (code 0)."
        echo "This is unexpected - the bot should run indefinitely."
        echo "Waiting before restart..."
    elif [[ $exit_code -eq 130 ]] || [[ $exit_code -eq 2 ]]; then
        # 130 = Ctrl+C (SIGINT), 2 = KeyboardInterrupt
        echo "[$timestamp] Bot stopped by user (Ctrl+C). Exiting."
        exit 0
    else
        echo "[$timestamp] Bot crashed with exit code $exit_code."
    fi

    # Check if we've exceeded max retries
    if [[ $attempt -ge $MAX_RETRIES ]]; then
        echo ""
        echo "========================================"
        echo "ERROR: Maximum retries ($MAX_RETRIES) exceeded."
        echo "Bot is failing repeatedly. Manual intervention required."
        echo "Check logs in: $LOG_DIR"
        echo "========================================"
        exit 1
    fi

    # Calculate and apply backoff
    backoff=$(calculate_backoff "$attempt")
    echo "Waiting ${backoff}s before restart (attempt $((attempt + 1))/$MAX_RETRIES)..."
    sleep "$backoff"
done
