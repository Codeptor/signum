#!/bin/bash
# Signum Trading Bot - VPS Setup Script
# Run this on your VPS as root or with sudo

set -e

echo "=========================================="
echo "Signum Trading Bot - VPS Setup"
echo "=========================================="

# Configuration
BOT_USER="signum"
BOT_GROUP="signum"
INSTALL_DIR="/opt/signum"
LOG_DIR="/var/log/signum"
SERVICE_NAME="signum-bot"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (use sudo)"
    exit 1
fi

echo ""
echo "Step 1: Installing dependencies..."
apt-get update
apt-get install -y python3 python3-pip python3-venv git logrotate

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo ""
echo "Step 2: Creating user and directories..."
# Create user if doesn't exist
if ! id "$BOT_USER" &>/dev/null; then
    useradd -r -s /bin/false -m -d "$INSTALL_DIR" "$BOT_USER"
fi

# Create directories
mkdir -p "$INSTALL_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$INSTALL_DIR/data"
mkdir -p "$INSTALL_DIR/logs"
chown -R "$BOT_USER:$BOT_GROUP" "$INSTALL_DIR"
chown -R "$BOT_USER:$BOT_GROUP" "$LOG_DIR"

echo ""
echo "Step 3: Setting up Python environment..."
cd "$INSTALL_DIR"

# Copy source code (you should have cloned/uploaded the repo first)
if [ ! -f "$INSTALL_DIR/pyproject.toml" ]; then
    echo "ERROR: Source code not found in $INSTALL_DIR"
    echo "Please clone the repository first:"
    echo "  git clone <your-repo> $INSTALL_DIR"
    exit 1
fi

# Create virtual environment and install dependencies
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[broker]"

echo ""
echo "Step 4: Setting up configuration..."
# Create .env file if it doesn't exist
if [ ! -f "$INSTALL_DIR/.env" ]; then
    cp "$INSTALL_DIR/.env.example" "$INSTALL_DIR/.env"
    echo ""
    echo "WARNING: Please edit $INSTALL_DIR/.env and add your Alpaca API credentials!"
    echo "  nano $INSTALL_DIR/.env"
fi

# Set proper permissions
chown "$BOT_USER:$BOT_GROUP" "$INSTALL_DIR/.env"
chmod 600 "$INSTALL_DIR/.env"

echo ""
echo "Step 5: Installing systemd service..."
cp "$INSTALL_DIR/deploy/signum-bot.service" /etc/systemd/system/
systemctl daemon-reload

echo ""
echo "Step 6: Setting up logrotate..."
cat > /etc/logrotate.d/signum << EOF
$LOG_DIR/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0644 $BOT_USER $BOT_GROUP
    sharedscripts
    postrotate
        systemctl reload $SERVICE_NAME > /dev/null 2>&1 || true
    endscript
}
EOF

echo ""
echo "Step 7: Creating state directory..."
mkdir -p "$INSTALL_DIR/data/bot_state"
chown -R "$BOT_USER:$BOT_GROUP" "$INSTALL_DIR/data"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit configuration:"
echo "   nano $INSTALL_DIR/.env"
echo ""
echo "2. Add your Alpaca Paper Trading API credentials:"
echo "   ALPACA_API_KEY=your_key_here"
echo "   ALPACA_API_SECRET=your_secret_here"
echo ""
echo "3. Optional: Add alert webhook:"
echo "   ALERT_WEBHOOK_URL=https://hooks.slack.com/..."
echo ""
echo "4. Review risk parameters in .env:"
echo "   MAX_POSITION_WEIGHT=0.30"
echo "   STOP_LOSS_PCT=0.05"
echo "   MAX_DRAWDOWN_LIMIT=0.15"
echo ""
echo "5. Test the bot manually:"
echo "   cd $INSTALL_DIR"
echo "   sudo -u $BOT_USER ./.venv/bin/python examples/live_bot.py"
echo ""
echo "6. Start the service:"
echo "   systemctl enable $SERVICE_NAME"
echo "   systemctl start $SERVICE_NAME"
echo ""
echo "7. Monitor logs:"
echo "   journalctl -u $SERVICE_NAME -f"
echo "   tail -f $LOG_DIR/signum.log"
echo ""
echo "8. Check service status:"
echo "   systemctl status $SERVICE_NAME"
echo ""
