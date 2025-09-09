#!/bin/bash
# Run script for TradeBot - Local execution with virtual environment

# Change to the project directory
cd "$(dirname "$0")"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Check if required packages are installed
if ! python -c "import alpaca_trade_api" &> /dev/null; then
    echo "Installing required packages..."
    pip install -r requirements_fixed.txt
fi

# Run the trading bot
echo "Starting TradeBot..."
python run_tradebot.py start "$@"

# Keep the script running
echo "TradeBot is running. Press Ctrl+C to stop."
wait
