#!/bin/bash
# Deployment script for TradeBot - Python-only version

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored messages
print_message() {
  echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

print_warning() {
  echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

print_error() {
  echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Check if Python is installed
if ! command -v python &> /dev/null; then
  print_error "Python is not installed. Please install Python first."
  exit 1
fi

# Create necessary directories
print_message "Creating necessary directories..."
mkdir -p data/db data/logs data/models data/backtest backups

# Check if virtual environment exists
if [ ! -d "venv" ]; then
  print_message "Creating virtual environment..."
  python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Check if .env file exists
if [ ! -f .env ]; then
  print_warning ".env file not found. Creating a template .env file."
  cat > .env << EOL
# Alpaca API credentials
ALPACA_API_KEY=your_api_key_here
ALPACA_API_SECRET=your_api_secret_here
ALPACA_API_ENDPOINT=https://paper-api.alpaca.markets

# NewsAPI credentials
NEWSAPI_KEY=your_newsapi_key_here

# Alpha Vantage API credentials
ALPHAVANTAGE_API_KEY=your_alphavantage_key_here

# Reddit API credentials
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here
REDDIT_USER_AGENT=tradebot:v1.0 (by /u/yourusername)

# Anthropic API credentials for Claude
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ANTHROPIC_MODEL=claude-3-opus-20240229

# Trading parameters
MAX_POSITIONS=3
MAX_POSITION_SIZE=0.3
MIN_PRICE=1.0
MAX_PRICE=20.0
STOP_LOSS=5.0
TRAILING_STOP_PCT=3.0

# Risk management parameters
MAX_LOSS_PER_TRADE=2.0
MAX_DAILY_LOSS=5.0
MAX_DRAWDOWN=10.0
CIRCUIT_BREAKER_THRESHOLD=5.0
CIRCUIT_BREAKER_DURATION=24

# Tax optimization parameters
TAX_AWARE_TRADING=True
TAX_LOSS_HARVESTING=True
SHORT_TERM_TAX_RATE=25.0
LONG_TERM_TAX_RATE=15.0
WASH_SALE_WINDOW_DAYS=30
EOL
  print_warning "Please edit the .env file with your API keys and settings before deploying."
  exit 1
fi

# Install dependencies if needed
install_dependencies() {
  print_message "Checking and installing dependencies..."
  pip install -r requirements_fixed.txt
}

# Function to deploy the application
deploy() {
  print_message "Deploying TradeBot..."
  
  # Install dependencies
  install_dependencies
  
  # Start the application
  nohup python run_tradebot.py start > data/logs/tradebot.log 2>&1 &
  echo $! > data/tradebot.pid
  
  print_message "TradeBot deployed successfully!"
  print_message "You can check the logs with: tail -f data/logs/tradebot.log"
}

# Function to stop the application
stop() {
  print_message "Stopping TradeBot..."
  if [ -f data/tradebot.pid ]; then
    PID=$(cat data/tradebot.pid)
    if ps -p $PID > /dev/null; then
      kill $PID
      print_message "TradeBot stopped."
    else
      print_warning "TradeBot is not running."
    fi
    rm data/tradebot.pid
  else
    print_warning "TradeBot is not running or PID file not found."
  fi
}

# Function to restart the application
restart() {
  print_message "Restarting TradeBot..."
  stop
  sleep 2
  deploy
  print_message "TradeBot restarted."
}

# Function to show logs
logs() {
  if [ -f data/logs/tradebot.log ]; then
    tail -f data/logs/tradebot.log
  else
    print_error "Log file not found."
  fi
}

# Function to backup the database
backup() {
  print_message "Creating database backup..."
  BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
  
  # Make sure the database exists
  if [ ! -f data/db/tradebot.db ]; then
    print_error "Database file not found."
    exit 1
  fi
  
  # Create backup directory if it doesn't exist
  mkdir -p backups
  
  # Create a temporary copy of the database
  sqlite3 data/db/tradebot.db ".backup '/tmp/tradebot_${BACKUP_DATE}.db'"
  
  # Compress the backup
  tar -czf "backups/tradebot_${BACKUP_DATE}.tar.gz" -C /tmp "tradebot_${BACKUP_DATE}.db"
  
  # Remove the temporary file
  rm "/tmp/tradebot_${BACKUP_DATE}.db"
  
  print_message "Backup created: backups/tradebot_${BACKUP_DATE}.tar.gz"
}

# Function to restore a database backup
restore() {
  if [ -z "$1" ]; then
    print_error "Please specify a backup file to restore."
    exit 1
  fi
  
  BACKUP_FILE=$1
  
  if [ ! -f "$BACKUP_FILE" ]; then
    print_error "Backup file not found: $BACKUP_FILE"
    exit 1
  fi
  
  print_message "Restoring database from backup: $BACKUP_FILE"
  print_warning "This will overwrite the current database. Press Ctrl+C to cancel or Enter to continue."
  read -r
  
  # Stop the application if it's running
  if [ -f data/tradebot.pid ]; then
    stop
  fi
  
  # Extract the backup
  mkdir -p tmp
  tar -xzf "$BACKUP_FILE" -C tmp
  
  # Get the extracted DB file name
  DB_FILE=$(find tmp -name "*.db" | head -n 1)
  
  if [ -z "$DB_FILE" ]; then
    print_error "No database file found in the backup."
    rm -rf tmp
    exit 1
  fi
  
  # Copy the backup to the data directory
  cp "$DB_FILE" data/db/tradebot.db
  
  # Clean up
  rm -rf tmp
  
  print_message "Database restored successfully!"
}

# Function to run backtests
backtest() {
  print_message "Running backtest..."
  python backtesting.py
  print_message "Backtest completed. Check backtest_results.png for results."
}

# Function to run tests
test() {
  print_message "Running tests..."
  python -m pytest integration_tests.py -v
  print_message "Tests completed."
}

# Function to show help
show_help() {
  echo "Usage: $0 [command]"
  echo ""
  echo "Commands:"
  echo "  deploy    Deploy the application"
  echo "  stop      Stop the application"
  echo "  restart   Restart the application"
  echo "  logs      Show application logs"
  echo "  backup    Create a database backup"
  echo "  restore   Restore a database backup"
  echo "  backtest  Run backtesting"
  echo "  test      Run tests"
  echo "  help      Show this help message"
}

# Main script
case "$1" in
  deploy)
    deploy
    ;;
  stop)
    stop
    ;;
  restart)
    restart
    ;;
  logs)
    logs
    ;;
  backup)
    backup
    ;;
  restore)
    restore "$2"
    ;;
  backtest)
    backtest
    ;;
  test)
    test
    ;;
  help|"")
    show_help
    ;;
  *)
    print_error "Unknown command: $1"
    show_help
    exit 1
    ;;
esac

exit 0
