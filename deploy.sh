#!/bin/bash
# Deployment script for TradeBot

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

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
  print_error "Docker is not installed. Please install Docker first."
  exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
  print_error "Docker Compose is not installed. Please install Docker Compose first."
  exit 1
fi

# Create necessary directories
print_message "Creating necessary directories..."
mkdir -p data/db data/logs data/models data/backtest backups

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

# Function to deploy the application
deploy() {
  print_message "Building and deploying TradeBot..."
  docker-compose build
  docker-compose up -d
  print_message "TradeBot deployed successfully!"
  print_message "You can check the logs with: docker-compose logs -f"
}

# Function to stop the application
stop() {
  print_message "Stopping TradeBot..."
  docker-compose down
  print_message "TradeBot stopped."
}

# Function to restart the application
restart() {
  print_message "Restarting TradeBot..."
  docker-compose restart
  print_message "TradeBot restarted."
}

# Function to show logs
logs() {
  docker-compose logs -f
}

# Function to backup the database
backup() {
  print_message "Creating database backup..."
  BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
  docker-compose exec -T tradebot sqlite3 /data/db/tradebot.db ".backup '/tmp/tradebot_${BACKUP_DATE}.db'"
  docker-compose exec -T tradebot tar -czf "/backups/tradebot_${BACKUP_DATE}.tar.gz" -C /tmp "tradebot_${BACKUP_DATE}.db"
  docker-compose exec -T tradebot rm "/tmp/tradebot_${BACKUP_DATE}.db"
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
  
  # Stop the application
  docker-compose down
  
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
  
  # Start the application
  docker-compose up -d
  
  print_message "Database restored successfully!"
}

# Function to run backtests
backtest() {
  print_message "Running backtest..."
  docker-compose run --rm tradebot python backtesting.py
  print_message "Backtest completed. Check backtest_results.png for results."
}

# Function to run tests
test() {
  print_message "Running tests..."
  docker-compose run --rm tradebot python -m pytest integration_tests.py -v
  print_message "Tests completed."
}

# Function to show help
show_help() {
  echo "Usage: $0 [command]"
  echo ""
  echo "Commands:"
  echo "  deploy    Build and deploy the application"
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
