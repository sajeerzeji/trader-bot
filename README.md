# TradeBot - Advanced Automated Trading System

TradeBot is a sophisticated automated trading system designed to work with small accounts (as low as $10) using fractional shares. It incorporates advanced technical analysis, machine learning, risk management, and alternative data to make intelligent trading decisions.

## Features

- **Advanced Technical Analysis**: Implements sophisticated technical indicators using TA-Lib
- **Multi-Position Portfolio Management**: Manages multiple concurrent positions with correlation constraints
- **Enhanced Risk Management**: Features stop-loss strategies, circuit breakers, and dynamic position sizing
- **Alternative Data Integration**: Incorporates news sentiment and social media data
- **Machine Learning Models**: Predicts price movements using Random Forest models
- **Infrastructure Improvements**: Includes database connection pooling, caching, and health monitoring
- **Tax Optimization**: Implements tax-aware trading strategies and tax-loss harvesting
- **Multi-Exchange Support**: Supports trading on multiple exchanges (Alpaca, Coinbase)
- **Backtesting Framework**: Tests strategies against historical data
- **Integration Testing**: Ensures all components work together correctly
- **Containerized Deployment**: Easy deployment using Docker and Docker Compose

## Prerequisites

### For Docker Deployment
- Docker and Docker Compose

### For Direct Execution
- Python 3.9 or higher
- pip (Python package manager)

### API Keys Required
- Alpaca (for trading)
- NewsAPI (for news sentiment)
- Alpha Vantage (for market data)
- Reddit (for social media sentiment)
- Anthropic (for Claude AI integration)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/tradebot.git
   cd tradebot
   ```

2. Set up your environment variables by creating a `.env` file:
   ```
   cp .env.example .env
   ```
   
3. Edit the `.env` file with your API keys and trading parameters.

4. Install required Python packages:
   ```
   pip install -r requirements.txt
   ```

5. Choose your deployment method:

   **Option 1: Docker Deployment**
   ```
   ./deploy.sh deploy
   ```

   **Option 2: Direct Execution**
   ```
   ./run_tradebot.py start
   ```

## Usage

### Docker Deployment Commands

The deployment script provides several commands to manage the trading bot:

- **Deploy**: `./deploy.sh deploy` - Build and deploy the application
- **Stop**: `./deploy.sh stop` - Stop the application
- **Restart**: `./deploy.sh restart` - Restart the application
- **Logs**: `./deploy.sh logs` - Show application logs
- **Backup**: `./deploy.sh backup` - Create a database backup
- **Restore**: `./deploy.sh restore <backup_file>` - Restore a database backup
- **Backtest**: `./deploy.sh backtest` - Run backtesting
- **Test**: `./deploy.sh test` - Run tests

### Direct Execution Commands

Alternatively, you can use the direct execution script:

- **Start**: `./run_tradebot.py start` - Start the trading bot
- **Stop**: `./run_tradebot.py stop` - Stop the trading bot
- **Backup**: `./run_tradebot.py backup` - Create a database backup
- **Restore**: `./run_tradebot.py restore --backup-file <backup_file>` - Restore from backup
- **Backtest**: `./run_tradebot.py backtest` - Run backtesting
- **Test**: `./run_tradebot.py test` - Run integration tests

## Components

### Main Components

- **Stock Scanner**: Discovers potential trading opportunities
- **Buy Engine**: Evaluates and executes buy orders
- **Monitor Engine**: Tracks and updates positions
- **Sell Engine**: Evaluates and executes sell orders

### Advanced Components

- **Advanced Technical Analysis**: Analyzes price patterns and indicators
- **Portfolio Manager**: Manages multiple positions and diversification
- **Risk Manager**: Implements risk control measures
- **Alternative Data**: Processes news and social media sentiment
- **ML Models**: Predicts price movements and volatility
- **Infrastructure Manager**: Handles database, caching, and monitoring
- **Tax Optimizer**: Implements tax-efficient trading strategies
- **Multi-Exchange Manager**: Supports trading across different platforms

## Deployment Architecture

The system is deployed using Docker Compose with the following services:

1. **TradeBot**: Main trading application
2. **Scheduler**: Manages scheduled tasks (market hours, after-hours)
3. **Backup**: Handles automated database backups

Data is persisted in volumes:
- `/data/db`: Database files
- `/data/logs`: Log files
- `/data/models`: ML model files
- `/data/backtest`: Backtesting results
- `/backups`: Database backups

## Backtesting

The system includes a comprehensive backtesting framework that allows you to:

1. Test different trading strategies
2. Use historical data from various sources (Yahoo Finance, Alpaca)
3. Analyze performance metrics (returns, Sharpe ratio, drawdown)
4. Visualize results with charts

To run a backtest:

**With Docker:**
```
./deploy.sh backtest
```

**Without Docker:**
```
./run_tradebot.py backtest
```

## Testing

Integration tests ensure all components work together correctly. To run tests:

**With Docker:**
```
./deploy.sh test
```

**Without Docker:**
```
./run_tradebot.py test
```

## Dashboard

TradeBot includes a real-time dashboard to visualize your trading activities, positions, and performance metrics.

### Features

- Summary statistics (total positions, open positions, profit/loss)
- Detailed transaction tables (positions, buy/sell pairs, orders)
- Performance visualizations (profit/loss by symbol, trading activity, asset allocation)
- Auto-refresh every 15 minutes

### Running the Dashboard

1. Generate the dashboard with your latest trading data:

```
python generate_html_dashboard.py
```

2. Open the generated HTML file in your web browser:

```
open dashboard.html
```

The dashboard will automatically refresh every 15 minutes to show the latest data. You can also manually regenerate it at any time by running the command above.

## Maintenance

- Regular database backups are performed automatically
- Database optimization runs daily after market hours
- ML models are retrained weekly
- Health checks run hourly

## Security

- Containers run as non-root users
- API keys are stored in environment variables
- Database is backed up regularly
- Circuit breakers prevent excessive losses

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Alpaca API for brokerage services
- TA-Lib for technical indicators
- scikit-learn for machine learning capabilities
- Anthropic's Claude for AI integration
