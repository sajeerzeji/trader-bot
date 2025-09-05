# Simple Automated Trading Bot - Technical Documentation

## Overview

Build a simple bot that continuously finds good stocks, buys them, monitors their performance, and sells when profitable. The entire process runs automatically in a loop.

---

## System Flow

```
Start → Find Best Stock → Buy Stock → Wait 30 minutes → Check Price → 
Should Sell? → Yes: Sell & Start Over | No: Wait 30 minutes → Check Price...
```

---

## Core Components

### 1. Stock Scanner
**Purpose**: Find the best stock to buy right now

**How it works**:
- Get a list of popular stocks (S&P 500 or custom watchlist)
- Check each stock's current price and recent performance
- Apply scoring criteria to rank stocks
- Return the highest-scoring stock

**Scoring Criteria Examples**:
- Stock is trending upward (price higher than 5 days ago)
- Trading volume is above average (more people buying/selling)
- Price is not at an all-time high (room for growth)
- Stock has good momentum (RSI between 30-70)

**Data Needed**:
- Current stock price
- Historical prices (last 5-20 days)
- Trading volume
- Basic technical indicators (RSI, moving averages)

### 2. Buy Engine
**Purpose**: Purchase the selected stock

**Process**:
- Check available cash in trading account
- Calculate position size (how much to buy)
- Place buy order through broker API
- Confirm order was filled
- Record purchase details (price, quantity, timestamp)

**Position Sizing Options**:
- Fixed dollar amount (e.g., $5 per trade)
- Percentage of portfolio (e.g., 50% of available cash)
- Fractional shares (buying partial shares)

### 3. Monitor Engine
**Purpose**: Check stock performance every 30 minutes

**Monitoring Process**:
- Get current price of owned stock
- Calculate current profit/loss
- Check if sell conditions are met
- Log performance data

**What to Monitor**:
- Current stock price
- Percentage gain/loss since purchase
- Time held (how long we've owned it)
- Overall market conditions

### 4. Sell Engine
**Purpose**: Decide when to sell and execute the sale

**Sell Conditions** (choose one or combine):
- **Profit Target**: Sell when stock gains X% (e.g., 3%)
- **Stop Loss**: Sell when stock loses X% (e.g., 2%)
- **Time-based**: Sell after holding for X hours (e.g., 24 hours)
- **Technical**: Sell when RSI > 70 (overbought) or trend reverses

**Sell Process**:
- Place sell order through broker API
- Confirm order was filled
- Record sale details and calculate profit/loss
- Update available cash for next purchase

---

## Technical Architecture

### Simple Architecture
```
Timer (30 min) → Main Controller → Stock Scanner → Buy Engine
                       ↓
Database ← Monitor Engine ← Sell Engine ← Portfolio Tracker
```

### Advanced Architecture
```
                                                 ┌─────────────────┐
                                                 │  AI Integration  │
                                                 └────────┬────────┘
                                                          │
                                                 ┌────────▼────────┐
                                                 │  AI Batch Mgr   │
                                                 └────────┬────────┘
                                                          │
Timer (30 min) → Main Controller → Stock Scanner → Buy Engine
                       │                                  ↑
                       ▼                                  │
           Advanced Technical Analysis ─────────────────────
                       │
                       ▼
Database ← Monitor Engine ← Sell Engine ← Portfolio Manager
    ↑           │                            │
    │           ▼                            ▼
    └─── Risk Manager ←───────────── Alternative Data
```

### Data Storage
**Simple Database Tables**:

**Positions Table**:
- symbol (AAPL, MSFT, etc.)
- quantity (how many shares)
- buy_price (price paid per share)
- buy_time (when purchased)
- current_price (latest price)
- profit_loss (current gain/loss)

**Trades Table**:
- symbol
- action (BUY/SELL)
- quantity
- price
- timestamp
- profit_loss (for sell trades)

**Settings Table**:
- max_position_size (how much to spend per stock)
- profit_target (percentage gain to sell)
- stop_loss (percentage loss to sell)
- max_hold_time (hours to hold before selling)

### Main Controller Logic

**Every 30 Minutes**:
1. Check if we currently own any stock
2. If YES: Monitor current stock → Check sell conditions → Sell if needed
3. If NO (or after selling): Find best stock → Buy it
4. Log all activities
5. Wait 30 minutes and repeat

---

## Implementation Requirements

### Required APIs

**Market Data API** (choose one):
- **Alpha Vantage**: Free tier, 5 calls per minute
- **Yahoo Finance**: Unlimited but unofficial
- **IEX Cloud**: 500k free messages per month

**Broker API** (choose one):
- **Alpaca**: Commission-free, supports fractional shares
- **Robinhood**: Commission-free, supports fractional shares
- **Interactive Brokers**: Supports fractional shares with more advanced features

### Programming Language
**Python** - Best choice because:
- Simple to learn and use
- Excellent financial libraries
- Good API integration support
- Lots of trading examples available

### Required Libraries
- **requests**: For API calls
- **pandas**: For data analysis
- **sqlite3**: For simple database
- **schedule**: For timing (every 30 minutes)
- **logging**: For keeping track of activities
- **numpy**: For numerical operations
- **scikit-learn**: For machine learning models
- **matplotlib/seaborn**: For visualization
- **ta**: For technical analysis indicators
- **anthropic**: For Claude AI integration

### Hardware Requirements
- Any computer that can run Python
- Stable internet connection
- Can run 24/7 (or during market hours)

---

## Simple Risk Management

### Basic Safety Rules
1. **Maximum Loss Per Trade**: Never risk more than 20% of account ($2)
2. **Maximum Daily Loss**: Stop trading if lose more than 30% in one day ($3)
3. **Position Size Limit**: Never put more than 80% of account in one stock ($8)
4. **Cash Reserve**: Always keep 20% of account in cash ($2)

### Emergency Stop Conditions
- If account loses more than 10% in total, stop all trading
- If API fails repeatedly, stop trading until fixed
- If unusual market conditions (market drops more than 5% in day)

---

## Configuration Settings

### Trading Parameters
```
MAX_POSITION_SIZE = $8          # Maximum to spend per stock (80% of $10)
PROFIT_TARGET = 5%               # Sell when stock gains 5%
STOP_LOSS = 3%                   # Sell when stock loses 3%
MAX_HOLD_TIME = 24 hours         # Sell after 24 hours regardless
CHECK_INTERVAL = 30 minutes      # How often to check prices
```

### Stock Selection Criteria
```
MIN_PRICE = $1                   # Focus on lower-priced stocks
MAX_PRICE = $100                 # Don't buy stocks over $100
MIN_VOLUME = 1 million shares    # Only liquid stocks
WATCHLIST = Penny stocks & ETFs   # Pool of stocks to choose from
```

---

## Daily Operation Flow

### Market Open (9:30 AM ET)
1. System starts up
2. Checks if holding any stocks from previous day
3. If holding stock, monitors it
4. If no stock, finds best stock and buys

### During Market Hours (9:30 AM - 4:00 PM ET)
1. Every 30 minutes: check current positions
2. If profit target hit: sell and find new stock
3. If stop loss hit: sell and find new stock
4. If max hold time reached: sell and find new stock
5. Log all activities

### Market Close (4:00 PM ET)
1. System continues to hold positions overnight (or sell everything)
2. Stop looking for new stocks to buy
3. Generate daily performance report

### Overnight/Weekends
- System sleeps (no trading outside market hours)
- Can still monitor positions if desired
- Prepare for next trading day

---

## Success Metrics

### Performance Tracking
- **Win Rate**: Percentage of profitable trades
- **Average Profit**: Average gain per winning trade
- **Average Loss**: Average loss per losing trade
- **Total Return**: Overall portfolio growth
- **Max Drawdown**: Largest losing streak

### Target Performance
- Win rate > 55% (more winning trades than losing)
- Average win > Average loss (wins make more than losses lose)
- Positive monthly returns
- Maximum drawdown < 15%

---

## Getting Started Steps

### Phase 1: Setup (Week 1)
1. Choose broker that supports fractional shares and open paper trading account
2. Get API keys for market data and broker
3. Set up development environment (Python)
4. Create basic database structure

### Phase 2: Build Core Functions (Week 2)
1. Build stock scanner function
2. Build buy function
3. Build monitor function  
4. Build sell function
5. Test each function individually

### Phase 3: Integration (Week 3)
1. Connect all functions together
2. Add main controller loop
3. Add error handling
4. Add logging system
5. Test complete system with paper trading

### Phase 4: Go Live (Week 4)
1. Run system with small amounts of real money
2. Monitor performance closely
3. Adjust parameters based on results
4. Scale up gradually if successful

---

## Common Issues and Solutions

### Problem: System keeps buying the same stock
**Solution**: Add cooldown period - don't buy same stock again for X hours

### Problem: System loses money on most trades
**Solution**: Adjust profit target and stop loss ratios, or improve stock selection criteria

### Problem: API limits exceeded
**Solution**: Add delay between API calls, or upgrade to paid API plan

### Problem: System doesn't find any good stocks
**Solution**: Relax selection criteria or expand watchlist

### Problem: Orders not executing
**Solution**: Use market orders instead of limit orders, or check account permissions

---

## Advanced Components

### AI Batch Manager
**Purpose**: Optimize API calls to Claude AI for cost efficiency and performance

**Features**:
- **Batch Processing**: Groups multiple AI queries into single batches
- **Response Caching**: Stores responses to avoid duplicate API calls
- **Priority Queuing**: Processes urgent queries first
- **Rate Limiting**: Respects API rate limits to prevent errors
- **Asynchronous Processing**: Non-blocking operation for better performance

**Implementation**:
- Runs in background thread to collect and process queries
- Configurable batch size and timing parameters
- Provides both synchronous and asynchronous interfaces
- Stores analysis results in database for future reference

### AI Integration
**Purpose**: Leverage Claude AI for enhanced trading decisions

**Use Cases**:
- Stock purchase evaluation with technical and sentiment data
- Position sale decisions with comprehensive analysis
- Market condition assessment for overall strategy adjustment
- Batch news sentiment analysis across multiple stocks

## Final Notes

This system now includes advanced functionality beyond the core requirements:
1. ✅ Find best stock to invest in (with AI-enhanced analysis)
2. ✅ Buy automatically (with optimized position sizing)
3. ✅ Check value every 30 minutes (with comprehensive metrics)
4. ✅ Sell when profitable (with sophisticated exit strategies)
5. ✅ Repeat the process (with continuous improvement)

**The system is designed for efficiency** with batch processing of AI queries to minimize API costs while maximizing the benefits of AI-enhanced decision making. The architecture supports both simple and advanced trading strategies with appropriate risk management.