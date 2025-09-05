import os
import sys
import logging
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from dotenv import load_dotenv
import yfinance as yf
import alpaca_trade_api as tradeapi
from tqdm import tqdm
import json
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("backtest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('Backtesting')

# Load environment variables
load_dotenv()

# Import bot components
try:
    from advanced_technical_analysis import AdvancedTechnicalAnalysis
    from portfolio_manager import PortfolioManager
    from risk_manager import RiskManager
    from alternative_data import AlternativeData
    from ml_models import MLModels
except ImportError as e:
    logger.error(f"Error importing components: {e}")
    sys.exit(1)

class BacktestDataSource:
    """Base class for backtest data sources"""
    
    def __init__(self, start_date, end_date):
        """Initialize the data source
        
        Args:
            start_date: Start date for backtest data
            end_date: End date for backtest data
        """
        self.start_date = start_date
        self.end_date = end_date
    
    def get_data(self, symbols, timeframe='1d'):
        """Get historical data for symbols
        
        Args:
            symbols: List of symbols to get data for
            timeframe: Timeframe for data (e.g., '1d', '1h')
            
        Returns:
            dict: Dictionary of DataFrames with historical data
        """
        raise NotImplementedError("Subclasses must implement get_data")
    
    def get_universe(self):
        """Get universe of tradable symbols
        
        Returns:
            list: List of tradable symbols
        """
        raise NotImplementedError("Subclasses must implement get_universe")


class YFinanceDataSource(BacktestDataSource):
    """Yahoo Finance data source for backtesting"""
    
    def __init__(self, start_date, end_date):
        """Initialize the Yahoo Finance data source"""
        super().__init__(start_date, end_date)
        self.cache_dir = 'backtest_data'
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_data(self, symbols, timeframe='1d'):
        """Get historical data from Yahoo Finance
        
        Args:
            symbols: List of symbols to get data for
            timeframe: Timeframe for data (e.g., '1d', '1h')
            
        Returns:
            dict: Dictionary of DataFrames with historical data
        """
        # Convert timeframe to yfinance format
        interval = '1d'
        if timeframe == '1h':
            interval = '1h'
        elif timeframe == '1m':
            interval = '1m'
        
        # Check cache first
        cache_file = os.path.join(self.cache_dir, f"{'-'.join(symbols)}_{interval}_{self.start_date.strftime('%Y%m%d')}_{self.end_date.strftime('%Y%m%d')}.pkl")
        
        if os.path.exists(cache_file):
            logger.info(f"Loading data from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        logger.info(f"Downloading data for {len(symbols)} symbols from Yahoo Finance")
        
        data = {}
        for symbol in tqdm(symbols):
            try:
                # Download data
                df = yf.download(
                    symbol,
                    start=self.start_date,
                    end=self.end_date,
                    interval=interval,
                    progress=False
                )
                
                # Rename columns to match our format
                df.columns = [c.lower() for c in df.columns]
                
                # Add symbol column
                df['symbol'] = symbol
                
                # Store in dictionary
                data[symbol] = df
                
            except Exception as e:
                logger.error(f"Error downloading data for {symbol}: {e}")
        
        # Cache data
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        
        return data
    
    def get_universe(self):
        """Get universe of tradable symbols
        
        Returns:
            list: List of tradable symbols
        """
        # For simplicity, we'll use a predefined list
        # In a real implementation, you might want to get this from a file or database
        return [
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM',
            'JNJ', 'V', 'PG', 'UNH', 'HD', 'BAC', 'MA', 'DIS', 'ADBE', 'CRM',
            'NFLX', 'INTC', 'VZ', 'CSCO', 'PFE', 'ABT', 'KO', 'PEP', 'WMT', 'MRK'
        ]


class AlpacaDataSource(BacktestDataSource):
    """Alpaca data source for backtesting"""
    
    def __init__(self, start_date, end_date):
        """Initialize the Alpaca data source"""
        super().__init__(start_date, end_date)
        
        # Load Alpaca API credentials
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.api_secret = os.getenv('ALPACA_API_SECRET')
        self.api_endpoint = os.getenv('ALPACA_API_ENDPOINT')
        
        if not all([self.api_key, self.api_secret, self.api_endpoint]):
            raise ValueError("Alpaca API credentials not found in .env file")
        
        # Initialize Alpaca API
        self.api = tradeapi.REST(
            self.api_key,
            self.api_secret,
            base_url=self.api_endpoint
        )
        
        self.cache_dir = 'backtest_data'
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_data(self, symbols, timeframe='1d'):
        """Get historical data from Alpaca
        
        Args:
            symbols: List of symbols to get data for
            timeframe: Timeframe for data (e.g., '1d', '1h')
            
        Returns:
            dict: Dictionary of DataFrames with historical data
        """
        # Convert timeframe to Alpaca format
        timeframe_map = {
            '1d': '1Day',
            '1h': '1Hour',
            '15m': '15Min',
            '5m': '5Min',
            '1m': '1Min'
        }
        
        alpaca_timeframe = timeframe_map.get(timeframe, '1Day')
        
        # Check cache first
        cache_file = os.path.join(self.cache_dir, f"alpaca_{'-'.join(symbols)}_{alpaca_timeframe}_{self.start_date.strftime('%Y%m%d')}_{self.end_date.strftime('%Y%m%d')}.pkl")
        
        if os.path.exists(cache_file):
            logger.info(f"Loading data from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        logger.info(f"Downloading data for {len(symbols)} symbols from Alpaca")
        
        # Format dates for Alpaca API
        start_str = self.start_date.strftime('%Y-%m-%d')
        end_str = self.end_date.strftime('%Y-%m-%d')
        
        data = {}
        
        try:
            # Get bars for all symbols
            bars = self.api.get_barset(symbols, alpaca_timeframe, start=start_str, end=end_str)
            
            # Process each symbol
            for symbol in symbols:
                if symbol not in bars:
                    logger.warning(f"No data for {symbol}")
                    continue
                
                symbol_bars = bars[symbol]
                
                # Convert to DataFrame
                df = pd.DataFrame()
                df['timestamp'] = [bar.t for bar in symbol_bars]
                df['open'] = [bar.o for bar in symbol_bars]
                df['high'] = [bar.h for bar in symbol_bars]
                df['low'] = [bar.l for bar in symbol_bars]
                df['close'] = [bar.c for bar in symbol_bars]
                df['volume'] = [bar.v for bar in symbol_bars]
                
                # Set timestamp as index
                df.set_index('timestamp', inplace=True)
                
                # Add symbol column
                df['symbol'] = symbol
                
                # Store in dictionary
                data[symbol] = df
            
            # Cache data
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            return data
            
        except Exception as e:
            logger.error(f"Error downloading data from Alpaca: {e}")
            return {}
    
    def get_universe(self):
        """Get universe of tradable symbols from Alpaca
        
        Returns:
            list: List of tradable symbols
        """
        try:
            assets = self.api.list_assets(status='active')
            return [asset.symbol for asset in assets if asset.tradable]
        except Exception as e:
            logger.error(f"Error getting universe from Alpaca: {e}")
            return []


class BacktestBroker:
    """Simulated broker for backtesting"""
    
    def __init__(self, initial_cash=10000.0, commission=0.0):
        """Initialize the backtest broker
        
        Args:
            initial_cash: Initial cash balance
            commission: Commission per trade (percentage)
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission = commission
        self.positions = {}  # symbol -> {'quantity': float, 'cost_basis': float}
        self.orders = []
        self.trades = []
        
        # Performance tracking
        self.equity_curve = []
        self.daily_returns = []
        self.current_date = None
    
    def reset(self):
        """Reset the broker to initial state"""
        self.cash = self.initial_cash
        self.positions = {}
        self.orders = []
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
        self.current_date = None
    
    def update_prices(self, date, prices):
        """Update prices and calculate portfolio value
        
        Args:
            date: Current date
            prices: Dictionary of symbol -> price
        """
        self.current_date = date
        
        # Calculate portfolio value
        portfolio_value = self.cash
        for symbol, position in self.positions.items():
            if symbol in prices:
                position['current_price'] = prices[symbol]
                position['market_value'] = position['quantity'] * prices[symbol]
                portfolio_value += position['market_value']
        
        # Record equity
        self.equity_curve.append({
            'date': date,
            'cash': self.cash,
            'portfolio_value': portfolio_value
        })
        
        # Calculate daily return
        if len(self.equity_curve) > 1:
            prev_value = self.equity_curve[-2]['portfolio_value']
            daily_return = (portfolio_value - prev_value) / prev_value
            self.daily_returns.append({
                'date': date,
                'return': daily_return
            })
    
    def place_order(self, symbol, quantity, price, order_type='market', side='buy'):
        """Place an order
        
        Args:
            symbol: Symbol to trade
            quantity: Order quantity
            price: Current price
            order_type: Order type (market, limit)
            side: Buy or sell
            
        Returns:
            dict: Order information
        """
        # Calculate order value
        order_value = quantity * price
        
        # Calculate commission
        commission_amount = order_value * self.commission
        
        # Check if we have enough cash for buy orders
        if side == 'buy':
            total_cost = order_value + commission_amount
            if total_cost > self.cash:
                # Adjust quantity to available cash
                max_quantity = (self.cash - commission_amount) / price
                quantity = max(0, max_quantity)
                order_value = quantity * price
                commission_amount = order_value * self.commission
                
                if quantity <= 0:
                    return {
                        'success': False,
                        'error': 'Insufficient funds'
                    }
        
        # Create order
        order = {
            'id': len(self.orders) + 1,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'type': order_type,
            'side': side,
            'status': 'filled',
            'commission': commission_amount,
            'date': self.current_date
        }
        
        self.orders.append(order)
        
        # Update positions and cash
        if side == 'buy':
            # Deduct cash
            self.cash -= (order_value + commission_amount)
            
            # Update position
            if symbol in self.positions:
                # Update existing position
                position = self.positions[symbol]
                old_quantity = position['quantity']
                old_cost = position['cost_basis']
                new_quantity = old_quantity + quantity
                new_cost = old_cost + order_value
                
                position['quantity'] = new_quantity
                position['cost_basis'] = new_cost
                position['avg_price'] = new_cost / new_quantity
            else:
                # Create new position
                self.positions[symbol] = {
                    'quantity': quantity,
                    'cost_basis': order_value,
                    'avg_price': price,
                    'current_price': price,
                    'market_value': order_value
                }
        else:  # sell
            # Add cash
            self.cash += (order_value - commission_amount)
            
            # Update position
            if symbol in self.positions:
                position = self.positions[symbol]
                old_quantity = position['quantity']
                old_cost = position['cost_basis']
                
                # Calculate profit/loss
                avg_price = position['avg_price']
                profit_loss = (price - avg_price) * quantity
                
                # Record trade
                trade = {
                    'symbol': symbol,
                    'quantity': quantity,
                    'buy_price': avg_price,
                    'sell_price': price,
                    'profit_loss': profit_loss,
                    'profit_loss_pct': (price - avg_price) / avg_price,
                    'commission': commission_amount,
                    'date': self.current_date
                }
                self.trades.append(trade)
                
                # Update position
                new_quantity = old_quantity - quantity
                if new_quantity <= 0:
                    # Remove position
                    del self.positions[symbol]
                else:
                    # Update quantity
                    position['quantity'] = new_quantity
                    # Cost basis is reduced proportionally
                    position['cost_basis'] = old_cost * (new_quantity / old_quantity)
            else:
                logger.warning(f"Attempted to sell {symbol} but no position exists")
                return {
                    'success': False,
                    'error': 'No position exists'
                }
        
        return {
            'success': True,
            'order': order
        }
    
    def get_positions(self):
        """Get current positions
        
        Returns:
            dict: Current positions
        """
        return self.positions
    
    def get_portfolio_value(self):
        """Get current portfolio value
        
        Returns:
            float: Portfolio value
        """
        if not self.equity_curve:
            return self.initial_cash
        
        return self.equity_curve[-1]['portfolio_value']
    
    def get_performance_stats(self):
        """Calculate performance statistics
        
        Returns:
            dict: Performance statistics
        """
        if not self.equity_curve or len(self.equity_curve) < 2:
            return {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0
            }
        
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('date', inplace=True)
        
        # Calculate returns
        returns_df = pd.DataFrame(self.daily_returns)
        returns_df.set_index('date', inplace=True)
        
        # Calculate statistics
        total_return = (equity_df['portfolio_value'].iloc[-1] / self.initial_cash) - 1
        
        # Annualized return
        days = (equity_df.index[-1] - equity_df.index[0]).days
        years = days / 365
        annualized_return = (1 + total_return) ** (1 / max(years, 0.01)) - 1
        
        # Sharpe ratio (assuming risk-free rate of 0)
        daily_returns_series = returns_df['return']
        sharpe_ratio = daily_returns_series.mean() / daily_returns_series.std() * (252 ** 0.5)  # Annualized
        
        # Maximum drawdown
        cumulative_returns = (1 + daily_returns_series).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        max_drawdown = drawdown.min()
        
        # Win rate
        if self.trades:
            winning_trades = sum(1 for trade in self.trades if trade['profit_loss'] > 0)
            win_rate = winning_trades / len(self.trades)
        else:
            win_rate = 0.0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(self.trades),
            'final_portfolio_value': equity_df['portfolio_value'].iloc[-1]
        }


class BacktestStrategy:
    """Base class for backtest strategies"""
    
    def __init__(self, broker):
        """Initialize the strategy
        
        Args:
            broker: BacktestBroker instance
        """
        self.broker = broker
    
    def initialize(self):
        """Initialize the strategy (called once at the start)"""
        pass
    
    def on_data(self, date, data):
        """Process new data and make trading decisions
        
        Args:
            date: Current date
            data: Dictionary of symbol -> DataFrame with OHLCV data
        """
        raise NotImplementedError("Subclasses must implement on_data")


class TechnicalAnalysisStrategy(BacktestStrategy):
    """Strategy based on technical analysis"""
    
    def __init__(self, broker):
        """Initialize the strategy"""
        super().__init__(broker)
        self.ta = AdvancedTechnicalAnalysis()
        self.lookback = 100  # Days of data needed for analysis
        self.max_positions = 3
        self.position_size = 0.3  # 30% of portfolio per position
    
    def on_data(self, date, data):
        """Process new data and make trading decisions"""
        # Get current prices
        current_prices = {}
        for symbol, df in data.items():
            if date in df.index:
                current_prices[symbol] = df.loc[date, 'close']
        
        # Update broker with current prices
        self.broker.update_prices(date, current_prices)
        
        # Get current positions
        positions = self.broker.get_positions()
        
        # Check if we should sell any positions
        for symbol, position in list(positions.items()):
            if symbol in data and len(data[symbol]) >= self.lookback:
                # Get historical data for analysis
                hist_data = data[symbol].loc[:date].tail(self.lookback)
                
                # Analyze with technical indicators
                analysis = self.ta.analyze_dataframe(hist_data)
                
                # Sell if score is below threshold
                if analysis['score'] < 40:
                    logger.info(f"{date}: Selling {symbol} at {current_prices[symbol]} (Score: {analysis['score']})")
                    self.broker.place_order(
                        symbol=symbol,
                        quantity=position['quantity'],
                        price=current_prices[symbol],
                        side='sell'
                    )
        
        # Check if we should buy any new positions
        if len(positions) < self.max_positions:
            # Calculate available slots
            available_slots = self.max_positions - len(positions)
            
            # Analyze all symbols
            analysis_results = []
            for symbol, df in data.items():
                if symbol not in positions and symbol in current_prices and len(df) >= self.lookback:
                    # Get historical data for analysis
                    hist_data = df.loc[:date].tail(self.lookback)
                    
                    # Analyze with technical indicators
                    analysis = self.ta.analyze_dataframe(hist_data)
                    
                    # Add to results if score is above threshold
                    if analysis['score'] > 60:
                        analysis_results.append({
                            'symbol': symbol,
                            'score': analysis['score'],
                            'price': current_prices[symbol]
                        })
            
            # Sort by score (highest first)
            analysis_results.sort(key=lambda x: x['score'], reverse=True)
            
            # Buy top symbols
            for result in analysis_results[:available_slots]:
                symbol = result['symbol']
                price = result['price']
                
                # Calculate position size
                portfolio_value = self.broker.get_portfolio_value()
                position_value = portfolio_value * self.position_size
                quantity = position_value / price
                
                logger.info(f"{date}: Buying {symbol} at {price} (Score: {result['score']})")
                self.broker.place_order(
                    symbol=symbol,
                    quantity=quantity,
                    price=price,
                    side='buy'
                )


class MLStrategy(BacktestStrategy):
    """Strategy based on machine learning predictions"""
    
    def __init__(self, broker):
        """Initialize the strategy"""
        super().__init__(broker)
        self.ml_models = MLModels()
        self.lookback = 100  # Days of data needed for analysis
        self.max_positions = 3
        self.position_size = 0.3  # 30% of portfolio per position
        
        # Train models flag
        self.models_trained = False
    
    def initialize(self):
        """Initialize the strategy"""
        # Models will be trained on first data
        pass
    
    def on_data(self, date, data):
        """Process new data and make trading decisions"""
        # Get current prices
        current_prices = {}
        for symbol, df in data.items():
            if date in df.index:
                current_prices[symbol] = df.loc[date, 'close']
        
        # Update broker with current prices
        self.broker.update_prices(date, current_prices)
        
        # Train models if needed
        if not self.models_trained:
            for symbol, df in data.items():
                if len(df) >= self.lookback:
                    # Train models on historical data
                    hist_data = df.loc[:date]
                    self._train_models(symbol, hist_data)
            
            self.models_trained = True
        
        # Get current positions
        positions = self.broker.get_positions()
        
        # Check if we should sell any positions
        for symbol, position in list(positions.items()):
            if symbol in data and len(data[symbol]) >= self.lookback:
                # Get historical data for analysis
                hist_data = data[symbol].loc[:date]
                
                # Get ML prediction
                prediction = self._get_prediction(symbol, hist_data)
                
                # Sell if prediction is bearish
                if prediction and prediction['predicted_direction'] == 'down' and prediction['direction_confidence'] > 60:
                    logger.info(f"{date}: Selling {symbol} at {current_prices[symbol]} (ML Prediction: {prediction['predicted_direction']})")
                    self.broker.place_order(
                        symbol=symbol,
                        quantity=position['quantity'],
                        price=current_prices[symbol],
                        side='sell'
                    )
        
        # Check if we should buy any new positions
        if len(positions) < self.max_positions:
            # Calculate available slots
            available_slots = self.max_positions - len(positions)
            
            # Analyze all symbols
            analysis_results = []
            for symbol, df in data.items():
                if symbol not in positions and symbol in current_prices and len(df) >= self.lookback:
                    # Get historical data for analysis
                    hist_data = df.loc[:date]
                    
                    # Get ML prediction
                    prediction = self._get_prediction(symbol, hist_data)
                    
                    # Add to results if prediction is bullish
                    if prediction and prediction['predicted_direction'] == 'up' and prediction['direction_confidence'] > 60:
                        analysis_results.append({
                            'symbol': symbol,
                            'confidence': prediction['direction_confidence'],
                            'price': current_prices[symbol]
                        })
            
            # Sort by confidence (highest first)
            analysis_results.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Buy top symbols
            for result in analysis_results[:available_slots]:
                symbol = result['symbol']
                price = result['price']
                
                # Calculate position size
                portfolio_value = self.broker.get_portfolio_value()
                position_value = portfolio_value * self.position_size
                quantity = position_value / price
                
                logger.info(f"{date}: Buying {symbol} at {price} (ML Confidence: {result['confidence']})")
                self.broker.place_order(
                    symbol=symbol,
                    quantity=quantity,
                    price=price,
                    side='buy'
                )
    
    def _train_models(self, symbol, data):
        """Train ML models on historical data"""
        try:
            # Prepare features
            features_df = self.ml_models.prepare_features(data)
            
            if features_df is not None and len(features_df) > 30:
                # Train models
                self.ml_models.train_price_prediction_model(symbol)
                self.ml_models.train_direction_prediction_model(symbol)
                logger.info(f"Trained ML models for {symbol}")
                return True
        except Exception as e:
            logger.error(f"Error training models for {symbol}: {e}")
        
        return False
    
    def _get_prediction(self, symbol, data):
        """Get ML prediction for a symbol"""
        try:
            # Prepare features
            features_df = self.ml_models.prepare_features(data)
            
            if features_df is not None:
                # Get prediction
                prediction = self.ml_models.predict_next_day(symbol)
                return prediction
        except Exception as e:
            logger.error(f"Error getting prediction for {symbol}: {e}")
        
        return None


class CombinedStrategy(BacktestStrategy):
    """Strategy combining technical analysis, ML, and risk management"""
    
    def __init__(self, broker):
        """Initialize the strategy"""
        super().__init__(broker)
        self.ta = AdvancedTechnicalAnalysis()
        self.ml_models = MLModels()
        self.risk_manager = RiskManager()
        self.portfolio_manager = PortfolioManager()
        
        self.lookback = 100  # Days of data needed for analysis
        self.max_positions = 3
        self.position_size = 0.3  # 30% of portfolio per position
        
        # Strategy weights
        self.ta_weight = 0.5
        self.ml_weight = 0.5
        
        # Models trained flag
        self.models_trained = False
    
    def on_data(self, date, data):
        """Process new data and make trading decisions"""
        # Get current prices
        current_prices = {}
        for symbol, df in data.items():
            if date in df.index:
                current_prices[symbol] = df.loc[date, 'close']
        
        # Update broker with current prices
        self.broker.update_prices(date, current_prices)
        
        # Train models if needed
        if not self.models_trained:
            for symbol, df in data.items():
                if len(df) >= self.lookback:
                    # Train models on historical data
                    hist_data = df.loc[:date]
                    self._train_models(symbol, hist_data)
            
            self.models_trained = True
        
        # Get current positions
        positions = self.broker.get_positions()
        
        # Check if we should sell any positions
        for symbol, position in list(positions.items()):
            if symbol in data and len(data[symbol]) >= self.lookback:
                # Get historical data for analysis
                hist_data = data[symbol].loc[:date].tail(self.lookback)
                
                # Get combined score
                score = self._get_combined_score(symbol, hist_data)
                
                # Check stop loss
                stop_loss_hit = self._check_stop_loss(symbol, position, current_prices[symbol])
                
                # Sell if score is below threshold or stop loss hit
                if score < 40 or stop_loss_hit:
                    reason = "Stop Loss" if stop_loss_hit else f"Low Score ({score})"
                    logger.info(f"{date}: Selling {symbol} at {current_prices[symbol]} ({reason})")
                    self.broker.place_order(
                        symbol=symbol,
                        quantity=position['quantity'],
                        price=current_prices[symbol],
                        side='sell'
                    )
        
        # Check if we should buy any new positions
        if len(positions) < self.max_positions:
            # Check if circuit breaker is active
            if self._check_circuit_breaker():
                logger.info(f"{date}: Circuit breaker active, no new positions")
                return
            
            # Calculate available slots
            available_slots = self.max_positions - len(positions)
            
            # Analyze all symbols
            analysis_results = []
            for symbol, df in data.items():
                if symbol not in positions and symbol in current_prices and len(df) >= self.lookback:
                    # Get historical data for analysis
                    hist_data = df.loc[:date].tail(self.lookback)
                    
                    # Get combined score
                    score = self._get_combined_score(symbol, hist_data)
                    
                    # Add to results if score is above threshold
                    if score > 60:
                        analysis_results.append({
                            'symbol': symbol,
                            'score': score,
                            'price': current_prices[symbol]
                        })
            
            # Sort by score (highest first)
            analysis_results.sort(key=lambda x: x['score'], reverse=True)
            
            # Buy top symbols
            for result in analysis_results[:available_slots]:
                symbol = result['symbol']
                price = result['price']
                
                # Calculate position size with risk management
                portfolio_value = self.broker.get_portfolio_value()
                position_value = self._calculate_position_size(portfolio_value, price, symbol)
                quantity = position_value / price
                
                if quantity > 0:
                    logger.info(f"{date}: Buying {symbol} at {price} (Score: {result['score']})")
                    self.broker.place_order(
                        symbol=symbol,
                        quantity=quantity,
                        price=price,
                        side='buy'
                    )
    
    def _train_models(self, symbol, data):
        """Train ML models on historical data"""
        try:
            # Prepare features
            features_df = self.ml_models.prepare_features(data)
            
            if features_df is not None and len(features_df) > 30:
                # Train models
                self.ml_models.train_price_prediction_model(symbol)
                self.ml_models.train_direction_prediction_model(symbol)
                logger.info(f"Trained ML models for {symbol}")
                return True
        except Exception as e:
            logger.error(f"Error training models for {symbol}: {e}")
        
        return False
    
    def _get_combined_score(self, symbol, data):
        """Get combined score from technical analysis and ML"""
        ta_score = 50
        ml_score = 50
        
        # Get technical analysis score
        try:
            ta_result = self.ta.analyze_dataframe(data)
            ta_score = ta_result['score']
        except Exception as e:
            logger.error(f"Error getting TA score for {symbol}: {e}")
        
        # Get ML score
        try:
            ml_result = self.ml_models.get_ml_score(symbol)
            if ml_result:
                ml_score = ml_result['ml_score']
        except Exception as e:
            logger.error(f"Error getting ML score for {symbol}: {e}")
        
        # Calculate combined score
        combined_score = (ta_score * self.ta_weight) + (ml_score * self.ml_weight)
        
        return combined_score
    
    def _check_stop_loss(self, symbol, position, current_price):
        """Check if stop loss is hit"""
        # Simple stop loss implementation
        if 'avg_price' in position:
            buy_price = position['avg_price']
            stop_loss_pct = 0.05  # 5% stop loss
            stop_loss_price = buy_price * (1 - stop_loss_pct)
            
            return current_price <= stop_loss_price
        
        return False
    
    def _check_circuit_breaker(self):
        """Check if circuit breaker is active"""
        # Simple circuit breaker implementation
        if len(self.broker.daily_returns) >= 3:
            # Check last 3 days
            recent_returns = [r['return'] for r in self.broker.daily_returns[-3:]]
            
            # If all negative and total loss > 5%
            if all(r < 0 for r in recent_returns) and sum(recent_returns) < -0.05:
                return True
        
        return False
    
    def _calculate_position_size(self, portfolio_value, price, symbol):
        """Calculate position size with risk management"""
        # Base position size
        base_size = portfolio_value * self.position_size
        
        # Adjust based on recent performance
        if len(self.broker.daily_returns) >= 5:
            recent_returns = [r['return'] for r in self.broker.daily_returns[-5:]]
            avg_return = sum(recent_returns) / len(recent_returns)
            
            # Reduce size if recent performance is poor
            if avg_return < -0.01:  # -1% average daily return
                base_size *= 0.5
            elif avg_return < 0:
                base_size *= 0.75
        
        return base_size


class Backtester:
    """Main backtesting engine"""
    
    def __init__(self, data_source, strategy_class, initial_cash=10000.0, commission=0.0):
        """Initialize the backtester
        
        Args:
            data_source: Data source for historical data
            strategy_class: Strategy class to use
            initial_cash: Initial cash balance
            commission: Commission per trade (percentage)
        """
        self.data_source = data_source
        self.strategy_class = strategy_class
        self.initial_cash = initial_cash
        self.commission = commission
        
        self.broker = None
        self.strategy = None
        self.results = None
    
    def run(self, symbols=None, start_date=None, end_date=None):
        """Run the backtest
        
        Args:
            symbols: List of symbols to backtest (if None, use universe)
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            dict: Backtest results
        """
        # Use data source dates if not provided
        if start_date is None:
            start_date = self.data_source.start_date
        if end_date is None:
            end_date = self.data_source.end_date
        
        # Get symbols from universe if not provided
        if symbols is None:
            symbols = self.data_source.get_universe()
            # Limit to 30 symbols for performance
            symbols = symbols[:30]
        
        logger.info(f"Running backtest on {len(symbols)} symbols from {start_date} to {end_date}")
        
        # Get historical data
        data = self.data_source.get_data(symbols)
        
        # Initialize broker and strategy
        self.broker = BacktestBroker(initial_cash=self.initial_cash, commission=self.commission)
        self.strategy = self.strategy_class(self.broker)
        
        # Initialize strategy
        self.strategy.initialize()
        
        # Create date range
        all_dates = set()
        for symbol, df in data.items():
            all_dates.update(df.index)
        
        date_range = sorted([d for d in all_dates if start_date <= d <= end_date])
        
        # Run backtest
        logger.info(f"Running backtest over {len(date_range)} trading days")
        
        for date in tqdm(date_range):
            # Process this date
            self.strategy.on_data(date, data)
        
        # Calculate results
        self.results = self._calculate_results()
        
        return self.results
    
    def _calculate_results(self):
        """Calculate backtest results
        
        Returns:
            dict: Backtest results
        """
        # Get performance stats
        stats = self.broker.get_performance_stats()
        
        # Create equity curve DataFrame
        equity_df = pd.DataFrame(self.broker.equity_curve)
        equity_df.set_index('date', inplace=True)
        
        # Create returns DataFrame
        returns_df = pd.DataFrame(self.broker.daily_returns)
        if not returns_df.empty:
            returns_df.set_index('date', inplace=True)
        
        # Create trades DataFrame
        trades_df = pd.DataFrame(self.broker.trades)
        
        return {
            'stats': stats,
            'equity_curve': equity_df,
            'returns': returns_df,
            'trades': trades_df
        }
    
    def plot_results(self):
        """Plot backtest results"""
        if self.results is None:
            logger.error("No backtest results to plot")
            return
        
        # Set up the figure
        plt.figure(figsize=(12, 10))
        
        # Plot equity curve
        plt.subplot(2, 1, 1)
        plt.plot(self.results['equity_curve'].index, self.results['equity_curve']['portfolio_value'])
        plt.title('Portfolio Value')
        plt.xlabel('Date')
        plt.ylabel('Value ($)')
        plt.grid(True)
        
        # Plot returns
        if not self.results['returns'].empty:
            plt.subplot(2, 1, 2)
            plt.plot(self.results['returns'].index, self.results['returns']['return'].cumsum())
            plt.title('Cumulative Returns')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Return')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('backtest_results.png')
        plt.close()
        
        # Plot trade analysis if we have trades
        if not self.results['trades'].empty and len(self.results['trades']) > 0:
            plt.figure(figsize=(12, 10))
            
            # Plot profit/loss distribution
            plt.subplot(2, 1, 1)
            sns.histplot(self.results['trades']['profit_loss_pct'], kde=True)
            plt.title('Profit/Loss Distribution')
            plt.xlabel('Profit/Loss %')
            plt.ylabel('Frequency')
            plt.grid(True)
            
            # Plot profit/loss by symbol
            plt.subplot(2, 1, 2)
            symbol_pnl = self.results['trades'].groupby('symbol')['profit_loss'].sum()
            symbol_pnl.sort_values().plot(kind='bar')
            plt.title('Profit/Loss by Symbol')
            plt.xlabel('Symbol')
            plt.ylabel('Profit/Loss ($)')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('trade_analysis.png')
            plt.close()
        
        logger.info("Plots saved to backtest_results.png and trade_analysis.png")
    
    def print_results(self):
        """Print backtest results"""
        if self.results is None:
            logger.error("No backtest results to print")
            return
        
        stats = self.results['stats']
        
        print("\n===== Backtest Results =====")
        print(f"Initial Capital: ${self.initial_cash:.2f}")
        print(f"Final Portfolio Value: ${stats['final_portfolio_value']:.2f}")
        print(f"Total Return: {stats['total_return']:.2%}")
        print(f"Annualized Return: {stats['annualized_return']:.2%}")
        print(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
        print(f"Maximum Drawdown: {stats['max_drawdown']:.2%}")
        print(f"Win Rate: {stats['win_rate']:.2%}")
        print(f"Total Trades: {stats['total_trades']}")
        
        if not self.results['trades'].empty and len(self.results['trades']) > 0:
            print("\n===== Trade Statistics =====")
            print(f"Average Profit/Loss: {self.results['trades']['profit_loss'].mean():.2f}")
            print(f"Average Profit/Loss %: {self.results['trades']['profit_loss_pct'].mean():.2%}")
            print(f"Best Trade: {self.results['trades']['profit_loss'].max():.2f}")
            print(f"Worst Trade: {self.results['trades']['profit_loss'].min():.2f}")
            
            # Top 5 symbols by profit
            top_symbols = self.results['trades'].groupby('symbol')['profit_loss'].sum().sort_values(ascending=False).head(5)
            print("\nTop 5 Symbols by Profit:")
            for symbol, profit in top_symbols.items():
                print(f"  {symbol}: ${profit:.2f}")


# Example usage
if __name__ == "__main__":
    # Set up dates
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 12, 31)
    
    # Create data source
    data_source = YFinanceDataSource(start_date, end_date)
    
    # Define symbols
    symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']
    
    # Create backtester with technical analysis strategy
    backtester = Backtester(
        data_source=data_source,
        strategy_class=TechnicalAnalysisStrategy,
        initial_cash=10000.0,
        commission=0.001  # 0.1% commission
    )
    
    # Run backtest
    results = backtester.run(symbols=symbols)
    
    # Print and plot results
    backtester.print_results()
    backtester.plot_results()
    
    # Try combined strategy
    print("\nRunning backtest with combined strategy...")
    backtester = Backtester(
        data_source=data_source,
        strategy_class=CombinedStrategy,
        initial_cash=10000.0,
        commission=0.001
    )
    
    # Run backtest
    results = backtester.run(symbols=symbols)
    
    # Print and plot results
    backtester.print_results()
    backtester.plot_results()
