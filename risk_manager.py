import os
import sqlite3
import logging
import numpy as np
import pandas as pd
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tradebot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('RiskManager')

# Load environment variables
load_dotenv()

class RiskManager:
    def __init__(self):
        """Initialize the Risk Manager with API keys and settings"""
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
        
        # Load settings from database
        self.load_settings()
        
        # Initialize risk state
        self.risk_state = {
            'circuit_breaker_triggered': False,
            'circuit_breaker_until': None,
            'consecutive_losses': 0,
            'daily_loss': 0.0,
            'max_drawdown': 0.0,
            'volatility_multiplier': 1.0,
            'last_check': datetime.now()
        }
        
        # Load risk state from database
        self.load_risk_state()
        
        logger.info("RiskManager initialized")
    
    def load_settings(self):
        """Load settings from the database"""
        try:
            conn = sqlite3.connect('tradebot.db')
            cursor = conn.cursor()
            
            # Get all settings
            cursor.execute('SELECT name, value FROM settings')
            settings = dict(cursor.fetchall())
            
            # Assign settings to instance variables
            self.max_loss_per_trade = float(settings.get('max_loss_per_trade', 2.0))
            self.max_daily_loss = float(settings.get('max_daily_loss', 3.0))
            self.max_drawdown = float(settings.get('max_drawdown', 5.0))
            self.stop_loss = float(settings.get('stop_loss', 3.0))
            self.trailing_stop_pct = float(settings.get('trailing_stop_pct', 2.0))
            self.circuit_breaker_threshold = float(settings.get('circuit_breaker_threshold', 5.0))
            self.circuit_breaker_duration = int(settings.get('circuit_breaker_duration', 24))
            self.position_sizing_method = settings.get('position_sizing_method', 'fixed')
            
            conn.close()
            logger.info("Settings loaded from database")
            
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
            # Use default values if database access fails
            self.max_loss_per_trade = 2.0
            self.max_daily_loss = 3.0
            self.max_drawdown = 5.0
            self.stop_loss = 3.0
            self.trailing_stop_pct = 2.0
            self.circuit_breaker_threshold = 5.0
            self.circuit_breaker_duration = 24
            self.position_sizing_method = 'fixed'
    
    def load_risk_state(self):
        """Load risk state from the database"""
        try:
            conn = sqlite3.connect('tradebot.db')
            cursor = conn.cursor()
            
            # Check if risk_state table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='risk_state'")
            if not cursor.fetchone():
                # Create risk_state table if it doesn't exist
                cursor.execute('''
                CREATE TABLE risk_state (
                    id INTEGER PRIMARY KEY,
                    name TEXT UNIQUE,
                    value TEXT,
                    updated_at TIMESTAMP
                )
                ''')
                conn.commit()
            
            # Get risk state
            cursor.execute('SELECT name, value FROM risk_state')
            state = dict(cursor.fetchall())
            
            if state:
                self.risk_state['circuit_breaker_triggered'] = state.get('circuit_breaker_triggered', 'False') == 'True'
                
                circuit_breaker_until = state.get('circuit_breaker_until')
                if circuit_breaker_until:
                    self.risk_state['circuit_breaker_until'] = datetime.fromisoformat(circuit_breaker_until)
                
                self.risk_state['consecutive_losses'] = int(state.get('consecutive_losses', '0'))
                self.risk_state['daily_loss'] = float(state.get('daily_loss', '0.0'))
                self.risk_state['max_drawdown'] = float(state.get('max_drawdown', '0.0'))
                self.risk_state['volatility_multiplier'] = float(state.get('volatility_multiplier', '1.0'))
                
                last_check = state.get('last_check')
                if last_check:
                    self.risk_state['last_check'] = datetime.fromisoformat(last_check)
            
            conn.close()
            logger.info("Risk state loaded from database")
            
        except Exception as e:
            logger.error(f"Error loading risk state: {e}")
    
    def save_risk_state(self):
        """Save risk state to the database"""
        try:
            conn = sqlite3.connect('tradebot.db')
            cursor = conn.cursor()
            
            # Convert datetime objects to strings
            state_to_save = self.risk_state.copy()
            if state_to_save['circuit_breaker_until']:
                state_to_save['circuit_breaker_until'] = state_to_save['circuit_breaker_until'].isoformat()
            state_to_save['last_check'] = state_to_save['last_check'].isoformat()
            
            # Save risk state
            for name, value in state_to_save.items():
                cursor.execute('''
                INSERT OR REPLACE INTO risk_state (name, value, updated_at)
                VALUES (?, ?, ?)
                ''', (name, str(value), datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            
            logger.info("Risk state saved to database")
            
        except Exception as e:
            logger.error(f"Error saving risk state: {e}")
    
    def reset_daily_metrics(self):
        """Reset daily risk metrics"""
        current_time = datetime.now()
        last_check = self.risk_state['last_check']
        
        # Check if we've crossed to a new day
        if current_time.date() > last_check.date():
            logger.info("Resetting daily risk metrics")
            self.risk_state['daily_loss'] = 0.0
            self.risk_state['last_check'] = current_time
            self.save_risk_state()
    
    def calculate_volatility_based_stop(self, symbol, price, lookback_days=20):
        """Calculate volatility-based stop loss using ATR
        
        Args:
            symbol: Stock symbol
            price: Current price
            lookback_days: Number of days to look back for ATR calculation
            
        Returns:
            float: Stop loss price
        """
        try:
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days + 10)  # Add buffer for weekends/holidays
            
            # Format dates for Alpaca API
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Get daily bars
            bars = self.api.get_barset(symbol, 'day', start=start_str, end=end_str, limit=lookback_days)
            
            if symbol not in bars or len(bars[symbol]) < lookback_days:
                logger.warning(f"Not enough data for {symbol} to calculate ATR")
                return price * (1 - self.stop_loss / 100)
            
            # Calculate ATR
            symbol_bars = bars[symbol]
            high_prices = np.array([bar.h for bar in symbol_bars])
            low_prices = np.array([bar.l for bar in symbol_bars])
            close_prices = np.array([bar.c for bar in symbol_bars])
            
            # True Range calculation
            tr1 = high_prices[1:] - low_prices[1:]
            tr2 = np.abs(high_prices[1:] - close_prices[:-1])
            tr3 = np.abs(low_prices[1:] - close_prices[:-1])
            
            true_ranges = np.maximum(np.maximum(tr1, tr2), tr3)
            atr = np.mean(true_ranges)
            
            # Calculate stop loss (2 ATRs below current price)
            volatility_stop = price - (2 * atr * self.risk_state['volatility_multiplier'])
            
            # Ensure stop loss is not too far from price (max 10%)
            min_stop = price * 0.9
            volatility_stop = max(volatility_stop, min_stop)
            
            logger.info(f"Calculated volatility-based stop for {symbol}: ${volatility_stop:.2f} (ATR: ${atr:.2f})")
            
            return volatility_stop
            
        except Exception as e:
            logger.error(f"Error calculating volatility-based stop: {e}")
            return price * (1 - self.stop_loss / 100)
    
    def calculate_trailing_stop(self, position):
        """Calculate trailing stop loss price
        
        Args:
            position: Position dictionary with buy_price, current_price, etc.
            
        Returns:
            float: Trailing stop loss price
        """
        try:
            buy_price = position['buy_price']
            current_price = position['current_price']
            highest_price = position.get('highest_price', current_price)
            
            # Update highest price if current price is higher
            if current_price > highest_price:
                highest_price = current_price
            
            # Calculate trailing stop (trailing_stop_pct% below highest price)
            trailing_stop = highest_price * (1 - self.trailing_stop_pct / 100)
            
            # Ensure trailing stop is above buy price if we're in profit
            if current_price > buy_price:
                trailing_stop = max(trailing_stop, buy_price)
            
            logger.info(f"Calculated trailing stop for {position['symbol']}: ${trailing_stop:.2f} (highest: ${highest_price:.2f})")
            
            return trailing_stop, highest_price
            
        except Exception as e:
            logger.error(f"Error calculating trailing stop: {e}")
            return position['buy_price'] * (1 - self.stop_loss / 100), position['current_price']
    
    def update_position_stops(self, position):
        """Update stop loss prices for a position
        
        Args:
            position: Position dictionary
            
        Returns:
            dict: Updated position with stop loss prices
        """
        symbol = position['symbol']
        current_price = position['current_price']
        
        # Calculate different types of stops
        fixed_stop = position['buy_price'] * (1 - self.stop_loss / 100)
        volatility_stop = self.calculate_volatility_based_stop(symbol, current_price)
        trailing_stop, highest_price = self.calculate_trailing_stop(position)
        
        # Use the highest (most conservative) stop
        stop_loss_price = max(fixed_stop, volatility_stop, trailing_stop)
        
        # Update position
        updated_position = position.copy()
        updated_position['stop_loss_price'] = stop_loss_price
        updated_position['highest_price'] = highest_price
        
        # Save to database
        self.save_position_stop(position['id'], stop_loss_price, highest_price)
        
        return updated_position
    
    def save_position_stop(self, position_id, stop_loss_price, highest_price):
        """Save stop loss price to database"""
        try:
            conn = sqlite3.connect('tradebot.db')
            cursor = conn.cursor()
            
            # Check if positions table has the necessary columns
            cursor.execute("PRAGMA table_info(positions)")
            columns = [col[1] for col in cursor.fetchall()]
            
            if 'stop_loss_price' not in columns:
                cursor.execute("ALTER TABLE positions ADD COLUMN stop_loss_price REAL")
            
            if 'highest_price' not in columns:
                cursor.execute("ALTER TABLE positions ADD COLUMN highest_price REAL")
            
            # Update position
            cursor.execute('''
            UPDATE positions
            SET stop_loss_price = ?, highest_price = ?
            WHERE id = ?
            ''', (stop_loss_price, highest_price, position_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving position stop: {e}")
    
    def check_stop_loss(self, position):
        """Check if a position has hit its stop loss
        
        Args:
            position: Position dictionary
            
        Returns:
            tuple: (hit_stop, stop_type)
        """
        current_price = position['current_price']
        stop_loss_price = position.get('stop_loss_price')
        
        if not stop_loss_price:
            # Calculate stop loss if not already set
            updated_position = self.update_position_stops(position)
            stop_loss_price = updated_position['stop_loss_price']
        
        # Check if price is below stop loss
        if current_price <= stop_loss_price:
            logger.warning(f"{position['symbol']} hit stop loss: ${current_price:.2f} <= ${stop_loss_price:.2f}")
            return True, "stop_loss"
        
        return False, None
    
    def record_trade_result(self, trade):
        """Record the result of a trade for risk management
        
        Args:
            trade: Trade dictionary with symbol, profit_loss, etc.
        """
        profit_loss = trade.get('profit_loss_pct', 0)
        
        # Update consecutive losses
        if profit_loss < 0:
            self.risk_state['consecutive_losses'] += 1
            self.risk_state['daily_loss'] += abs(profit_loss)
            
            # Check if we need to adjust volatility multiplier
            if self.risk_state['consecutive_losses'] >= 2:
                # Reduce position size for next trades
                self.risk_state['volatility_multiplier'] = min(2.0, self.risk_state['volatility_multiplier'] + 0.2)
                logger.info(f"Increased volatility multiplier to {self.risk_state['volatility_multiplier']}")
        else:
            self.risk_state['consecutive_losses'] = 0
            
            # Reset volatility multiplier if we have a win
            if self.risk_state['volatility_multiplier'] > 1.0:
                self.risk_state['volatility_multiplier'] = max(1.0, self.risk_state['volatility_multiplier'] - 0.1)
                logger.info(f"Decreased volatility multiplier to {self.risk_state['volatility_multiplier']}")
        
        # Check for circuit breaker conditions
        if self.risk_state['daily_loss'] >= self.max_daily_loss:
            self.trigger_circuit_breaker("Max daily loss exceeded")
        
        if self.risk_state['consecutive_losses'] >= 3:
            self.trigger_circuit_breaker("Three consecutive losses")
        
        # Save updated risk state
        self.save_risk_state()
    
    def trigger_circuit_breaker(self, reason):
        """Trigger circuit breaker to pause trading
        
        Args:
            reason: Reason for triggering circuit breaker
        """
        if not self.risk_state['circuit_breaker_triggered']:
            self.risk_state['circuit_breaker_triggered'] = True
            self.risk_state['circuit_breaker_until'] = datetime.now() + timedelta(hours=self.circuit_breaker_duration)
            
            logger.warning(f"Circuit breaker triggered: {reason}. Trading paused until {self.risk_state['circuit_breaker_until']}")
            
            # Save risk state
            self.save_risk_state()
    
    def check_circuit_breaker(self):
        """Check if circuit breaker is active
        
        Returns:
            bool: True if circuit breaker is active, False otherwise
        """
        if not self.risk_state['circuit_breaker_triggered']:
            return False
        
        current_time = datetime.now()
        circuit_breaker_until = self.risk_state['circuit_breaker_until']
        
        if current_time >= circuit_breaker_until:
            # Reset circuit breaker
            self.risk_state['circuit_breaker_triggered'] = False
            self.risk_state['circuit_breaker_until'] = None
            logger.info("Circuit breaker reset")
            self.save_risk_state()
            return False
        
        return True
    
    def calculate_position_size(self, account_value, price, volatility=None):
        """Calculate position size based on risk parameters
        
        Args:
            account_value: Total account value
            price: Current price of the stock
            volatility: Optional volatility measure (e.g., ATR)
            
        Returns:
            float: Dollar amount to invest
        """
        # Reset daily metrics if needed
        self.reset_daily_metrics()
        
        # Check if circuit breaker is active
        if self.check_circuit_breaker():
            logger.warning("Circuit breaker active, position size set to 0")
            return 0
        
        # Base position size (fixed percentage of account)
        base_position_size = account_value * 0.8 / 3  # 80% of account divided by max positions (3)
        
        # Apply volatility multiplier
        adjusted_position_size = base_position_size / self.risk_state['volatility_multiplier']
        
        # Cap at max position size
        max_position_size = min(self.max_position_size, account_value * 0.4)  # Max 40% of account in one position
        position_size = min(adjusted_position_size, max_position_size)
        
        logger.info(f"Calculated position size: ${position_size:.2f} (base: ${base_position_size:.2f}, volatility multiplier: {self.risk_state['volatility_multiplier']})")
        
        return position_size
    
    def check_max_drawdown(self, account_value):
        """Check if account has exceeded maximum drawdown
        
        Args:
            account_value: Current account value
            
        Returns:
            bool: True if max drawdown exceeded, False otherwise
        """
        try:
            conn = sqlite3.connect('tradebot.db')
            cursor = conn.cursor()
            
            # Get initial account value (first day's starting_balance)
            cursor.execute('SELECT starting_balance FROM performance ORDER BY date ASC LIMIT 1')
            result = cursor.fetchone()
            
            if not result:
                conn.close()
                return False
            
            initial_value = result[0]
            
            # Calculate drawdown
            drawdown = (initial_value - account_value) / initial_value * 100
            
            conn.close()
            
            # Update max drawdown
            if drawdown > self.risk_state['max_drawdown']:
                self.risk_state['max_drawdown'] = drawdown
                self.save_risk_state()
            
            # Check if max drawdown exceeded
            if drawdown >= self.max_drawdown:
                logger.warning(f"Max drawdown exceeded: {drawdown:.2f}% >= {self.max_drawdown:.2f}%")
                self.trigger_circuit_breaker("Max drawdown exceeded")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking max drawdown: {e}")
            return False
    
    def get_risk_metrics(self):
        """Get current risk metrics
        
        Returns:
            dict: Risk metrics
        """
        return {
            'circuit_breaker_active': self.check_circuit_breaker(),
            'circuit_breaker_until': self.risk_state['circuit_breaker_until'],
            'consecutive_losses': self.risk_state['consecutive_losses'],
            'daily_loss': self.risk_state['daily_loss'],
            'max_drawdown': self.risk_state['max_drawdown'],
            'volatility_multiplier': self.risk_state['volatility_multiplier']
        }


# Test the risk manager if run directly
if __name__ == "__main__":
    risk_manager = RiskManager()
    
    # Get risk metrics
    metrics = risk_manager.get_risk_metrics()
    
    print("Risk Metrics:")
    print(f"Circuit Breaker Active: {metrics['circuit_breaker_active']}")
    if metrics['circuit_breaker_until']:
        print(f"Circuit Breaker Until: {metrics['circuit_breaker_until']}")
    print(f"Consecutive Losses: {metrics['consecutive_losses']}")
    print(f"Daily Loss: {metrics['daily_loss']:.2f}%")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"Volatility Multiplier: {metrics['volatility_multiplier']:.2f}x")
    
    # Test position size calculation
    account_value = 10.0
    position_size = risk_manager.calculate_position_size(account_value, 5.0)
    print(f"\nCalculated Position Size for ${account_value} account: ${position_size:.2f}")
