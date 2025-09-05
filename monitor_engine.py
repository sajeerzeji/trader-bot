import os
import sqlite3
import logging
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
logger = logging.getLogger('MonitorEngine')

# Load environment variables
load_dotenv()

class MonitorEngine:
    def __init__(self):
        """Initialize the MonitorEngine with API keys and settings"""
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
        
        logger.info("MonitorEngine initialized")
    
    def load_settings(self):
        """Load settings from the database"""
        try:
            conn = sqlite3.connect('tradebot.db')
            cursor = conn.cursor()
            
            # Get all settings
            cursor.execute('SELECT name, value FROM settings')
            settings = dict(cursor.fetchall())
            
            # Assign settings to instance variables
            self.profit_target = settings.get('profit_target', 5.0)
            self.stop_loss = settings.get('stop_loss', 3.0)
            self.max_hold_time = settings.get('max_hold_time', 24.0)
            
            conn.close()
            logger.info("Settings loaded from database")
            
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
            # Use default values if database access fails
            self.profit_target = 5.0
            self.stop_loss = 3.0
            self.max_hold_time = 24.0
    
    def get_active_positions(self):
        """Get all active positions from the database"""
        try:
            conn = sqlite3.connect('tradebot.db')
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT id, symbol, quantity, buy_price, buy_time
            FROM positions
            WHERE status = 'active'
            ''')
            
            positions = []
            for row in cursor.fetchall():
                positions.append({
                    'id': row[0],
                    'symbol': row[1],
                    'quantity': row[2],
                    'buy_price': row[3],
                    'buy_time': datetime.fromisoformat(row[4]),
                })
            
            conn.close()
            
            logger.info(f"Found {len(positions)} active positions")
            return positions
            
        except Exception as e:
            logger.error(f"Error getting active positions: {e}")
            return []
    
    def get_current_price(self, symbol):
        """Get the current price of a stock"""
        try:
            # Get latest quote
            quote = self.api.get_latest_quote(symbol)
            price = (quote.ap + quote.bp) / 2  # Average of ask and bid price
            
            logger.info(f"Current price of {symbol}: ${price:.2f}")
            return price
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            
            # Try alternative method if quote fails
            try:
                barset = self.api.get_barset(symbol, 'minute', limit=1)
                price = barset[symbol][0].c  # Close price of the latest bar
                
                logger.info(f"Current price of {symbol} (from barset): ${price:.2f}")
                return price
                
            except Exception as e2:
                logger.error(f"Error getting barset for {symbol}: {e2}")
                return None
    
    def update_position(self, position_id, current_price, profit_loss):
        """Update a position in the database with current price and profit/loss"""
        try:
            conn = sqlite3.connect('tradebot.db')
            cursor = conn.cursor()
            
            cursor.execute('''
            UPDATE positions
            SET current_price = ?, profit_loss = ?
            WHERE id = ?
            ''', (current_price, profit_loss, position_id))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Updated position {position_id} with price ${current_price:.2f} and P/L {profit_loss:.2f}%")
            
        except Exception as e:
            logger.error(f"Error updating position {position_id}: {e}")
    
    def should_sell(self, position, current_price):
        """Determine if a position should be sold based on our criteria"""
        # Calculate profit/loss percentage
        profit_loss_pct = ((current_price / position['buy_price']) - 1) * 100
        
        # Check profit target
        if profit_loss_pct >= self.profit_target:
            logger.info(f"Profit target reached: {profit_loss_pct:.2f}% >= {self.profit_target:.2f}%")
            return True, "profit_target"
        
        # Check stop loss
        if profit_loss_pct <= -self.stop_loss:
            logger.info(f"Stop loss triggered: {profit_loss_pct:.2f}% <= -{self.stop_loss:.2f}%")
            return True, "stop_loss"
        
        # Check max hold time
        hold_time = datetime.now() - position['buy_time']
        max_hold_time_delta = timedelta(hours=self.max_hold_time)
        
        if hold_time >= max_hold_time_delta:
            logger.info(f"Max hold time reached: {hold_time} >= {max_hold_time_delta}")
            return True, "max_hold_time"
        
        # Not time to sell yet
        return False, None
    
    def check_positions(self):
        """Check all active positions and determine if any should be sold"""
        positions = self.get_active_positions()
        positions_to_sell = []
        
        for position in positions:
            symbol = position['symbol']
            current_price = self.get_current_price(symbol)
            
            if current_price is None:
                logger.warning(f"Could not get current price for {symbol}, skipping")
                continue
            
            # Calculate profit/loss percentage
            profit_loss_pct = ((current_price / position['buy_price']) - 1) * 100
            
            # Update position in database
            self.update_position(position['id'], current_price, profit_loss_pct)
            
            # Log current status
            logger.info(f"Position {position['id']}: {symbol} - Buy: ${position['buy_price']:.2f}, Current: ${current_price:.2f}, P/L: {profit_loss_pct:.2f}%")
            
            # Check if we should sell
            should_sell, reason = self.should_sell(position, current_price)
            
            if should_sell:
                position['current_price'] = current_price
                position['profit_loss_pct'] = profit_loss_pct
                position['sell_reason'] = reason
                positions_to_sell.append(position)
        
        return positions_to_sell
    
    def monitor_positions(self):
        """Monitor all active positions and return any that should be sold"""
        logger.info("Monitoring active positions...")
        
        # Check if market is open
        try:
            clock = self.api.get_clock()
            if not clock.is_open:
                logger.info("Market is closed. Monitoring positions but no trading.")
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
        
        # Check positions
        positions_to_sell = self.check_positions()
        
        if positions_to_sell:
            logger.info(f"Found {len(positions_to_sell)} positions to sell")
            return positions_to_sell
        else:
            logger.info("No positions need to be sold at this time")
            return []

# Test the monitor engine if run directly
if __name__ == "__main__":
    monitor = MonitorEngine()
    positions_to_sell = monitor.monitor_positions()
    
    if positions_to_sell:
        for position in positions_to_sell:
            print(f"Should sell {position['symbol']} - P/L: {position['profit_loss_pct']:.2f}% - Reason: {position['sell_reason']}")
    else:
        print("No positions to sell at this time")
