import os
import time
import logging
import schedule
import sqlite3
from dotenv import load_dotenv
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi

# Import our components
from stock_scanner import StockScanner
from buy_engine import BuyEngine
from monitor_engine import MonitorEngine
from sell_engine import SellEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tradebot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MainController')

# Load environment variables
load_dotenv()

class MainController:
    def __init__(self):
        """Initialize the Main Controller with components and settings"""
        # Initialize database if it doesn't exist
        self.init_database()
        
        # Load settings
        self.load_settings()
        
        # Initialize components
        self.scanner = StockScanner()
        self.buyer = BuyEngine()
        self.monitor = MonitorEngine()
        self.seller = SellEngine()
        
        # Initialize Alpaca API for market hours
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.api_secret = os.getenv('ALPACA_API_SECRET')
        self.api_endpoint = os.getenv('ALPACA_API_ENDPOINT')
        
        self.api = tradeapi.REST(
            self.api_key,
            self.api_secret,
            base_url=self.api_endpoint
        )
        
        logger.info("MainController initialized")
    
    def init_database(self):
        """Initialize the database if it doesn't exist"""
        if not os.path.exists('tradebot.db'):
            logger.info("Database not found, creating...")
            try:
                from create_database import create_database
                create_database()
            except Exception as e:
                logger.error(f"Error creating database: {e}")
                raise
    
    def load_settings(self):
        """Load settings from the database"""
        try:
            conn = sqlite3.connect('tradebot.db')
            cursor = conn.cursor()
            
            # Get all settings
            cursor.execute('SELECT name, value FROM settings')
            settings = dict(cursor.fetchall())
            
            # Assign settings to instance variables
            self.check_interval = int(settings.get('check_interval', 30))
            
            conn.close()
            logger.info("Settings loaded from database")
            
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
            # Use default values if database access fails
            self.check_interval = 30
    
    def is_market_open(self):
        """Check if the market is currently open"""
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            return False
    
    def get_next_market_open(self):
        """Get the next market open time"""
        try:
            clock = self.api.get_clock()
            return clock.next_open
        except Exception as e:
            logger.error(f"Error getting next market open: {e}")
            return None
    
    def get_active_positions_count(self):
        """Get the count of active positions"""
        try:
            conn = sqlite3.connect('tradebot.db')
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM positions WHERE status = "active"')
            count = cursor.fetchone()[0]
            
            conn.close()
            return count
            
        except Exception as e:
            logger.error(f"Error getting active positions count: {e}")
            return 0
    
    def trading_cycle(self):
        """Run one complete trading cycle"""
        logger.info("Starting trading cycle")
        
        # Check if market is open
        if not self.is_market_open():
            next_open = self.get_next_market_open()
            if next_open:
                logger.info(f"Market is closed. Next open: {next_open}")
            else:
                logger.info("Market is closed.")
            return
        
        # Check if we have any active positions
        active_positions = self.get_active_positions_count()
        
        # Get portfolio manager to check if we need more positions
        from portfolio_manager import PortfolioManager
        portfolio = PortfolioManager()
        needs_more, positions_needed = portfolio.needs_more_positions()
        
        if active_positions > 0:
            logger.info(f"Found {active_positions} active positions. Monitoring...")
            
            # Monitor positions
            positions_to_sell = self.monitor.monitor_positions()
            
            # Sell positions if needed
            if positions_to_sell:
                logger.info(f"Selling {len(positions_to_sell)} positions")
                self.seller.sell_positions(positions_to_sell)
                
                # After selling, find new stocks to buy
                self.find_and_buy_stocks()
            elif needs_more:
                # We have active positions but need more to meet minimum requirement
                logger.info(f"Need {positions_needed} more positions to meet minimum requirement of {portfolio.min_positions}")
                self.find_and_buy_stocks()
            else:
                logger.info("No positions need to be sold at this time")
        else:
            logger.info("No active positions. Finding stocks to buy...")
            self.find_and_buy_stocks()
    
    def find_and_buy_stocks(self):
        """Find multiple stocks and buy them"""
        # Get portfolio manager to check if we need more positions
        from portfolio_manager import PortfolioManager
        portfolio = PortfolioManager()
        needs_more, positions_needed = portfolio.needs_more_positions()
        
        # Find best stocks
        best_stocks = self.scanner.find_best_stocks(
            min_positions=portfolio.min_positions,
            max_positions=portfolio.max_positions
        )
        
        if best_stocks:
            symbols = [stock['symbol'] for stock in best_stocks]
            logger.info(f"Found {len(best_stocks)} candidate stocks: {', '.join(symbols)}")
            
            # Buy the stocks
            order_results = self.buyer.buy_multiple_stocks(best_stocks)
            
            if order_results:
                logger.info(f"Successfully bought {len(order_results)} stocks")
                for order in order_results:
                    logger.info(f"Bought {order['quantity']} shares of {order['symbol']} at ~${order['price']:.2f}")
            else:
                logger.warning("Failed to buy any stocks")
        else:
            logger.warning("No suitable stocks found")
    
    def run(self):
        """Run the trading bot continuously"""
        logger.info("Starting trading bot")
        
        # Run once immediately
        self.trading_cycle()
        
        # Schedule to run at regular intervals
        schedule.every(self.check_interval).minutes.do(self.trading_cycle)
        
        # Keep running
        while True:
            schedule.run_pending()
            time.sleep(1)

if __name__ == "__main__":
    try:
        controller = MainController()
        controller.run()
    except KeyboardInterrupt:
        logger.info("Trading bot stopped by user")
    except Exception as e:
        logger.error(f"Error running trading bot: {e}")
