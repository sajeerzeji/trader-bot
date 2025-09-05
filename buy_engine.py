import os
import sqlite3
import logging
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tradebot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('BuyEngine')

# Load environment variables
load_dotenv()

class BuyEngine:
    def __init__(self):
        """Initialize the BuyEngine with API keys and settings"""
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
        
        logger.info("BuyEngine initialized")
    
    def load_settings(self):
        """Load settings from the database and environment variables"""
        try:
            # Load from environment variables first
            self.initial_account_size = float(os.getenv('INITIAL_ACCOUNT_SIZE', 10.0))
            self.min_positions = int(os.getenv('MIN_POSITIONS', 3))
            self.max_positions = int(os.getenv('MAX_POSITIONS', 5))
            
            # Calculate max position size as percentage of account
            max_position_pct = float(os.getenv('MAX_POSITION_SIZE', 0.3))
            self.max_position_size = max_position_pct * self.initial_account_size
            
            conn = sqlite3.connect('tradebot.db')
            cursor = conn.cursor()
            
            # Get all settings
            cursor.execute('SELECT name, value FROM settings')
            settings = dict(cursor.fetchall())
            
            # Load remaining settings from database
            self.cash_reserve = float(settings.get('cash_reserve', 2.0))
            
            conn.close()
            logger.info("Settings loaded from database and environment variables")
            logger.info(f"Initial account size: ${self.initial_account_size:.2f}")
            logger.info(f"Max position size: ${self.max_position_size:.2f} ({max_position_pct*100:.0f}% of account)")
            logger.info(f"Target positions: {self.min_positions}-{self.max_positions}")
            
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
            # Use default values if loading fails
            self.initial_account_size = 10.0
            self.min_positions = 3
            self.max_positions = 5
            self.max_position_size = 3.0  # 30% of $10
            self.cash_reserve = 2.0
    
    def check_market_status(self):
        """Check if the market is open"""
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            return False
    
    def get_account_info(self):
        """Get account information including cash balance"""
        try:
            account = self.api.get_account()
            return {
                'cash': float(account.cash),
                'equity': float(account.equity),
                'buying_power': float(account.buying_power)
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None
    
    def calculate_position_size(self, stock_price, num_positions=1):
        """Calculate how many shares to buy based on available cash and max position size
        
        Args:
            stock_price: Current price of the stock
            num_positions: Number of positions to allocate cash for
            
        Returns:
            tuple: (quantity, dollar_amount)
        """
        try:
            # Get account info
            account_info = self.get_account_info()
            
            # Use initial account size if account info is not available
            if not account_info:
                logger.warning(f"Could not get account info, using initial account size: ${self.initial_account_size:.2f}")
                available_cash = self.initial_account_size - self.cash_reserve
            else:
                available_cash = float(account_info['cash']) - self.cash_reserve
            
            # Calculate per-position size based on number of positions
            # Ensure we don't exceed max_position_size per position
            per_position_size = min(
                available_cash / max(num_positions, 1),  # Divide available cash
                self.max_position_size                   # Cap at max position size
            )
            
            # If not enough cash, return 0
            if per_position_size <= 0:
                logger.warning(f"Not enough cash available. Available: ${available_cash:.2f}")
                return 0, 0
            
            # Calculate quantity (can be fractional)
            quantity = per_position_size / stock_price
            
            logger.info(f"Calculated position size: ${per_position_size:.2f} = {quantity:.4f} shares at ${stock_price:.2f}")
            return quantity, per_position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0, 0
    
    def place_buy_order(self, symbol, price, num_positions=1):
        """Place a buy order for a stock
        
        Args:
            symbol: Stock symbol
            price: Current price
            num_positions: Number of positions to allocate cash for
            
        Returns:
            dict: Order information if successful, None otherwise
        """
        try:
            # Check if market is open
            if not self.check_market_status():
                logger.warning("Market is closed. Cannot place order.")
                return None
            
            # Calculate quantity to buy
            quantity, dollar_amount = self.calculate_position_size(price, num_positions)
            if quantity <= 0:
                logger.warning(f"Cannot buy {symbol}: insufficient funds or zero quantity")
                return None
            
            # First check if the asset is fractionable
            try:
                asset = self.api.get_asset(symbol)
                is_fractionable = asset.fractionable
            except Exception as e:
                logger.warning(f"Could not determine if {symbol} is fractionable: {e}")
                is_fractionable = False
            
            # Handle non-fractionable stocks by rounding down to whole shares
            if not is_fractionable:
                original_quantity = quantity
                quantity = int(quantity)  # Round down to whole number
                logger.info(f"Asset {symbol} is not fractionable. Adjusting quantity from {original_quantity:.4f} to {quantity}")
                
                # If quantity becomes 0 after rounding, skip this stock
                if quantity == 0:
                    logger.warning(f"Cannot buy {symbol}: quantity would be 0 after rounding to whole shares")
                    return None
                
                # Recalculate dollar amount for whole shares
                dollar_amount = quantity * price
            else:
                # Round to 4 decimal places for fractional shares
                quantity = round(quantity, 4)
            
            # Place market order
            order = self.api.submit_order(
                symbol=symbol,
                qty=quantity,
                side='buy',
                type='market',
                time_in_force='day'
            )
            
            logger.info(f"Buy order placed for {quantity} shares of {symbol} at ~${price:.2f} (${dollar_amount:.2f})")
            
            # Record the order in the database
            self.record_trade(symbol, 'BUY', quantity, price)
            
            return {
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'order_id': order.id,
                'status': order.status
            }
            
        except Exception as e:
            logger.error(f"Error placing buy order for {symbol}: {e}")
            return None
    
    def buy_multiple_stocks(self, stock_list):
        """Buy multiple stocks
        
        Args:
            stock_list: List of dictionaries with symbol, price, and score
            
        Returns:
            list: List of order results
        """
        if not stock_list:
            logger.warning("No stocks provided to buy_multiple_stocks")
            return []
        
        # Check if market is open
        if not self.check_market_status():
            logger.warning("Market is closed. Cannot place orders.")
            return []
        
        # Pre-check which stocks are fractionable
        for stock in stock_list:
            symbol = stock['symbol']
            try:
                asset = self.api.get_asset(symbol)
                stock['fractionable'] = asset.fractionable
            except Exception as e:
                logger.warning(f"Could not determine if {symbol} is fractionable: {e}")
                stock['fractionable'] = False
        
        # For non-fractionable stocks, check if we can buy at least 1 share
        per_position_size = self.max_position_size * self.initial_account_size
        for stock in stock_list:
            if not stock['fractionable']:
                if stock['price'] > per_position_size:
                    stock['affordable'] = False
                    logger.warning(f"{stock['symbol']} at ${stock['price']} is not affordable with position size ${per_position_size}")
                else:
                    stock['affordable'] = True
            else:
                stock['affordable'] = True
        
        # First prioritize by affordability, then by score
        affordable_stocks = [s for s in stock_list if s.get('affordable', True)]
        if not affordable_stocks:
            logger.warning("No affordable stocks found with our position size")
            return []
            
        # Sort affordable stocks by score (highest first)
        affordable_stocks = sorted(affordable_stocks, key=lambda x: x.get('score', 0), reverse=True)
        
        # Determine how many stocks we'll actually buy
        num_stocks = min(len(affordable_stocks), self.max_positions)
        stocks_to_buy = affordable_stocks[:num_stocks]
        
        logger.info(f"Attempting to buy {num_stocks} stocks: {[s['symbol'] for s in stocks_to_buy]}")
        
        # Place orders
        orders = []
        for stock in stocks_to_buy:
            symbol = stock['symbol']
            price = stock['price']
            
            # Place order with position sizing for multiple positions
            order_result = self.place_buy_order(symbol, price, num_stocks)
            if order_result:
                orders.append(order_result)
        
        logger.info(f"Successfully placed {len(orders)} buy orders out of {num_stocks} attempted")
        return orders
    
    def buy_stock(self, stock_info):
        """Buy a single stock
        
        Args:
            stock_info: Dictionary with symbol and price
            
        Returns:
            dict: Order information if successful, None otherwise
        """
        if not stock_info or 'symbol' not in stock_info or 'price' not in stock_info:
            logger.warning("Invalid stock information provided")
            return None
        
        symbol = stock_info['symbol']
        price = stock_info['price']
        
        return self.place_buy_order(symbol, price)
    
    def record_trade(self, symbol, action, quantity, price):
        """Record a trade in the database"""
        try:
            conn = sqlite3.connect('tradebot.db')
            cursor = conn.cursor()
            
            # Insert into trades table
            cursor.execute('''
            INSERT INTO trades (symbol, action, quantity, price, timestamp)
            VALUES (?, ?, ?, ?, ?)
            ''', (symbol, action, quantity, price, datetime.now()))
            
            # If this is a buy, also insert into positions table
            if action == 'BUY':
                cursor.execute('''
                INSERT INTO positions (symbol, quantity, buy_price, buy_time, current_price)
                VALUES (?, ?, ?, ?, ?)
                ''', (symbol, quantity, price, datetime.now(), price))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Trade recorded: {action} {quantity} shares of {symbol} at ${price:.2f}")
            
        except Exception as e:
            logger.error(f"Error recording trade in database: {e}")
    
    def check_order_status(self, order_id):
        """Check the status of an order"""
        try:
            order = self.api.get_order(order_id)
            return order.status
        except Exception as e:
            logger.error(f"Error checking order status: {e}")
            return None
    
    def buy_stock(self, stock_info):
        """Buy a stock based on the provided information"""
        if not stock_info or 'symbol' not in stock_info or 'price' not in stock_info:
            logger.error("Invalid stock information provided")
            return None
        
        symbol = stock_info['symbol']
        price = stock_info['price']
        
        logger.info(f"Attempting to buy {symbol} at ${price:.2f}")
        
        # Place the buy order
        order_info = self.place_buy_order(symbol, price)
        
        if order_info:
            # Wait for order to fill (in a real system, this would be async)
            status = self.check_order_status(order_info['order_id'])
            logger.info(f"Order status: {status}")
            
            return order_info
        else:
            logger.warning(f"Failed to place buy order for {symbol}")
            return None

# Test the buy engine if run directly
if __name__ == "__main__":
    buyer = BuyEngine()
    account = buyer.get_account_info()
    
    if account:
        print(f"Account cash: ${account['cash']:.2f}")
        print(f"Account equity: ${account['equity']:.2f}")
        print(f"Buying power: ${account['buying_power']:.2f}")
        
        # Test with a sample stock
        test_stock = {
            'symbol': 'AAPL',
            'price': 150.00
        }
        
        quantity = buyer.calculate_position_size(test_stock['price'])
        print(f"Would buy {quantity:.4f} shares of {test_stock['symbol']} at ${test_stock['price']:.2f}")
        
        # Uncomment to actually place an order
        # order = buyer.buy_stock(test_stock)
        # if order:
        #     print(f"Order placed: {order}")
    else:
        print("Could not retrieve account information")
