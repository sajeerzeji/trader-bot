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
logger = logging.getLogger('SellEngine')

# Load environment variables
load_dotenv()

class SellEngine:
    def __init__(self):
        """Initialize the SellEngine with API keys and settings"""
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
        
        logger.info("SellEngine initialized")
    
    def check_market_status(self):
        """Check if the market is open"""
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            return False
    
    def place_sell_order(self, position):
        """Place a sell order for a position"""
        try:
            # Check if market is open
            if not self.check_market_status():
                logger.warning("Market is closed. Cannot place sell order.")
                return None
            
            symbol = position['symbol']
            quantity = position['quantity']
            
            # Place market order
            order = self.api.submit_order(
                symbol=symbol,
                qty=quantity,
                side='sell',
                type='market',
                time_in_force='day'
            )
            
            logger.info(f"Sell order placed: {quantity} shares of {symbol}")
            
            return {
                'symbol': symbol,
                'quantity': quantity,
                'price': position.get('current_price'),
                'order_id': order.id,
                'position_id': position['id'],
                'profit_loss_pct': position.get('profit_loss_pct', 0),
                'sell_reason': position.get('sell_reason', 'manual')
            }
            
        except Exception as e:
            logger.error(f"Error placing sell order for {position['symbol']}: {e}")
            return None
    
    def check_order_status(self, order_id):
        """Check the status of an order"""
        try:
            order = self.api.get_order(order_id)
            return order.status
        except Exception as e:
            logger.error(f"Error checking order status: {e}")
            return None
    
    def record_sale(self, sale_info):
        """Record a sale in the database"""
        try:
            conn = sqlite3.connect('tradebot.db')
            cursor = conn.cursor()
            
            # Update position status
            cursor.execute('''
            UPDATE positions
            SET status = 'closed', current_price = ?, profit_loss = ?
            WHERE id = ?
            ''', (sale_info['price'], sale_info['profit_loss_pct'], sale_info['position_id']))
            
            # Insert into trades table
            cursor.execute('''
            INSERT INTO trades (symbol, action, quantity, price, timestamp, profit_loss)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (sale_info['symbol'], 'SELL', sale_info['quantity'], 
                  sale_info['price'], datetime.now(), sale_info['profit_loss_pct']))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Sale recorded: {sale_info['quantity']} shares of {sale_info['symbol']} at ${sale_info['price']:.2f}, P/L: {sale_info['profit_loss_pct']:.2f}%")
            
        except Exception as e:
            logger.error(f"Error recording sale in database: {e}")
    
    def update_performance(self, sale_info):
        """Update daily performance metrics"""
        try:
            conn = sqlite3.connect('tradebot.db')
            cursor = conn.cursor()
            
            today = datetime.now().date().isoformat()
            
            # Check if we have a performance record for today
            cursor.execute('SELECT * FROM performance WHERE date = ?', (today,))
            performance = cursor.fetchone()
            
            # Get account info for current balance
            try:
                account = self.api.get_account()
                ending_balance = float(account.equity)
            except Exception as e:
                logger.error(f"Error getting account info: {e}")
                ending_balance = 0
            
            # Determine if this was a win or loss
            is_win = sale_info['profit_loss_pct'] > 0
            
            if performance:
                # Update existing performance record
                performance_id = performance[0]
                win_count = performance[5] + (1 if is_win else 0)
                loss_count = performance[6] + (0 if is_win else 1)
                total_trades = performance[7] + 1
                
                cursor.execute('''
                UPDATE performance
                SET ending_balance = ?, profit_loss = ending_balance - starting_balance,
                    win_count = ?, loss_count = ?, total_trades = ?
                WHERE id = ?
                ''', (ending_balance, win_count, loss_count, total_trades, performance_id))
            else:
                # Create new performance record for today
                cursor.execute('''
                INSERT INTO performance (date, starting_balance, ending_balance, profit_loss, win_count, loss_count, total_trades)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (today, ending_balance, ending_balance, 0, 
                      1 if is_win else 0, 0 if is_win else 1, 1))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Performance updated for {today}")
            
        except Exception as e:
            logger.error(f"Error updating performance: {e}")
    
    def sell_position(self, position):
        """Sell a position and record the sale"""
        logger.info(f"Selling position: {position['symbol']}")
        
        # Place sell order
        sale_info = self.place_sell_order(position)
        
        if sale_info:
            # Wait for order to fill (in a real system, this would be async)
            status = self.check_order_status(sale_info['order_id'])
            logger.info(f"Order status: {status}")
            
            # Record the sale in the database
            self.record_sale(sale_info)
            
            # Update performance metrics
            self.update_performance(sale_info)
            
            return sale_info
        else:
            logger.warning(f"Failed to place sell order for {position['symbol']}")
            return None
    
    def sell_positions(self, positions_to_sell):
        """Sell multiple positions"""
        results = []
        
        for position in positions_to_sell:
            result = self.sell_position(position)
            if result:
                results.append(result)
        
        return results

# Test the sell engine if run directly
if __name__ == "__main__":
    seller = SellEngine()
    
    # Test with a sample position
    test_position = {
        'id': 1,
        'symbol': 'AAPL',
        'quantity': 0.1,
        'buy_price': 150.00,
        'current_price': 155.00,
        'profit_loss_pct': 3.33,
        'sell_reason': 'profit_target'
    }
    
    # Uncomment to test selling
    # result = seller.sell_position(test_position)
    # if result:
    #     print(f"Position sold: {result}")
    # else:
    #     print("Failed to sell position")
