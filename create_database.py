import sqlite3
import os
import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_database():
    """Create the SQLite database and tables for the trading bot"""
    
    # Create data directory if it doesn't exist
    os.makedirs('data/db', exist_ok=True)
    
    # Connect to database (creates it if it doesn't exist)
    conn = sqlite3.connect('data/db/tradebot.db')
    cursor = conn.cursor()
    
    # Create positions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        quantity REAL NOT NULL,
        buy_price REAL NOT NULL,
        current_price REAL NOT NULL,
        buy_date TIMESTAMP NOT NULL,
        sell_price REAL,
        sell_date TIMESTAMP,
        status TEXT DEFAULT 'open',
        stop_loss_price REAL
    )
    ''')
    
    # Create orders table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        quantity REAL NOT NULL,
        price REAL NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        status TEXT DEFAULT 'filled',
        source TEXT
    )
    ''')
    
    # Create settings table and insert default values from .env
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS settings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        value REAL NOT NULL,
        description TEXT
    )
    ''')
    
    # Insert default settings from .env file
    settings = [
        ('max_position_size', float(os.getenv('MAX_POSITION_SIZE', 8.0)), 'Maximum to spend per stock'),
        ('profit_target', float(os.getenv('PROFIT_TARGET', 5.0)), 'Sell when stock gains this percentage'),
        ('stop_loss', float(os.getenv('STOP_LOSS', 3.0)), 'Sell when stock loses this percentage'),
        ('max_hold_time', float(os.getenv('MAX_HOLD_TIME', 24.0)), 'Sell after this many hours'),
        ('check_interval', float(os.getenv('CHECK_INTERVAL', 30.0)), 'How often to check prices (minutes)'),
        ('min_price', float(os.getenv('MIN_PRICE', 1.0)), 'Minimum stock price to consider'),
        ('max_price', float(os.getenv('MAX_PRICE', 100.0)), 'Maximum stock price to consider'),
        ('min_volume', float(os.getenv('MIN_VOLUME', 1000000.0)), 'Minimum trading volume'),
        ('max_loss_per_trade', float(os.getenv('MAX_LOSS_PER_TRADE', 2.0)), 'Maximum loss per trade'),
        ('max_daily_loss', float(os.getenv('MAX_DAILY_LOSS', 3.0)), 'Maximum daily loss'),
        ('cash_reserve', float(os.getenv('CASH_RESERVE', 2.0)), 'Cash to keep in reserve')
    ]
    
    # Use INSERT OR REPLACE to update settings if they already exist
    cursor.executemany('''
    INSERT OR REPLACE INTO settings (name, value, description) 
    VALUES (?, ?, ?)
    ''', settings)
    
    # Create performance table to track daily performance
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS performance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date DATE UNIQUE NOT NULL,
        starting_balance REAL NOT NULL,
        ending_balance REAL NOT NULL,
        profit_loss REAL NOT NULL,
        win_count INTEGER DEFAULT 0,
        loss_count INTEGER DEFAULT 0,
        total_trades INTEGER DEFAULT 0
    )
    ''')
    
    # Add sample data from the screenshot
    def populate_sample_data():
        # Sample orders data from screenshot
        sample_orders = [
            ('AUPH', 'buy', 0.2415, 12.438, '2025-09-08 08:52:40', 'filled', 'access_key'),
            ('ATEX', 'buy', 0.2745, 22.138, '2025-09-08 08:52:39', 'filled', 'access_key'),
            ('ASX', 'buy', 0.2619, 11.458, '2025-09-08 08:52:38', 'filled', 'access_key'),
            ('ASUR', 'buy', 0.3686, 8.162, '2025-09-08 08:52:36', 'filled', 'access_key'),
            ('ATAI', 'buy', 2.027, 4.55, '2025-09-08 08:52:35', 'filled', 'access_key'),
            ('CWCO', 'sell', 0.1784, 33.918, '2025-09-08 08:50:46', 'filled', 'access_key'),
            ('CWCO', 'sell', 0.1802, 33.918, '2025-09-08 08:50:45', 'filled', 'access_key'),
            ('CWCO', 'sell', 0.1802, 33.918, '2025-09-08 08:50:44', 'filled', 'access_key'),
            ('CWCO', 'sell', 0.1802, 33.918, '2025-09-08 08:50:43', 'filled', 'access_key'),
            ('CWK', 'buy', 0.1844, 16.268, '2025-09-08 07:50:22', 'filled', 'access_key'),
            ('CWH', 'buy', 0.1706, 17.596, '2025-09-08 07:50:21', 'filled', 'access_key'),
            ('CWCO', 'buy', 0.1784, 33.77, '2025-09-08 07:50:19', 'filled', 'access_key'),
            ('CVLG', 'buy', 0.2512, 23.788, '2025-09-08 07:50:18', 'filled', 'access_key'),
            ('CVGI', 'buy', 1.00, 1.80, '2025-09-08 07:50:17', 'filled', 'access_key'),
            ('CWH', 'sell', 0.1705, 17.566, '2025-09-08 07:48:31', 'filled', 'access_key'),
            ('CWK', 'buy', 0.1844, 16.248, '2025-09-08 07:42:32', 'filled', 'access_key'),
            ('CWH', 'buy', 0.1705, 17.584, '2025-09-08 07:42:31', 'filled', 'access_key'),
            ('CWCO', 'buy', 0.1802, 33.794, '2025-09-08 07:42:30', 'filled', 'access_key')
        ]
        
        # Insert sample orders
        cursor.executemany('''
        INSERT INTO orders (symbol, side, quantity, price, timestamp, status, source)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', sample_orders)
        
        # Create sample positions based on orders
        # Group buy orders by symbol
        buy_orders = {}
        for order in sample_orders:
            symbol, side, quantity, price = order[0], order[1], order[2], order[3]
            if side == 'buy':
                if symbol not in buy_orders:
                    buy_orders[symbol] = []
                buy_orders[symbol].append((quantity, price, order[4]))
        
        # Create positions from buy orders
        sample_positions = []
        for symbol, orders in buy_orders.items():
            for qty, price, timestamp in orders:
                # Set some positions as closed (50% chance)
                import random
                is_closed = random.choice([True, False])
                status = 'closed' if is_closed else 'open'
                
                # For closed positions, add sell data
                sell_price = None
                sell_date = None
                if is_closed:
                    # Find matching sell order or create fictional one
                    sell_price = price * random.uniform(0.95, 1.15)  # +/- 15%
                    from datetime import datetime, timedelta
                    buy_date = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                    sell_date = (buy_date + timedelta(hours=random.randint(1, 8))).strftime('%Y-%m-%d %H:%M:%S')
                
                # Current price is either sell price or slightly different from buy price
                current_price = sell_price if sell_price else price * random.uniform(0.97, 1.10)
                
                # Add position
                sample_positions.append((
                    symbol, 
                    qty, 
                    price, 
                    current_price, 
                    timestamp,  # buy_date 
                    sell_price, 
                    sell_date, 
                    status,
                    price * 0.95  # stop_loss_price
                ))
        
        # Insert sample positions
        cursor.executemany('''
        INSERT INTO positions (symbol, quantity, buy_price, current_price, buy_date, sell_price, sell_date, status, stop_loss_price)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', sample_positions)
    
    # Call the function to populate sample data
    populate_sample_data()
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print("Database created successfully with sample data!")

if __name__ == "__main__":
    create_database()
