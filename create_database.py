import sqlite3
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_database():
    """Create the SQLite database and tables for the trading bot"""
    
    # Connect to database (creates it if it doesn't exist)
    conn = sqlite3.connect('tradebot.db')
    cursor = conn.cursor()
    
    # Create positions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        quantity REAL NOT NULL,
        buy_price REAL NOT NULL,
        buy_time TIMESTAMP NOT NULL,
        current_price REAL,
        profit_loss REAL,
        status TEXT DEFAULT 'active'
    )
    ''')
    
    # Create trades table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        action TEXT NOT NULL,
        quantity REAL NOT NULL,
        price REAL NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        profit_loss REAL
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
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print("Database created successfully!")

if __name__ == "__main__":
    create_database()
