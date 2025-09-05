import os
import logging
import sqlite3
import json
import time
import hmac
import hashlib
import base64
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tradebot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MultiExchange')

# Load environment variables
load_dotenv()

class ExchangeBase(ABC):
    """Base class for exchange implementations"""
    
    def __init__(self, exchange_id):
        """Initialize the exchange
        
        Args:
            exchange_id: Unique identifier for the exchange
        """
        self.exchange_id = exchange_id
        self.name = None
        self.description = None
        self.enabled = False
        self.load_settings()
        
        logger.info(f"Initialized {self.name} exchange")
    
    def load_settings(self):
        """Load exchange settings from database"""
        try:
            conn = sqlite3.connect('tradebot.db')
            cursor = conn.cursor()
            
            # Check if exchanges table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='exchanges'")
            if not cursor.fetchone():
                self._create_exchanges_table(cursor)
            
            # Get exchange settings
            cursor.execute(
                "SELECT name, description, enabled, settings FROM exchanges WHERE exchange_id = ?",
                (self.exchange_id,)
            )
            
            row = cursor.fetchone()
            if row:
                self.name = row[0]
                self.description = row[1]
                self.enabled = row[2] == 1
                self.settings = json.loads(row[3]) if row[3] else {}
            else:
                # Insert default settings
                self._insert_default_settings(cursor)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error loading exchange settings: {e}")
            self.settings = {}
    
    def _create_exchanges_table(self, cursor):
        """Create exchanges table if it doesn't exist"""
        cursor.execute('''
        CREATE TABLE exchanges (
            exchange_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            enabled INTEGER DEFAULT 0,
            settings TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        logger.info("Created exchanges table")
    
    def _insert_default_settings(self, cursor):
        """Insert default settings for this exchange"""
        default_name = self.exchange_id.capitalize()
        default_description = f"{default_name} Exchange"
        default_settings = self.get_default_settings()
        
        cursor.execute(
            "INSERT INTO exchanges (exchange_id, name, description, enabled, settings) VALUES (?, ?, ?, ?, ?)",
            (self.exchange_id, default_name, default_description, 0, json.dumps(default_settings))
        )
        
        self.name = default_name
        self.description = default_description
        self.enabled = False
        self.settings = default_settings
        
        logger.info(f"Inserted default settings for {self.exchange_id}")
    
    def save_settings(self):
        """Save exchange settings to database"""
        try:
            conn = sqlite3.connect('tradebot.db')
            cursor = conn.cursor()
            
            cursor.execute(
                "UPDATE exchanges SET name = ?, description = ?, enabled = ?, settings = ?, updated_at = ? WHERE exchange_id = ?",
                (
                    self.name,
                    self.description,
                    1 if self.enabled else 0,
                    json.dumps(self.settings),
                    datetime.now().isoformat(),
                    self.exchange_id
                )
            )
            
            conn.commit()
            conn.close()
            
            logger.info(f"Saved settings for {self.exchange_id}")
            
        except Exception as e:
            logger.error(f"Error saving exchange settings: {e}")
    
    def enable(self):
        """Enable the exchange"""
        self.enabled = True
        self.save_settings()
        logger.info(f"Enabled {self.name} exchange")
    
    def disable(self):
        """Disable the exchange"""
        self.enabled = False
        self.save_settings()
        logger.info(f"Disabled {self.name} exchange")
    
    def is_enabled(self):
        """Check if the exchange is enabled
        
        Returns:
            bool: True if enabled, False otherwise
        """
        return self.enabled
    
    def update_setting(self, key, value):
        """Update a setting
        
        Args:
            key: Setting key
            value: Setting value
        """
        self.settings[key] = value
        self.save_settings()
    
    @abstractmethod
    def get_default_settings(self):
        """Get default settings for this exchange
        
        Returns:
            dict: Default settings
        """
        pass
    
    @abstractmethod
    def get_account(self):
        """Get account information
        
        Returns:
            dict: Account information
        """
        pass
    
    @abstractmethod
    def get_positions(self):
        """Get current positions
        
        Returns:
            list: Current positions
        """
        pass
    
    @abstractmethod
    def get_balance(self):
        """Get account balance
        
        Returns:
            float: Account balance
        """
        pass
    
    @abstractmethod
    def get_market_data(self, symbol, timeframe='1d', limit=100):
        """Get market data
        
        Args:
            symbol: Symbol to get data for
            timeframe: Timeframe (e.g., '1m', '1h', '1d')
            limit: Maximum number of data points
            
        Returns:
            DataFrame: Market data
        """
        pass
    
    @abstractmethod
    def place_order(self, symbol, side, quantity, order_type='market', price=None):
        """Place an order
        
        Args:
            symbol: Symbol to trade
            side: 'buy' or 'sell'
            quantity: Order quantity
            order_type: Order type (market, limit, etc.)
            price: Price for limit orders
            
        Returns:
            dict: Order information
        """
        pass
    
    @abstractmethod
    def get_order(self, order_id):
        """Get order information
        
        Args:
            order_id: Order ID
            
        Returns:
            dict: Order information
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id):
        """Cancel an order
        
        Args:
            order_id: Order ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass


class AlpacaExchange(ExchangeBase):
    """Alpaca exchange implementation"""
    
    def __init__(self):
        """Initialize the Alpaca exchange"""
        super().__init__('alpaca')
        self.api = None
        self._connect()
    
    def _connect(self):
        """Connect to the Alpaca API"""
        try:
            api_key = self.settings.get('api_key') or os.getenv('ALPACA_API_KEY')
            api_secret = self.settings.get('api_secret') or os.getenv('ALPACA_API_SECRET')
            base_url = self.settings.get('base_url') or os.getenv('ALPACA_API_ENDPOINT')
            
            if not all([api_key, api_secret, base_url]):
                logger.warning("Alpaca API credentials not found")
                return
            
            self.api = tradeapi.REST(
                api_key,
                api_secret,
                base_url=base_url
            )
            
            # Test connection
            account = self.api.get_account()
            logger.info(f"Connected to Alpaca API: Account ID {account.id}")
            
        except Exception as e:
            logger.error(f"Error connecting to Alpaca API: {e}")
            self.api = None
    
    def get_default_settings(self):
        """Get default settings for Alpaca exchange"""
        return {
            'api_key': os.getenv('ALPACA_API_KEY', ''),
            'api_secret': os.getenv('ALPACA_API_SECRET', ''),
            'base_url': os.getenv('ALPACA_API_ENDPOINT', 'https://paper-api.alpaca.markets'),
            'max_positions': 5,
            'max_position_size': 0.2,  # 20% of account
            'default_timeframe': '1d'
        }
    
    def get_account(self):
        """Get account information"""
        if not self.api:
            return {'error': 'Not connected to Alpaca API'}
        
        try:
            account = self.api.get_account()
            
            return {
                'id': account.id,
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'buying_power': float(account.buying_power),
                'equity': float(account.equity),
                'currency': account.currency,
                'pattern_day_trader': account.pattern_day_trader,
                'trading_blocked': account.trading_blocked,
                'account_blocked': account.account_blocked,
                'created_at': account.created_at
            }
            
        except Exception as e:
            logger.error(f"Error getting Alpaca account: {e}")
            return {'error': str(e)}
    
    def get_positions(self):
        """Get current positions"""
        if not self.api:
            return []
        
        try:
            positions = self.api.list_positions()
            
            result = []
            for position in positions:
                result.append({
                    'symbol': position.symbol,
                    'quantity': float(position.qty),
                    'avg_entry_price': float(position.avg_entry_price),
                    'market_value': float(position.market_value),
                    'cost_basis': float(position.cost_basis),
                    'unrealized_pl': float(position.unrealized_pl),
                    'unrealized_plpc': float(position.unrealized_plpc),
                    'current_price': float(position.current_price),
                    'exchange': 'alpaca'
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting Alpaca positions: {e}")
            return []
    
    def get_balance(self):
        """Get account balance"""
        if not self.api:
            return 0.0
        
        try:
            account = self.api.get_account()
            return float(account.cash)
            
        except Exception as e:
            logger.error(f"Error getting Alpaca balance: {e}")
            return 0.0
    
    def get_market_data(self, symbol, timeframe='1d', limit=100):
        """Get market data"""
        if not self.api:
            return None
        
        try:
            # Convert timeframe to Alpaca format
            timeframe_map = {
                '1m': '1Min',
                '5m': '5Min',
                '15m': '15Min',
                '1h': '1Hour',
                '1d': '1Day'
            }
            
            alpaca_timeframe = timeframe_map.get(timeframe, '1Day')
            
            # Calculate start and end dates
            end_date = datetime.now()
            
            # Adjust start date based on timeframe and limit
            if timeframe == '1d':
                start_date = end_date - timedelta(days=limit)
            elif timeframe == '1h':
                start_date = end_date - timedelta(hours=limit)
            else:
                start_date = end_date - timedelta(minutes=int(timeframe[:-1]) * limit)
            
            # Format dates for Alpaca API
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Get bars
            bars = self.api.get_barset(symbol, alpaca_timeframe, start=start_str, end=end_str, limit=limit)
            
            if symbol not in bars:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame()
            symbol_bars = bars[symbol]
            
            df['timestamp'] = [bar.t for bar in symbol_bars]
            df['open'] = [bar.o for bar in symbol_bars]
            df['high'] = [bar.h for bar in symbol_bars]
            df['low'] = [bar.l for bar in symbol_bars]
            df['close'] = [bar.c for bar in symbol_bars]
            df['volume'] = [bar.v for bar in symbol_bars]
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting Alpaca market data: {e}")
            return None
    
    def place_order(self, symbol, side, quantity, order_type='market', price=None):
        """Place an order"""
        if not self.api:
            return {'error': 'Not connected to Alpaca API'}
        
        try:
            # Convert parameters to Alpaca format
            alpaca_side = side.lower()
            alpaca_type = order_type.lower()
            
            # Place order
            if alpaca_type == 'market':
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side=alpaca_side,
                    type=alpaca_type,
                    time_in_force='day'
                )
            elif alpaca_type == 'limit':
                if price is None:
                    return {'error': 'Price is required for limit orders'}
                
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side=alpaca_side,
                    type=alpaca_type,
                    time_in_force='day',
                    limit_price=price
                )
            else:
                return {'error': f'Unsupported order type: {order_type}'}
            
            return {
                'id': order.id,
                'symbol': order.symbol,
                'side': order.side,
                'quantity': float(order.qty),
                'type': order.type,
                'status': order.status,
                'created_at': order.created_at,
                'exchange': 'alpaca'
            }
            
        except Exception as e:
            logger.error(f"Error placing Alpaca order: {e}")
            return {'error': str(e)}
    
    def get_order(self, order_id):
        """Get order information"""
        if not self.api:
            return {'error': 'Not connected to Alpaca API'}
        
        try:
            order = self.api.get_order(order_id)
            
            return {
                'id': order.id,
                'symbol': order.symbol,
                'side': order.side,
                'quantity': float(order.qty),
                'filled_quantity': float(order.filled_qty),
                'type': order.type,
                'status': order.status,
                'created_at': order.created_at,
                'filled_at': order.filled_at,
                'exchange': 'alpaca'
            }
            
        except Exception as e:
            logger.error(f"Error getting Alpaca order: {e}")
            return {'error': str(e)}
    
    def cancel_order(self, order_id):
        """Cancel an order"""
        if not self.api:
            return False
        
        try:
            self.api.cancel_order(order_id)
            return True
            
        except Exception as e:
            logger.error(f"Error canceling Alpaca order: {e}")
            return False


class CoinbaseExchange(ExchangeBase):
    """Coinbase exchange implementation"""
    
    def __init__(self):
        """Initialize the Coinbase exchange"""
        super().__init__('coinbase')
        self.api_url = 'https://api.exchange.coinbase.com'
    
    def get_default_settings(self):
        """Get default settings for Coinbase exchange"""
        return {
            'api_key': os.getenv('COINBASE_API_KEY', ''),
            'api_secret': os.getenv('COINBASE_API_SECRET', ''),
            'passphrase': os.getenv('COINBASE_PASSPHRASE', ''),
            'max_positions': 3,
            'max_position_size': 0.1,  # 10% of account
            'default_timeframe': '1h'
        }
    
    def _get_auth_headers(self, method, request_path, body=''):
        """Get authentication headers for Coinbase API
        
        Args:
            method: HTTP method (GET, POST, etc.)
            request_path: API endpoint path
            body: Request body for POST requests
            
        Returns:
            dict: Authentication headers
        """
        api_key = self.settings.get('api_key')
        api_secret = self.settings.get('api_secret')
        passphrase = self.settings.get('passphrase')
        
        if not all([api_key, api_secret, passphrase]):
            logger.warning("Coinbase API credentials not found")
            return {}
        
        timestamp = str(time.time())
        message = timestamp + method + request_path + (body or '')
        
        # Create signature
        hmac_key = base64.b64decode(api_secret)
        signature = hmac.new(hmac_key, message.encode('utf-8'), hashlib.sha256)
        signature_b64 = base64.b64encode(signature.digest()).decode('utf-8')
        
        return {
            'CB-ACCESS-KEY': api_key,
            'CB-ACCESS-SIGN': signature_b64,
            'CB-ACCESS-TIMESTAMP': timestamp,
            'CB-ACCESS-PASSPHRASE': passphrase,
            'Content-Type': 'application/json'
        }
    
    def _api_request(self, method, endpoint, params=None, data=None):
        """Make an API request to Coinbase
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Query parameters
            data: Request body for POST requests
            
        Returns:
            dict: API response
        """
        url = self.api_url + endpoint
        
        # Prepare request body
        body = json.dumps(data) if data else ''
        
        # Get authentication headers
        headers = self._get_auth_headers(method, endpoint, body)
        
        try:
            if method == 'GET':
                response = requests.get(url, params=params, headers=headers)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers)
            else:
                logger.error(f"Unsupported HTTP method: {method}")
                return {'error': f'Unsupported HTTP method: {method}'}
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Coinbase API error: {response.status_code} - {response.text}")
                return {'error': f'API error: {response.status_code}', 'details': response.text}
            
        except Exception as e:
            logger.error(f"Error making Coinbase API request: {e}")
            return {'error': str(e)}
    
    def get_account(self):
        """Get account information"""
        response = self._api_request('GET', '/accounts')
        
        if 'error' in response:
            return response
        
        # Process accounts
        accounts = []
        for account in response:
            accounts.append({
                'id': account['id'],
                'currency': account['currency'],
                'balance': float(account['balance']),
                'available': float(account['available']),
                'hold': float(account['hold'])
            })
        
        return {
            'accounts': accounts,
            'exchange': 'coinbase'
        }
    
    def get_positions(self):
        """Get current positions"""
        # Coinbase doesn't have a direct positions endpoint
        # We'll use the accounts endpoint to get balances
        response = self._api_request('GET', '/accounts')
        
        if 'error' in response:
            return []
        
        # Filter accounts with non-zero balance
        positions = []
        for account in response:
            balance = float(account['balance'])
            if balance > 0:
                # Get current price for this currency
                currency_pair = account['currency'] + '-USD'
                ticker = self._api_request('GET', f'/products/{currency_pair}/ticker')
                
                current_price = float(ticker['price']) if 'price' in ticker else 0
                
                positions.append({
                    'symbol': account['currency'],
                    'quantity': balance,
                    'current_price': current_price,
                    'market_value': balance * current_price,
                    'exchange': 'coinbase'
                })
        
        return positions
    
    def get_balance(self):
        """Get account balance (in USD)"""
        response = self._api_request('GET', '/accounts')
        
        if 'error' in response:
            return 0.0
        
        # Find USD account
        for account in response:
            if account['currency'] == 'USD':
                return float(account['available'])
        
        return 0.0
    
    def get_market_data(self, symbol, timeframe='1h', limit=100):
        """Get market data"""
        # Convert timeframe to Coinbase format
        timeframe_map = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '1h': 3600,
            '1d': 86400
        }
        
        granularity = timeframe_map.get(timeframe, 3600)
        
        # Ensure symbol is in correct format (BTC-USD)
        if '-' not in symbol:
            symbol = symbol + '-USD'
        
        # Make API request
        endpoint = f'/products/{symbol}/candles'
        params = {
            'granularity': granularity
        }
        
        response = self._api_request('GET', endpoint, params=params)
        
        if 'error' in response:
            return None
        
        # Convert to DataFrame
        # Coinbase returns: [timestamp, low, high, open, close, volume]
        df = pd.DataFrame(response, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Limit results
        if len(df) > limit:
            df = df.tail(limit)
        
        return df
    
    def place_order(self, symbol, side, quantity, order_type='market', price=None):
        """Place an order"""
        # Ensure symbol is in correct format (BTC-USD)
        if '-' not in symbol:
            symbol = symbol + '-USD'
        
        # Prepare order data
        order_data = {
            'product_id': symbol,
            'side': side.lower(),
            'size': str(quantity)
        }
        
        if order_type.lower() == 'limit':
            if price is None:
                return {'error': 'Price is required for limit orders'}
            
            order_data['type'] = 'limit'
            order_data['price'] = str(price)
        else:
            order_data['type'] = 'market'
        
        # Place order
        response = self._api_request('POST', '/orders', data=order_data)
        
        if 'error' in response:
            return response
        
        return {
            'id': response['id'],
            'symbol': response['product_id'],
            'side': response['side'],
            'quantity': float(response['size']),
            'type': response['type'],
            'status': response['status'],
            'created_at': response['created_at'],
            'exchange': 'coinbase'
        }
    
    def get_order(self, order_id):
        """Get order information"""
        response = self._api_request('GET', f'/orders/{order_id}')
        
        if 'error' in response:
            return response
        
        return {
            'id': response['id'],
            'symbol': response['product_id'],
            'side': response['side'],
            'quantity': float(response['size']),
            'filled_quantity': float(response.get('filled_size', 0)),
            'type': response['type'],
            'status': response['status'],
            'created_at': response['created_at'],
            'exchange': 'coinbase'
        }
    
    def cancel_order(self, order_id):
        """Cancel an order"""
        response = self._api_request('DELETE', f'/orders/{order_id}')
        
        if 'error' in response:
            return False
        
        return True


class MultiExchangeManager:
    """Manages multiple exchanges"""
    
    def __init__(self):
        """Initialize the multi-exchange manager"""
        self.exchanges = {}
        self.init_database()
        self.load_exchanges()
        
        logger.info("MultiExchangeManager initialized")
    
    def init_database(self):
        """Initialize database tables"""
        try:
            conn = sqlite3.connect('tradebot.db')
            cursor = conn.cursor()
            
            # Create exchange_orders table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS exchange_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                exchange_id TEXT NOT NULL,
                order_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL,
                type TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(exchange_id, order_id)
            )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Multi-exchange database tables initialized")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def load_exchanges(self):
        """Load and initialize all supported exchanges"""
        # Initialize Alpaca exchange
        self.exchanges['alpaca'] = AlpacaExchange()
        
        # Initialize Coinbase exchange
        self.exchanges['coinbase'] = CoinbaseExchange()
        
        logger.info(f"Loaded {len(self.exchanges)} exchanges")
    
    def get_exchange(self, exchange_id):
        """Get an exchange by ID
        
        Args:
            exchange_id: Exchange ID
            
        Returns:
            ExchangeBase: Exchange instance or None
        """
        return self.exchanges.get(exchange_id)
    
    def get_exchanges(self):
        """Get all exchanges
        
        Returns:
            dict: Exchange instances
        """
        return self.exchanges
    
    def get_enabled_exchanges(self):
        """Get enabled exchanges
        
        Returns:
            dict: Enabled exchange instances
        """
        return {id: exchange for id, exchange in self.exchanges.items() if exchange.is_enabled()}
    
    def get_exchange_info(self):
        """Get information about all exchanges
        
        Returns:
            list: Exchange information
        """
        info = []
        for exchange_id, exchange in self.exchanges.items():
            info.append({
                'id': exchange_id,
                'name': exchange.name,
                'description': exchange.description,
                'enabled': exchange.is_enabled()
            })
        
        return info
    
    def enable_exchange(self, exchange_id):
        """Enable an exchange
        
        Args:
            exchange_id: Exchange ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        exchange = self.get_exchange(exchange_id)
        if exchange:
            exchange.enable()
            return True
        
        return False
    
    def disable_exchange(self, exchange_id):
        """Disable an exchange
        
        Args:
            exchange_id: Exchange ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        exchange = self.get_exchange(exchange_id)
        if exchange:
            exchange.disable()
            return True
        
        return False
    
    def get_all_positions(self):
        """Get positions from all enabled exchanges
        
        Returns:
            list: Positions from all exchanges
        """
        all_positions = []
        
        for exchange_id, exchange in self.get_enabled_exchanges().items():
            positions = exchange.get_positions()
            all_positions.extend(positions)
        
        return all_positions
    
    def get_total_balance(self):
        """Get total balance across all enabled exchanges
        
        Returns:
            float: Total balance
        """
        total = 0.0
        
        for exchange_id, exchange in self.get_enabled_exchanges().items():
            total += exchange.get_balance()
        
        return total
    
    def place_order(self, exchange_id, symbol, side, quantity, order_type='market', price=None):
        """Place an order on a specific exchange
        
        Args:
            exchange_id: Exchange ID
            symbol: Symbol to trade
            side: 'buy' or 'sell'
            quantity: Order quantity
            order_type: Order type (market, limit, etc.)
            price: Price for limit orders
            
        Returns:
            dict: Order information
        """
        exchange = self.get_exchange(exchange_id)
        if not exchange:
            return {'error': f'Exchange {exchange_id} not found'}
        
        if not exchange.is_enabled():
            return {'error': f'Exchange {exchange_id} is not enabled'}
        
        # Place order
        order = exchange.place_order(symbol, side, quantity, order_type, price)
        
        # Record order in database
        if 'error' not in order:
            self._record_order(exchange_id, order)
        
        return order
    
    def _record_order(self, exchange_id, order):
        """Record an order in the database
        
        Args:
            exchange_id: Exchange ID
            order: Order information
        """
        try:
            conn = sqlite3.connect('tradebot.db')
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT OR REPLACE INTO exchange_orders (
                exchange_id, order_id, symbol, side, quantity, price, type, status, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                exchange_id,
                order['id'],
                order['symbol'],
                order['side'],
                order['quantity'],
                order.get('price', 0),
                order['type'],
                order['status'],
                order.get('created_at', datetime.now().isoformat()),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error recording order: {e}")
    
    def get_order(self, exchange_id, order_id):
        """Get order information from a specific exchange
        
        Args:
            exchange_id: Exchange ID
            order_id: Order ID
            
        Returns:
            dict: Order information
        """
        exchange = self.get_exchange(exchange_id)
        if not exchange:
            return {'error': f'Exchange {exchange_id} not found'}
        
        return exchange.get_order(order_id)
    
    def cancel_order(self, exchange_id, order_id):
        """Cancel an order on a specific exchange
        
        Args:
            exchange_id: Exchange ID
            order_id: Order ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        exchange = self.get_exchange(exchange_id)
        if not exchange:
            return False
        
        return exchange.cancel_order(order_id)
    
    def get_market_data(self, symbol, exchange_id=None, timeframe='1d', limit=100):
        """Get market data from exchanges
        
        Args:
            symbol: Symbol to get data for
            exchange_id: Optional exchange ID (if None, use first enabled exchange)
            timeframe: Timeframe (e.g., '1m', '1h', '1d')
            limit: Maximum number of data points
            
        Returns:
            DataFrame: Market data
        """
        if exchange_id:
            # Use specific exchange
            exchange = self.get_exchange(exchange_id)
            if not exchange or not exchange.is_enabled():
                return None
            
            return exchange.get_market_data(symbol, timeframe, limit)
        else:
            # Use first enabled exchange
            for exchange_id, exchange in self.get_enabled_exchanges().items():
                data = exchange.get_market_data(symbol, timeframe, limit)
                if data is not None:
                    return data
            
            return None
    
    def allocate_funds(self, total_amount, allocation_strategy='equal'):
        """Allocate funds across enabled exchanges
        
        Args:
            total_amount: Total amount to allocate
            allocation_strategy: Strategy for allocation (equal, weighted, etc.)
            
        Returns:
            dict: Allocation per exchange
        """
        enabled_exchanges = self.get_enabled_exchanges()
        if not enabled_exchanges:
            return {}
        
        if allocation_strategy == 'equal':
            # Equal allocation
            amount_per_exchange = total_amount / len(enabled_exchanges)
            return {id: amount_per_exchange for id in enabled_exchanges}
        
        # Default to equal allocation
        amount_per_exchange = total_amount / len(enabled_exchanges)
        return {id: amount_per_exchange for id in enabled_exchanges}


# Test the multi-exchange component if run directly
if __name__ == "__main__":
    manager = MultiExchangeManager()
    
    # Get exchange info
    exchanges = manager.get_exchange_info()
    
    print("Available Exchanges:")
    for exchange in exchanges:
        print(f"- {exchange['name']} ({exchange['id']}): {'Enabled' if exchange['enabled'] else 'Disabled'}")
    
    # Enable Alpaca exchange
    manager.enable_exchange('alpaca')
    
    # Get positions
    positions = manager.get_all_positions()
    
    print("\nCurrent Positions:")
    for position in positions:
        print(f"- {position['symbol']}: {position['quantity']} shares at ${position['avg_entry_price']:.2f} (${position['market_value']:.2f})")
    
    # Get total balance
    total_balance = manager.get_total_balance()
    print(f"\nTotal Balance: ${total_balance:.2f}")
    
    # Get market data
    data = manager.get_market_data('AAPL')
    if data is not None:
        print(f"\nAAPL Market Data (last 5 rows):")
        print(data.tail(5))
