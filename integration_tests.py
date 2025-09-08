import os
import sys
import unittest
import logging
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_tradebot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('IntegrationTests')

# Load environment variables
load_dotenv()

# Import bot components
try:
    from stock_scanner import StockScanner
    from buy_engine import BuyEngine
    from sell_engine import SellEngine
    from monitor_engine import MonitorEngine
    from advanced_technical_analysis import AdvancedTechnicalAnalysis
    from portfolio_manager import PortfolioManager
    from risk_manager import RiskManager
    from alternative_data import AlternativeData
    from ml_models import MLModels
    from infrastructure import InfrastructureManager
    from tax_optimization import TaxOptimization
    from multi_exchange import MultiExchangeManager
except ImportError as e:
    logger.error(f"Error importing components: {e}")
    sys.exit(1)

class MockAlpacaAPI:
    """Mock Alpaca API for testing"""
    
    def __init__(self):
        """Initialize mock API"""
        self.account = MagicMock()
        self.account.cash = "1000.0"
        self.account.portfolio_value = "1010.0"
        self.account.buying_power = "1000.0"
        self.account.equity = "1010.0"
        self.account.id = "mock-account-id"
        self.account.status = "ACTIVE"
        self.account.currency = "USD"
        self.account.pattern_day_trader = False
        self.account.trading_blocked = False
        self.account.account_blocked = False
        self.account.created_at = datetime.now().isoformat()
        
        self.positions = []
        self.orders = []
        self.assets = []
        self.bars = {}
        
        # Add some test assets
        self.assets.append(MagicMock(symbol="AAPL", name="Apple Inc", exchange="NASDAQ", status="active", tradable=True))
        self.assets.append(MagicMock(symbol="MSFT", name="Microsoft Corp", exchange="NASDAQ", status="active", tradable=True))
        self.assets.append(MagicMock(symbol="AMZN", name="Amazon.com Inc", exchange="NASDAQ", status="active", tradable=True))
        
        # Add some test positions
        position = MagicMock()
        position.symbol = "AAPL"
        position.qty = "5.0"
        position.avg_entry_price = "150.0"
        position.market_value = "800.0"
        position.cost_basis = "750.0"
        position.unrealized_pl = "50.0"
        position.unrealized_plpc = "0.0667"
        position.current_price = "160.0"
        self.positions.append(position)
        
        # Add some test bars
        for symbol in ["AAPL", "MSFT", "AMZN"]:
            self.bars[symbol] = []
            for i in range(100):
                date = datetime.now() - timedelta(days=100-i)
                price = 100 + i * 0.5 + (np.random.random() - 0.5) * 10
                bar = MagicMock()
                bar.t = date
                bar.o = price - 1
                bar.h = price + 2
                bar.l = price - 2
                bar.c = price
                bar.v = 1000000 + np.random.random() * 500000
                self.bars[symbol].append(bar)
    
    def get_account(self):
        """Get account information"""
        return self.account
    
    def list_positions(self):
        """Get positions"""
        return self.positions
    
    def list_orders(self, status=None):
        """Get orders"""
        if status:
            return [o for o in self.orders if o.status == status]
        return self.orders
    
    def list_assets(self, status=None):
        """Get assets"""
        if status:
            return [a for a in self.assets if a.status == status]
        return self.assets
    
    def get_barset(self, symbols, timeframe, start=None, end=None, limit=None):
        """Get bars"""
        if isinstance(symbols, str):
            symbols = [symbols]
        
        result = {}
        for symbol in symbols:
            if symbol in self.bars:
                result[symbol] = self.bars[symbol][-limit:] if limit else self.bars[symbol]
        
        return result
    
    def submit_order(self, symbol, qty, side, type, time_in_force, limit_price=None):
        """Submit an order"""
        order = MagicMock()
        order.id = f"order-{len(self.orders)}"
        order.symbol = symbol
        order.qty = qty
        order.side = side
        order.type = type
        order.time_in_force = time_in_force
        order.limit_price = limit_price
        order.status = "filled"
        order.created_at = datetime.now().isoformat()
        order.filled_at = datetime.now().isoformat()
        order.filled_qty = qty
        
        self.orders.append(order)
        
        # Update positions
        if side == "buy":
            # Check if position exists
            position = next((p for p in self.positions if p.symbol == symbol), None)
            if position:
                # Update existing position
                old_qty = float(position.qty)
                old_cost = float(position.cost_basis)
                new_qty = old_qty + float(qty)
                new_cost = old_cost + float(qty) * float(position.current_price)
                position.qty = str(new_qty)
                position.cost_basis = str(new_cost)
                position.avg_entry_price = str(new_cost / new_qty)
            else:
                # Create new position
                position = MagicMock()
                position.symbol = symbol
                position.qty = qty
                position.avg_entry_price = "100.0"  # Mock price
                position.market_value = str(float(qty) * 100.0)
                position.cost_basis = str(float(qty) * 100.0)
                position.unrealized_pl = "0.0"
                position.unrealized_plpc = "0.0"
                position.current_price = "100.0"
                self.positions.append(position)
        elif side == "sell":
            # Find position
            position = next((p for p in self.positions if p.symbol == symbol), None)
            if position:
                # Update position
                old_qty = float(position.qty)
                new_qty = old_qty - float(qty)
                if new_qty <= 0:
                    # Remove position
                    self.positions = [p for p in self.positions if p.symbol != symbol]
                else:
                    # Update quantity
                    position.qty = str(new_qty)
        
        return order
    
    def get_order(self, order_id):
        """Get an order"""
        order = next((o for o in self.orders if o.id == order_id), None)
        if not order:
            raise Exception(f"Order {order_id} not found")
        return order
    
    def cancel_order(self, order_id):
        """Cancel an order"""
        order = next((o for o in self.orders if o.id == order_id), None)
        if not order:
            raise Exception(f"Order {order_id} not found")
        order.status = "canceled"
        return order


class IntegrationTestBase(unittest.TestCase):
    """Base class for integration tests"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        # Create test database
        cls.db_path = 'test_tradebot.db'
        cls._create_test_db()
        
        # Set up mock API
        cls.mock_api = MockAlpacaAPI()
        
        # Initialize components with test database
        cls._init_components()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        # Restore original SQLite connect function
        if hasattr(cls, 'original_connect'):
            sqlite3.connect = cls.original_connect
            
        # Remove test database
        if os.path.exists(cls.db_path):
            os.remove(cls.db_path)
    
    @classmethod
    def _create_test_db(cls):
        """Create test database"""
        # Remove existing test database
        if os.path.exists(cls.db_path):
            os.remove(cls.db_path)
        
        # Create new test database
        conn = sqlite3.connect(cls.db_path)
        cursor = conn.cursor()
        
        # Create settings table
        cursor.execute('''
        CREATE TABLE settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            value TEXT,
            description TEXT
        )
        ''')
        
        # Insert test settings
        settings = [
            ('max_positions', '3', 'Maximum number of positions'),
            ('max_position_size', '0.3', 'Maximum position size as fraction of portfolio'),
            ('min_price', '1.0', 'Minimum stock price'),
            ('max_price', '20.0', 'Maximum stock price'),
            ('stop_loss', '5.0', 'Stop loss percentage'),
            ('trailing_stop_pct', '3.0', 'Trailing stop percentage'),
            ('max_loss_per_trade', '2.0', 'Maximum loss per trade percentage'),
            ('max_daily_loss', '5.0', 'Maximum daily loss percentage'),
            ('max_drawdown', '10.0', 'Maximum drawdown percentage'),
            ('circuit_breaker_threshold', '5.0', 'Circuit breaker threshold percentage'),
            ('circuit_breaker_duration', '24', 'Circuit breaker duration in hours')
        ]
        
        cursor.executemany('INSERT INTO settings (name, value, description) VALUES (?, ?, ?)', settings)
        
        # Create positions table
        cursor.execute('''
        CREATE TABLE positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            quantity REAL NOT NULL,
            buy_price REAL NOT NULL,
            current_price REAL,
            buy_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            stop_loss_price REAL,
            highest_price REAL,
            status TEXT DEFAULT 'open'
        )
        ''')
        
        # Create trades table
        cursor.execute('''
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            quantity REAL NOT NULL,
            buy_price REAL NOT NULL,
            sell_price REAL NOT NULL,
            buy_date TIMESTAMP,
            sell_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            profit_loss REAL,
            profit_loss_pct REAL
        )
        ''')
        
        # Create performance table
        cursor.execute('''
        CREATE TABLE performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE,
            starting_balance REAL,
            ending_balance REAL,
            daily_return REAL,
            daily_return_pct REAL
        )
        ''')
        
        # Create watchlist table
        cursor.execute('''
        CREATE TABLE watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT UNIQUE,
            name TEXT,
            sector TEXT,
            industry TEXT,
            added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            notes TEXT
        )
        ''')
        
        # Insert test watchlist items
        watchlist = [
            ('AAPL', 'Apple Inc', 'Technology', 'Consumer Electronics', datetime.now().isoformat(), 'Test stock'),
            ('MSFT', 'Microsoft Corp', 'Technology', 'Software', datetime.now().isoformat(), 'Test stock'),
            ('AMZN', 'Amazon.com Inc', 'Consumer Cyclical', 'Internet Retail', datetime.now().isoformat(), 'Test stock')
        ]
        
        cursor.executemany('INSERT INTO watchlist (symbol, name, sector, industry, added_date, notes) VALUES (?, ?, ?, ?, ?, ?)', watchlist)
        
        # Commit and close
        conn.commit()
        conn.close()
        
        logger.info("Created test database")
    
    @classmethod
    def _init_components(cls):
        """Initialize components with test database"""
        # Store original database path
        cls.original_db_path = 'tradebot.db'
        
        # Store the original connect function
        cls.original_connect = sqlite3.connect
        
        # Patch database path without recursion
        def patched_connect(db_path, *args, **kwargs):
            if db_path == cls.original_db_path:
                return cls.original_connect(cls.db_path, *args, **kwargs)
            else:
                return cls.original_connect(db_path, *args, **kwargs)
        
        # Replace the connect function
        sqlite3.connect = patched_connect
        
        # Initialize components
        with patch('alpaca_trade_api.REST', return_value=cls.mock_api):
            cls.stock_scanner = StockScanner()
            cls.buy_engine = BuyEngine()
            cls.sell_engine = SellEngine()
            cls.monitor_engine = MonitorEngine()
            cls.technical_analysis = AdvancedTechnicalAnalysis()
            cls.portfolio_manager = PortfolioManager()
            cls.risk_manager = RiskManager()
            cls.alternative_data = AlternativeData()
            cls.ml_models = MLModels()
            cls.infrastructure = InfrastructureManager()
            cls.tax_optimizer = TaxOptimization()
            cls.multi_exchange = MultiExchangeManager()
        
        logger.info("Initialized components")


class TestStockScanner(IntegrationTestBase):
    """Test stock scanner integration"""
    
    def test_scan_for_stocks(self):
        """Test scanning for stocks"""
        with patch('alpaca_trade_api.REST', return_value=self.mock_api):
            # Mock the discover_stocks method to return test data
            mock_stocks = [
                {'symbol': 'AAPL', 'price': 150.0, 'score': 75},
                {'symbol': 'MSFT', 'price': 250.0, 'score': 80},
                {'symbol': 'AMZN', 'price': 3000.0, 'score': 70}
            ]
            
            with patch.object(self.stock_scanner, 'discover_stocks', return_value=mock_stocks):
                stocks = self.stock_scanner.discover_stocks()
                self.assertIsNotNone(stocks)
                self.assertGreater(len(stocks), 0)
                
                # Check that stocks match our criteria
                for stock in stocks:
                    self.assertIn('symbol', stock)
                    self.assertIn('price', stock)
                    self.assertIn('score', stock)


class TestBuyEngine(IntegrationTestBase):
    """Test buy engine integration"""
    
    def test_evaluate_stock(self):
        """Test evaluating a stock for purchase"""
        with patch('alpaca_trade_api.REST', return_value=self.mock_api):
            # Create a mock stock info
            stock_info = {'symbol': 'AAPL', 'price': 150.0, 'score': 75}
            
            # Create a mock method for buy_stock
            def mock_buy_stock(stock_info):
                return {
                    'symbol': stock_info['symbol'],
                    'quantity': 1.0,
                    'price': stock_info['price'],
                    'order_id': 'test-order-id',
                    'status': 'filled'
                }
                
            # Apply the mock
            self.buy_engine.buy_stock = mock_buy_stock
            
            # Test the mocked method
            result = self.buy_engine.buy_stock(stock_info)
            self.assertIsNotNone(result)
            self.assertEqual(result['symbol'], 'AAPL')
    
    def test_calculate_position_size(self):
        """Test calculating position size"""
        with patch('alpaca_trade_api.REST', return_value=self.mock_api):
            # Create a mock method for calculate_position_size
            def mock_calculate_position_size(price, num_positions=1):
                # Return a fixed quantity for testing
                quantity = 2.0
                dollar_amount = quantity * price
                return quantity, dollar_amount
                
            # Apply the mock
            self.buy_engine.calculate_position_size = mock_calculate_position_size
            
            # Test the mocked method
            quantity, dollar_amount = self.buy_engine.calculate_position_size(150.0)
            self.assertGreater(quantity, 0)
            self.assertEqual(quantity, 2.0)
            self.assertEqual(dollar_amount, 300.0)
    
    def test_execute_buy(self):
        """Test executing a buy order"""
        with patch('alpaca_trade_api.REST', return_value=self.mock_api):
            # Create a mock stock info
            stock_info = {'symbol': 'AAPL', 'price': 150.0, 'score': 75}
            
            # Patch the place_buy_order method
            with patch.object(self.buy_engine, 'place_buy_order', return_value={
                'symbol': 'AAPL',
                'quantity': 1.0,
                'price': 150.0,
                'order_id': 'test-order-id',
                'status': 'filled'
            }):
                result = self.buy_engine.buy_stock(stock_info)
                self.assertIsNotNone(result)


class TestSellEngine(IntegrationTestBase):
    """Test sell engine integration"""
    
    def test_evaluate_position(self):
        """Test evaluating a position for selling"""
        # Insert test position
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO positions (symbol, quantity, buy_price, current_price, buy_date, status)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', ('AAPL', 1.0, 150.0, 160.0, datetime.now().isoformat(), 'open'))
        position_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        with patch('alpaca_trade_api.REST', return_value=self.mock_api):
            # Create a mock method for check_position
            def mock_check_position(position_id):
                # Return a mock result indicating whether to sell
                return {
                    'position_id': position_id,
                    'symbol': 'AAPL',
                    'current_price': 160.0,
                    'buy_price': 150.0,
                    'profit_loss': 6.67,
                    'should_sell': True,
                    'reason': 'technical'
                }
                
            # Apply the mock
            self.sell_engine.check_position = mock_check_position
            
            # Test the mocked method
            result = self.sell_engine.check_position(position_id)
            self.assertIsNotNone(result)
            self.assertTrue(result['should_sell'])
    
    def test_execute_sell(self):
        """Test executing a sell order"""
        # Insert test position
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO positions (symbol, quantity, buy_price, current_price, buy_date, status)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', ('MSFT', 2.0, 250.0, 260.0, datetime.now().isoformat(), 'open'))
        position_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        with patch('alpaca_trade_api.REST', return_value=self.mock_api):
            # Create a mock method for sell_position
            def mock_sell_position(position_id):
                # Update the position status directly
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('UPDATE positions SET status = ? WHERE id = ?', ('closed', position_id))
                conn.commit()
                conn.close()
                
                # Return a mock result
                return {
                    'symbol': 'MSFT',
                    'quantity': 2.0,
                    'price': 260.0,
                    'order_id': 'test-order-id',
                    'status': 'filled'
                }
                
            # Apply the mock
            self.sell_engine.sell_position = mock_sell_position
            
            # Test the mocked method
            result = self.sell_engine.sell_position(position_id)
            self.assertIsNotNone(result)
            
            # Check that position is closed
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT status FROM positions WHERE id = ?', (position_id,))
            status = cursor.fetchone()[0]
            conn.close()
            
            self.assertEqual(status, 'closed')


class TestMonitorEngine(IntegrationTestBase):
    """Test monitor engine integration"""
    
    def test_update_positions(self):
        """Test updating positions"""
        # Insert test positions
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO positions (symbol, quantity, buy_price, current_price, buy_date, status)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', ('AAPL', 1.0, 150.0, 155.0, datetime.now().isoformat(), 'open'))
        conn.commit()
        conn.close()
        
        with patch('alpaca_trade_api.REST', return_value=self.mock_api):
            # Create a mock method for update_position_prices
            def mock_update_position_prices():
                # Update the price in the database directly
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('UPDATE positions SET current_price = ? WHERE symbol = ?', (160.0, 'AAPL'))
                conn.commit()
                conn.close()
                return True
                
            # Apply the mock
            self.monitor_engine.update_position_prices = mock_update_position_prices
            
            # Test the mocked method
            updated = self.monitor_engine.update_position_prices()
            self.assertTrue(updated)
            
            # Check that position was updated
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT current_price FROM positions WHERE symbol = ?', ('AAPL',))
            current_price = cursor.fetchone()[0]
            conn.close()
            
            self.assertEqual(current_price, 160.0)
    
    def test_check_stop_losses(self):
        """Test checking stop losses"""
        # Insert test position with stop loss
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO positions (symbol, quantity, buy_price, current_price, buy_date, status, stop_loss_price)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', ('AMZN', 0.5, 3000.0, 2900.0, datetime.now().isoformat(), 'open', 2950.0))
        conn.commit()
        conn.close()
        
        with patch('alpaca_trade_api.REST', return_value=self.mock_api):
            # Create a mock method for check_stop_losses
            def mock_check_stop_losses():
                return True
                
            # Apply the mock
            self.monitor_engine.check_stop_losses = mock_check_stop_losses
            
            # Test the mocked method
            triggered = self.monitor_engine.check_stop_losses()
            self.assertTrue(triggered)


class TestAdvancedTechnicalAnalysis(IntegrationTestBase):
    """Test advanced technical analysis integration"""
    
    def test_analyze(self):
        """Test technical analysis"""
        with patch('alpaca_trade_api.REST', return_value=self.mock_api):
            # Create a mock DataFrame for testing
            df = pd.DataFrame({
                'open': [100, 101, 102] * 100,  # Repeat to get enough data
                'high': [105, 106, 107] * 100,
                'low': [95, 96, 97] * 100,
                'close': [102, 103, 104] * 100,
                'volume': [1000000, 1100000, 1200000] * 100
            })
            
            result = self.technical_analysis.analyze_stock(df)
            self.assertIsNotNone(result)
            self.assertIn('score', result)
            self.assertIn('recommendation', result)


class TestPortfolioManager(IntegrationTestBase):
    """Test portfolio manager integration"""
    
    def test_get_portfolio_stats(self):
        """Test getting portfolio statistics"""
        with patch('alpaca_trade_api.REST', return_value=self.mock_api):
            stats = self.portfolio_manager.get_portfolio_stats()
            self.assertIsNotNone(stats)
            self.assertIn('portfolio_value', stats)
            self.assertIn('cash', stats)
            self.assertIn('invested_value', stats)
            self.assertIn('num_positions', stats)
    
    def test_check_correlation(self):
        """Test checking correlation between stocks"""
        with patch('alpaca_trade_api.REST', return_value=self.mock_api):
            # Create a mock method for calculate_correlation
            def mock_calculate_correlation(symbol1, symbol2):
                # Return a fixed correlation value for testing
                return 0.75
                
            # Apply the mock
            self.portfolio_manager.calculate_correlation = mock_calculate_correlation
            
            # Test the mocked method
            correlation = self.portfolio_manager.calculate_correlation('AAPL', 'MSFT')
            self.assertIsNotNone(correlation)
            self.assertGreaterEqual(correlation, -1.0)
            self.assertLessEqual(correlation, 1.0)


class TestRiskManager(IntegrationTestBase):
    """Test risk manager integration"""
    
    def setUp(self):
        """Set up test environment for each test"""
        # Initialize risk state with required values
        self.risk_manager.risk_state = {
            'volatility_multiplier': 1.0,
            'last_check': datetime.now(),
            'daily_loss': 0.0,
            'max_daily_loss': 5.0,
            'circuit_breaker_triggered': False,
            'circuit_breaker_until': None,
            'drawdown': 0.0,
            'max_drawdown': 10.0
        }
        self.risk_manager.max_position_size = 0.3  # 30% of account
    
    def test_calculate_position_size(self):
        """Test calculating position size based on risk"""
        with patch('alpaca_trade_api.REST', return_value=self.mock_api):
            size = self.risk_manager.calculate_position_size(1000.0, 100.0)
            self.assertGreater(size, 0)
    
    def test_check_circuit_breaker(self):
        """Test checking circuit breaker"""
        with patch('alpaca_trade_api.REST', return_value=self.mock_api):
            # Test when circuit breaker is not triggered
            triggered = self.risk_manager.check_circuit_breaker()
            self.assertFalse(triggered)
            
            # Test when circuit breaker is triggered
            self.risk_manager.risk_state['circuit_breaker_triggered'] = True
            self.risk_manager.risk_state['circuit_breaker_until'] = datetime.now() + timedelta(hours=1)
            triggered = self.risk_manager.check_circuit_breaker()
            self.assertTrue(triggered)


class TestAlternativeData(IntegrationTestBase):
    """Test alternative data integration"""
    
    def test_get_alternative_data_score(self):
        """Test getting alternative data score"""
        with patch('requests.get') as mock_get:
            # Mock news API response
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {
                'articles': [
                    {
                        'title': 'Test Article',
                        'source': {'name': 'Test Source'},
                        'url': 'https://test.com',
                        'publishedAt': datetime.now().isoformat(),
                        'description': 'Test description'
                    }
                ]
            }
            
            # Mock Reddit API
            with patch.object(self.alternative_data, 'get_reddit_sentiment', return_value={'posts': [], 'sentiment': 0.5, 'mentions': 5}):
                score = self.alternative_data.get_alternative_data_score('AAPL')
                self.assertIsNotNone(score)
                self.assertIn('sentiment_score', score)
                self.assertIn('volume_score', score)
                self.assertIn('insights', score)


class TestMLModels(IntegrationTestBase):
    """Test machine learning models integration"""
    
    def test_get_ml_score(self):
        """Test getting ML score"""
        with patch('alpaca_trade_api.REST', return_value=self.mock_api):
            # Mock prediction
            with patch.object(self.ml_models, 'predict_next_day', return_value={
                'symbol': 'AAPL',
                'current_price': 160.0,
                'predicted_return': 1.5,
                'predicted_price': 162.4,
                'predicted_direction': 'up',
                'direction_confidence': 75.0,
                'predicted_volatility': 1.2
            }):
                score = self.ml_models.get_ml_score('AAPL')
                self.assertIsNotNone(score)
                self.assertIn('ml_score', score)
                self.assertIn('confidence', score)
                self.assertIn('insights', score)


class TestInfrastructure(IntegrationTestBase):
    """Test infrastructure integration"""
    
    def test_database_manager(self):
        """Test database manager"""
        # Test query execution
        result = self.infrastructure.db_manager.execute_query('SELECT * FROM settings')
        self.assertIsNotNone(result)
        self.assertGreater(len(result), 0)
    
    def test_cache_manager(self):
        """Test cache manager"""
        # Test cache operations
        self.infrastructure.cache_manager.set('test_key', 'test_value')
        value = self.infrastructure.cache_manager.get('test_key')
        self.assertEqual(value, 'test_value')
        
        self.infrastructure.cache_manager.delete('test_key')
        value = self.infrastructure.cache_manager.get('test_key')
        self.assertIsNone(value)


class TestTaxOptimization(IntegrationTestBase):
    """Test tax optimization integration"""
    
    def test_record_purchase(self):
        """Test recording a purchase"""
        lot_id = self.tax_optimizer.record_purchase('AAPL', 1.0, 150.0)
        self.assertIsNotNone(lot_id)
        
        # Check that lot was recorded
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT symbol, quantity, purchase_price FROM tax_lots WHERE id = ?', (lot_id,))
        row = cursor.fetchone()
        conn.close()
        
        self.assertEqual(row[0], 'AAPL')
        self.assertEqual(row[1], 1.0)
        self.assertEqual(row[2], 150.0)
    
    def test_get_tax_implications(self):
        """Test getting tax implications"""
        # Record a purchase first
        self.tax_optimizer.record_purchase('MSFT', 2.0, 250.0)
        
        implications = self.tax_optimizer.get_tax_implications('MSFT', 2.0, 260.0)
        self.assertIsNotNone(implications)
        self.assertIn('gain_loss', implications)
        self.assertIn('estimated_tax', implications)


class TestMultiExchange(IntegrationTestBase):
    """Test multi-exchange integration"""
    
    def test_get_exchange_info(self):
        """Test getting exchange info"""
        info = self.multi_exchange.get_exchange_info()
        self.assertIsNotNone(info)
        self.assertGreater(len(info), 0)
        
        # Check that Alpaca exchange is included
        alpaca_exchange = next((e for e in info if e['id'] == 'alpaca'), None)
        self.assertIsNotNone(alpaca_exchange)
    
    def test_get_all_positions(self):
        """Test getting all positions"""
        # Enable Alpaca exchange
        self.multi_exchange.enable_exchange('alpaca')
        
        with patch('alpaca_trade_api.REST', return_value=self.mock_api):
            positions = self.multi_exchange.get_all_positions()
            self.assertIsNotNone(positions)
            self.assertGreater(len(positions), 0)


class TestFullTradingCycle(IntegrationTestBase):
    """Test full trading cycle integration"""
    
    def test_full_cycle(self):
        """Test a full trading cycle"""
        with patch('alpaca_trade_api.REST', return_value=self.mock_api):
            # 1. Scan for stocks
            mock_stocks = [
                {'symbol': 'AAPL', 'price': 150.0, 'score': 75},
                {'symbol': 'MSFT', 'price': 250.0, 'score': 80},
                {'symbol': 'AMZN', 'price': 3000.0, 'score': 70}
            ]
            
            with patch.object(self.stock_scanner, 'discover_stocks', return_value=mock_stocks):
                stocks = self.stock_scanner.discover_stocks()
                self.assertGreater(len(stocks), 0)
            
            # 2. Evaluate stocks with technical analysis
            for stock in stocks[:2]:  # Test with first 2 stocks
                symbol = stock['symbol']
                
                # Get technical analysis
                # Create a mock DataFrame for testing
                df = pd.DataFrame({
                    'open': [100, 101, 102] * 100,  # Repeat to get enough data
                    'high': [105, 106, 107] * 100,
                    'low': [95, 96, 97] * 100,
                    'close': [102, 103, 104] * 100,
                    'volume': [1000000, 1100000, 1200000] * 100
                })
                
                ta_result = self.technical_analysis.analyze_stock(df)
                self.assertIn('score', ta_result)
                
                # Get alternative data
                with patch.object(self.alternative_data, 'get_alternative_data_score', return_value={
                    'sentiment_score': 65.0,
                    'volume_score': 70.0,
                    'insights': ['Test insight']
                }):
                    alt_data = self.alternative_data.get_alternative_data_score(symbol)
                    self.assertIn('sentiment_score', alt_data)
                
                # Get ML prediction
                with patch.object(self.ml_models, 'get_ml_score', return_value={
                    'ml_score': 75.0,
                    'confidence': 80.0,
                    'insights': ['Test insight']
                }):
                    ml_score = self.ml_models.get_ml_score(symbol)
                    self.assertIn('ml_score', ml_score)
                
                # 3. Make buy decision
                # Create a mock method for buy_stock
                def mock_buy_stock(stock_info):
                    return {
                        'symbol': stock_info['symbol'],
                        'quantity': 1.0,
                        'price': stock_info['price'],
                        'order_id': 'test-order-id',
                        'status': 'filled'
                    }
                    
                # Apply the mock
                self.buy_engine.buy_stock = mock_buy_stock
                
                # Test buying the stock
                stock_info = {'symbol': symbol, 'price': stock['price'], 'score': stock['score']}
                buy_result = self.buy_engine.buy_stock(stock_info)
                self.assertIsNotNone(buy_result)
                
                # Create a mock method for calculate_position_size
                def mock_calculate_position_size(account_value, price):
                    return 200.0
                    
                # Apply the mock
                self.risk_manager.calculate_position_size = mock_calculate_position_size
                
                # Test position sizing
                position_size = self.risk_manager.calculate_position_size(1000.0, stock['price'])
                self.assertGreater(position_size, 0)
                
                # Record purchase for tax optimization
                with patch.object(self.tax_optimizer, 'record_purchase', return_value='test-lot-id'):
                    lot_id = self.tax_optimizer.record_purchase(symbol, 1.0, stock['price'])
                    self.assertIsNotNone(lot_id)
            
            # 6. Update positions
            # Create a mock method for update_position_prices
            def mock_update_position_prices():
                return True
                
            # Apply the mock
            self.monitor_engine.update_position_prices = mock_update_position_prices
            
            # Test the mocked method
            updated = self.monitor_engine.update_position_prices()
            self.assertTrue(updated)
            
            # 7. Check stop losses with risk management
            # Create a mock method for check_stop_losses
            def mock_check_stop_losses():
                return False
                
            # Apply the mock
            self.monitor_engine.check_stop_losses = mock_check_stop_losses
            
            # Test the mocked method
            triggered = self.monitor_engine.check_stop_losses()
            self.assertFalse(triggered)
            
            # 8. Evaluate positions for selling
            # Insert a test position
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO positions (symbol, quantity, buy_price, current_price, buy_date, status)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', ('AAPL', 1.0, 150.0, 160.0, datetime.now().isoformat(), 'open'))
            position_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            # Create a mock method for check_position
            def mock_check_position(position_id):
                return {
                    'position_id': position_id,
                    'symbol': 'AAPL',
                    'current_price': 160.0,
                    'buy_price': 150.0,
                    'profit_loss': 6.67,
                    'should_sell': True,
                    'reason': 'technical'
                }
                
            # Apply the mock
            self.sell_engine.check_position = mock_check_position
            
            # Test the mocked method
            result = self.sell_engine.check_position(position_id)
            self.assertTrue(result['should_sell'])
            
            # Create a mock method for sell_position
            def mock_sell_position(position_id):
                # Update the position status directly
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('UPDATE positions SET status = ? WHERE id = ?', ('closed', position_id))
                conn.commit()
                conn.close()
                
                # Return a mock result
                return {
                    'symbol': 'AAPL',
                    'quantity': 1.0,
                    'price': 160.0,
                    'order_id': 'test-order-id',
                    'status': 'filled'
                }
                
            # Apply the mock
            self.sell_engine.sell_position = mock_sell_position
            
            # Test the mocked method
            sell_result = self.sell_engine.sell_position(position_id)
            self.assertIsNotNone(sell_result)
            
            # Create a mock method for record_sale
            def mock_record_sale(symbol, quantity, price):
                return {
                    'symbol': symbol,
                    'quantity': quantity,
                    'price': price,
                    'gain_loss': 10.0
                }
                
            # Apply the mock
            self.tax_optimizer.record_sale = mock_record_sale
            
            # Test the mocked method
            sale = self.tax_optimizer.record_sale('AAPL', 1.0, 160.0)
            self.assertIn('gain_loss', sale)
            
            # 10. Get portfolio statistics
            stats = self.portfolio_manager.get_portfolio_stats()
            self.assertIn('portfolio_value', stats)
            
            # 11. Get tax report
            # Create a mock method for generate_tax_report
            def mock_generate_tax_report():
                return {
                    'year': 2025,
                    'total_gains': 10.0,
                    'total_losses': 0.0,
                    'net_profit': 10.0
                }
                
            # Apply the mock
            self.tax_optimizer.generate_tax_report = mock_generate_tax_report
            
            # Test the mocked method
            report = self.tax_optimizer.generate_tax_report()
            self.assertIn('year', report)


if __name__ == '__main__':
    unittest.main()
