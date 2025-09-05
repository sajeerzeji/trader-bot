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
        
        # Patch database path
        sqlite3.connect = lambda db_path, *args, **kwargs: sqlite3.connect(cls.db_path if db_path == cls.original_db_path else db_path, *args, **kwargs)
        
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
            stocks = self.stock_scanner.scan_for_stocks()
            self.assertIsNotNone(stocks)
            self.assertGreater(len(stocks), 0)
            
            # Check that stocks match our criteria
            for stock in stocks:
                self.assertIn('symbol', stock)
                self.assertIn('price', stock)
                self.assertGreaterEqual(stock['price'], 1.0)
                self.assertLessEqual(stock['price'], 20.0)


class TestBuyEngine(IntegrationTestBase):
    """Test buy engine integration"""
    
    def test_evaluate_stock(self):
        """Test evaluating a stock for purchase"""
        with patch('alpaca_trade_api.REST', return_value=self.mock_api):
            result = self.buy_engine.evaluate_stock('AAPL')
            self.assertIsNotNone(result)
            self.assertIn('symbol', result)
            self.assertIn('score', result)
            self.assertIn('buy_decision', result)
    
    def test_calculate_position_size(self):
        """Test calculating position size"""
        with patch('alpaca_trade_api.REST', return_value=self.mock_api):
            size = self.buy_engine.calculate_position_size('AAPL', 150.0)
            self.assertGreater(size, 0)
            
            # Test with portfolio manager integration
            with patch.object(self.portfolio_manager, 'get_available_cash', return_value=1000.0):
                with patch.object(self.portfolio_manager, 'calculate_position_size', return_value=300.0):
                    size = self.buy_engine.calculate_position_size('AAPL', 150.0)
                    self.assertEqual(size, 300.0)
    
    def test_execute_buy(self):
        """Test executing a buy order"""
        with patch('alpaca_trade_api.REST', return_value=self.mock_api):
            result = self.buy_engine.execute_buy('AAPL', 1.0, 150.0)
            self.assertIsNotNone(result)
            self.assertTrue(result['success'])
            self.assertIn('order_id', result)


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
            # Test with technical analysis integration
            with patch.object(self.technical_analysis, 'analyze', return_value={'score': 60, 'recommendation': 'sell'}):
                result = self.sell_engine.evaluate_position(position_id)
                self.assertIsNotNone(result)
                self.assertIn('symbol', result)
                self.assertIn('score', result)
                self.assertIn('sell_decision', result)
    
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
            result = self.sell_engine.execute_sell(position_id)
            self.assertIsNotNone(result)
            self.assertTrue(result['success'])
            self.assertIn('order_id', result)
            
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
            updated = self.monitor_engine.update_positions()
            self.assertTrue(updated)
            
            # Check that position was updated
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT current_price FROM positions WHERE symbol = ?', ('AAPL',))
            current_price = cursor.fetchone()[0]
            conn.close()
            
            self.assertNotEqual(current_price, 155.0)  # Should be updated to mock API price
    
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
            # Test with risk manager integration
            with patch.object(self.risk_manager, 'check_stop_loss', return_value=(True, 'stop_loss')):
                with patch.object(self.sell_engine, 'execute_sell', return_value={'success': True}):
                    triggered = self.monitor_engine.check_stop_losses()
                    self.assertTrue(triggered)


class TestAdvancedTechnicalAnalysis(IntegrationTestBase):
    """Test advanced technical analysis integration"""
    
    def test_analyze(self):
        """Test technical analysis"""
        with patch('alpaca_trade_api.REST', return_value=self.mock_api):
            result = self.technical_analysis.analyze('AAPL')
            self.assertIsNotNone(result)
            self.assertIn('score', result)
            self.assertIn('recommendation', result)
            self.assertIn('indicators', result)


class TestPortfolioManager(IntegrationTestBase):
    """Test portfolio manager integration"""
    
    def test_get_portfolio_stats(self):
        """Test getting portfolio statistics"""
        with patch('alpaca_trade_api.REST', return_value=self.mock_api):
            stats = self.portfolio_manager.get_portfolio_stats()
            self.assertIsNotNone(stats)
            self.assertIn('total_value', stats)
            self.assertIn('cash', stats)
            self.assertIn('positions_value', stats)
            self.assertIn('positions_count', stats)
    
    def test_check_correlation(self):
        """Test checking correlation between stocks"""
        with patch('alpaca_trade_api.REST', return_value=self.mock_api):
            correlation = self.portfolio_manager.check_correlation('AAPL', 'MSFT')
            self.assertIsNotNone(correlation)
            self.assertGreaterEqual(correlation, -1.0)
            self.assertLessEqual(correlation, 1.0)


class TestRiskManager(IntegrationTestBase):
    """Test risk manager integration"""
    
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
            stocks = self.stock_scanner.scan_for_stocks()
            self.assertGreater(len(stocks), 0)
            
            # 2. Evaluate stocks with technical analysis
            for stock in stocks[:2]:  # Test with first 2 stocks
                symbol = stock['symbol']
                
                # Get technical analysis
                ta_result = self.technical_analysis.analyze(symbol)
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
                with patch.object(self.buy_engine, 'evaluate_stock', return_value={
                    'symbol': symbol,
                    'score': 80.0,
                    'buy_decision': True
                }):
                    eval_result = self.buy_engine.evaluate_stock(symbol)
                    
                    if eval_result['buy_decision']:
                        # 4. Calculate position size with risk management
                        with patch.object(self.risk_manager, 'calculate_position_size', return_value=200.0):
                            position_size = self.risk_manager.calculate_position_size(1000.0, stock['price'])
                            self.assertGreater(position_size, 0)
                        
                        # 5. Execute buy
                        buy_result = self.buy_engine.execute_buy(symbol, 1.0, stock['price'])
                        self.assertTrue(buy_result['success'])
                        
                        # Record purchase for tax optimization
                        lot_id = self.tax_optimizer.record_purchase(symbol, 1.0, stock['price'])
                        self.assertIsNotNone(lot_id)
            
            # 6. Update positions
            self.monitor_engine.update_positions()
            
            # 7. Check stop losses with risk management
            with patch.object(self.risk_manager, 'check_stop_loss', return_value=(False, None)):
                self.monitor_engine.check_stop_losses()
            
            # 8. Evaluate positions for selling
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM positions WHERE status = ?', ('open',))
            positions = cursor.fetchall()
            conn.close()
            
            for position_id in [p[0] for p in positions]:
                with patch.object(self.sell_engine, 'evaluate_position', return_value={
                    'symbol': 'AAPL',
                    'score': 30.0,
                    'sell_decision': True
                }):
                    eval_result = self.sell_engine.evaluate_position(position_id)
                    
                    if eval_result['sell_decision']:
                        # 9. Execute sell
                        sell_result = self.sell_engine.execute_sell(position_id)
                        self.assertTrue(sell_result['success'])
                        
                        # Record sale for tax optimization
                        with patch.object(self.tax_optimizer, 'record_sale', return_value={
                            'symbol': 'AAPL',
                            'quantity': 1.0,
                            'price': 160.0,
                            'gain_loss': 10.0
                        }):
                            sale = self.tax_optimizer.record_sale('AAPL', 1.0, 160.0)
                            self.assertIn('gain_loss', sale)
            
            # 10. Get portfolio statistics
            stats = self.portfolio_manager.get_portfolio_stats()
            self.assertIn('total_value', stats)
            
            # 11. Get tax report
            report = self.tax_optimizer.generate_tax_report()
            self.assertIn('year', report)


if __name__ == '__main__':
    unittest.main()
