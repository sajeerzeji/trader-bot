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
logger = logging.getLogger('PortfolioManager')

# Load environment variables
load_dotenv()

class PortfolioManager:
    def __init__(self):
        """Initialize the Portfolio Manager with API keys and settings"""
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
        
        logger.info("PortfolioManager initialized")
    
    def load_settings(self):
        """Load settings from the database and environment variables"""
        try:
            conn = sqlite3.connect('tradebot.db')
            cursor = conn.cursor()
            
            # Get all settings
            cursor.execute('SELECT name, value FROM settings')
            settings = dict(cursor.fetchall())
            
            # Load from environment variables first, then database, then defaults
            self.initial_account_size = float(os.getenv('INITIAL_ACCOUNT_SIZE', 10.0))
            self.min_positions = int(os.getenv('MIN_POSITIONS', 3))
            self.max_positions = int(os.getenv('MAX_POSITIONS', 5))
            self.max_position_size = float(os.getenv('MAX_POSITION_SIZE', 0.3)) * self.initial_account_size
            self.max_portfolio_pct = float(settings.get('max_portfolio_pct', 80.0))
            self.cash_reserve = float(settings.get('cash_reserve', 2.0))
            self.sector_diversification = settings.get('sector_diversification', 'True').lower() == 'true'
            self.correlation_threshold = float(settings.get('correlation_threshold', 0.7))
            
            conn.close()
            logger.info("Settings loaded from database and environment variables")
            logger.info(f"Initial account size: ${self.initial_account_size:.2f}")
            logger.info(f"Position requirements: {self.min_positions}-{self.max_positions} positions")
            
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
            # Use default values if database access fails
            self.initial_account_size = 10.0
            self.min_positions = 3
            self.max_positions = 5
            self.max_position_size = 3.0  # 30% of $10
            self.max_portfolio_pct = 80.0
            self.cash_reserve = 2.0
            self.sector_diversification = True
            self.correlation_threshold = 0.7
    
    def get_account_info(self):
        """Get account information from Alpaca"""
        try:
            account = self.api.get_account()
            return {
                'cash': float(account.cash),
                'equity': float(account.equity),
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value)
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None
    
    def get_active_positions(self):
        """Get all active positions from the database"""
        try:
            conn = sqlite3.connect('tradebot.db')
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT id, symbol, quantity, buy_price, buy_time, current_price, profit_loss
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
                    'current_price': row[5] if row[5] else row[3],  # Use buy_price if current_price is None
                    'profit_loss': row[6] if row[6] else 0.0
                })
            
            conn.close()
            
            logger.info(f"Found {len(positions)} active positions")
            return positions
            
        except Exception as e:
            logger.error(f"Error getting active positions: {e}")
            return []
    
    def get_position_value(self, position):
        """Calculate the current value of a position"""
        return position['quantity'] * position['current_price']
    
    def get_portfolio_value(self, positions=None):
        """Calculate the total value of all positions"""
        if positions is None:
            positions = self.get_active_positions()
        
        return sum(self.get_position_value(position) for position in positions)
    
    def get_position_weights(self, positions=None):
        """Calculate the weight of each position in the portfolio"""
        if positions is None:
            positions = self.get_active_positions()
        
        portfolio_value = self.get_portfolio_value(positions)
        if portfolio_value == 0:
            return {position['symbol']: 0 for position in positions}
        
        return {
            position['symbol']: self.get_position_value(position) / portfolio_value
            for position in positions
        }
    
    def calculate_optimal_position_size(self, symbol, price, conviction_score=None):
        """Calculate the optimal position size for a new stock
        
        Args:
            symbol: Stock symbol
            price: Current price of the stock
            conviction_score: Optional score (0-100) indicating conviction level
            
        Returns:
            quantity: Number of shares to buy
            dollar_amount: Dollar amount to invest
        """
        try:
            # Get account info
            account_info = self.get_account_info()
            if not account_info:
                logger.error("Could not get account information")
                return 0, 0
            
            # Get current positions
            positions = self.get_active_positions()
            
            # Calculate available cash
            available_cash = float(account_info['cash']) - self.cash_reserve
            if available_cash <= 0:
                logger.warning("Not enough cash available (need to maintain reserve)")
                return 0, 0
            
            # Check if we already have too many positions
            if len(positions) >= self.max_positions:
                logger.warning(f"Already at maximum positions ({self.max_positions})")
                return 0, 0
            
            # Calculate current portfolio allocation
            portfolio_value = float(account_info['equity'])
            current_allocation = self.get_portfolio_value(positions) / portfolio_value * 100
            
            # Check if we're already at maximum portfolio allocation
            if current_allocation >= self.max_portfolio_pct:
                logger.warning(f"Already at maximum portfolio allocation ({self.max_portfolio_pct}%)")
                return 0, 0
            
            # Base position size calculation
            base_position_size = min(
                self.max_position_size,
                available_cash,
                (self.max_portfolio_pct - current_allocation) / 100 * portfolio_value
            )
            
            # Adjust based on conviction if provided
            if conviction_score is not None:
                # Scale between 50% and 100% of base position size based on conviction
                conviction_factor = 0.5 + (conviction_score / 100) * 0.5
                position_size = base_position_size * conviction_factor
            else:
                position_size = base_position_size
            
            # Calculate quantity
            quantity = position_size / price
            
            logger.info(f"Calculated position size: ${position_size:.2f} = {quantity:.4f} shares at ${price:.2f}")
            
            return quantity, position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0, 0
    
    def calculate_correlation(self, symbols, lookback_days=30):
        """Calculate correlation between stocks
        
        Args:
            symbols: List of stock symbols
            lookback_days: Number of days to look back
            
        Returns:
            correlation_matrix: DataFrame with correlations
        """
        if not symbols or len(symbols) < 2:
            return pd.DataFrame()
        
        try:
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Format dates for Alpaca API
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Get daily bars for all symbols
            bars = self.api.get_barset(symbols, 'day', start=start_str, end=end_str)
            
            # Create price DataFrame
            prices = pd.DataFrame()
            for symbol in symbols:
                if symbol in bars:
                    symbol_bars = bars[symbol]
                    if len(symbol_bars) > 0:
                        prices[symbol] = [bar.c for bar in symbol_bars]
            
            # Calculate returns
            returns = prices.pct_change().dropna()
            
            # Calculate correlation matrix
            correlation_matrix = returns.corr()
            
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Error calculating correlations: {e}")
            return pd.DataFrame()
    
    def check_correlation_constraint(self, new_symbol, existing_symbols):
        """Check if adding a new symbol would violate correlation constraints
        
        Args:
            new_symbol: Symbol to add
            existing_symbols: List of existing symbols in portfolio
            
        Returns:
            bool: True if correlation constraint is satisfied, False otherwise
        """
        if not existing_symbols:
            return True
        
        try:
            # Calculate correlation
            symbols = existing_symbols + [new_symbol]
            correlation_matrix = self.calculate_correlation(symbols)
            
            if correlation_matrix.empty:
                logger.warning("Could not calculate correlation matrix")
                return True
            
            # Check if new symbol is highly correlated with any existing symbol
            for existing_symbol in existing_symbols:
                if existing_symbol in correlation_matrix.columns and new_symbol in correlation_matrix.columns:
                    correlation = correlation_matrix.loc[new_symbol, existing_symbol]
                    if correlation > self.correlation_threshold:
                        logger.warning(f"High correlation ({correlation:.2f}) between {new_symbol} and {existing_symbol}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking correlation constraint: {e}")
            return True  # Default to allowing the trade if we can't check
    
    def get_sector(self, symbol):
        """Get the sector for a stock"""
        try:
            asset = self.api.get_asset(symbol)
            # Note: Alpaca doesn't provide sector information directly
            # In a real implementation, you would use a data provider that offers this
            # For now, we'll return a placeholder
            return "Unknown"
        except Exception as e:
            logger.error(f"Error getting sector for {symbol}: {e}")
            return "Unknown"
    
    def check_sector_diversification(self, new_symbol, existing_symbols):
        """Check if adding a new symbol would maintain sector diversification
        
        Args:
            new_symbol: Symbol to add
            existing_symbols: List of existing symbols in portfolio
            
        Returns:
            bool: True if sector diversification is maintained, False otherwise
        """
        if not self.sector_diversification or not existing_symbols:
            return True
        
        try:
            # Get sectors for all symbols
            new_sector = self.get_sector(new_symbol)
            existing_sectors = [self.get_sector(symbol) for symbol in existing_symbols]
            
            # Count occurrences of each sector
            sector_counts = {}
            for sector in existing_sectors:
                if sector in sector_counts:
                    sector_counts[sector] += 1
                else:
                    sector_counts[sector] = 1
            
            # Check if adding the new symbol would create too much concentration
            if new_sector in sector_counts:
                sector_count = sector_counts[new_sector] + 1
                max_allowed = max(1, len(existing_symbols) // 2)  # Allow at most half of portfolio in one sector
                
                if sector_count > max_allowed:
                    logger.warning(f"Too many positions in sector {new_sector} ({sector_count} > {max_allowed})")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking sector diversification: {e}")
            return True  # Default to allowing the trade if we can't check
    
    def rebalance_portfolio(self):
        """Rebalance the portfolio to maintain target weights
        
        Returns:
            list: List of trades to execute (buy/sell)
        """
        try:
            # Get current positions
            positions = self.get_active_positions()
            if not positions:
                logger.info("No positions to rebalance")
                return []
            
            # Calculate current weights
            current_weights = self.get_position_weights(positions)
            
            # Calculate target weights (equal weight for now)
            target_weight = 1.0 / len(positions)
            target_weights = {position['symbol']: target_weight for position in positions}
            
            # Calculate weight differences
            weight_diffs = {
                symbol: target_weights[symbol] - current_weights[symbol]
                for symbol in current_weights
            }
            
            # Determine trades needed
            trades = []
            for position in positions:
                symbol = position['symbol']
                weight_diff = weight_diffs[symbol]
                
                # Only rebalance if the difference is significant (>5%)
                if abs(weight_diff) > 0.05:
                    portfolio_value = self.get_portfolio_value(positions)
                    dollar_amount = weight_diff * portfolio_value
                    quantity = dollar_amount / position['current_price']
                    
                    trades.append({
                        'symbol': symbol,
                        'action': 'buy' if weight_diff > 0 else 'sell',
                        'quantity': abs(quantity),
                        'dollar_amount': abs(dollar_amount)
                    })
            
            return trades
            
        except Exception as e:
            logger.error(f"Error rebalancing portfolio: {e}")
            return []
    
    def kelly_criterion(self, win_rate, win_loss_ratio):
        """Calculate Kelly Criterion for optimal position sizing
        
        Args:
            win_rate: Probability of winning (0-1)
            win_loss_ratio: Ratio of average win to average loss
            
        Returns:
            float: Optimal fraction of bankroll to bet
        """
        # Kelly formula: f* = p - (1-p)/r
        # where p is probability of winning, r is win/loss ratio
        kelly = win_rate - (1 - win_rate) / win_loss_ratio
        
        # Limit to reasonable values (half-Kelly is often used in practice)
        kelly = max(0, min(kelly, 0.5))
        
        return kelly
    
    def calculate_position_size_kelly(self, symbol, conviction_score):
        """Calculate position size using Kelly criterion
        
        Args:
            symbol: Stock symbol
            conviction_score: Score (0-100) indicating conviction level
            
        Returns:
            float: Optimal position size as percentage of portfolio
        """
        # Convert conviction score to win rate (50-100 -> 0.5-0.8)
        win_rate = 0.5 + (conviction_score - 50) / 100 * 0.3
        win_rate = max(0.5, min(0.8, win_rate))
        
        # Assume win/loss ratio based on conviction
        win_loss_ratio = 1.5 + (conviction_score / 100)
        
        # Calculate Kelly criterion
        kelly = self.kelly_criterion(win_rate, win_loss_ratio)
        
        return kelly
    
    def can_add_position(self, symbol, price, conviction_score=None):
        """Check if a new position can be added to the portfolio
        
        Args:
            symbol: Stock symbol
            price: Current price
            conviction_score: Optional conviction score (0-100)
            
        Returns:
            tuple: (can_add, reason)
        """
        # Get current positions
        positions = self.get_active_positions()
        
        # Check if we already have this symbol
        if any(p['symbol'] == symbol for p in positions):
            return False, "Symbol already in portfolio"
        
        # Check if we're at max positions
        if len(positions) >= self.max_positions:
            return False, f"Already at maximum positions ({self.max_positions})"
        
        # Check correlation constraint
        existing_symbols = [p['symbol'] for p in positions]
        if not self.check_correlation_constraint(symbol, existing_symbols):
            return False, "Would violate correlation constraint"
        
        # Check sector diversification
        if not self.check_sector_diversification(symbol, existing_symbols):
            return False, "Would violate sector diversification constraint"
        
        # Calculate current total investment
        current_investment = sum(p['quantity'] * p['current_price'] for p in positions)
        
        # Check if adding this position would exceed initial account size
        estimated_quantity, estimated_dollar_amount = self.calculate_optimal_position_size(symbol, price, conviction_score)
        if current_investment + estimated_dollar_amount > self.initial_account_size:
            return False, f"Adding this position would exceed initial account size of ${self.initial_account_size:.2f}"
        
        # Check if we have enough cash
        quantity, dollar_amount = self.calculate_optimal_position_size(symbol, price, conviction_score)
        if quantity <= 0:
            return False, "Not enough cash available"
        
        return True, f"Can add position: {quantity:.4f} shares (${dollar_amount:.2f})"
    
    def needs_more_positions(self):
        """Check if the portfolio needs more positions to meet the minimum requirement"""
        positions = self.get_active_positions()
        current_count = len(positions)
        
        if current_count < self.min_positions:
            logger.info(f"Portfolio has {current_count} positions, below minimum of {self.min_positions}")
            return True, self.min_positions - current_count
        else:
            return False, 0
    
    def get_portfolio_stats(self):
        """Get portfolio statistics"""
        try:
            # Get account info
            account_info = self.get_account_info()
            if not account_info:
                return None
            
            # Get current positions
            positions = self.get_active_positions()
            
            # Calculate statistics
            portfolio_value = float(account_info['equity']) if account_info else self.initial_account_size
            cash = float(account_info['cash']) if account_info else self.initial_account_size
            invested_value = self.get_portfolio_value(positions)
            cash_pct = cash / portfolio_value * 100 if portfolio_value > 0 else 0
            invested_pct = invested_value / portfolio_value * 100 if portfolio_value > 0 else 0
            
            # Calculate position weights
            weights = self.get_position_weights(positions)
            
            # Calculate diversification metrics
            diversification = 0
            if positions:
                # Herfindahl-Hirschman Index (HHI) - lower is more diversified
                hhi = sum(w**2 for w in weights.values())
                # Convert to a 0-100 diversification score (100 is most diversified)
                diversification = 100 * (1 - hhi)
            
            # Check if we need more positions
            needs_more, positions_needed = self.needs_more_positions()
            
            return {
                'portfolio_value': portfolio_value,
                'cash': cash,
                'invested_value': invested_value,
                'cash_pct': cash_pct,
                'invested_pct': invested_pct,
                'num_positions': len(positions),
                'min_positions': self.min_positions,
                'max_positions': self.max_positions,
                'needs_more_positions': needs_more,
                'positions_needed': positions_needed,
                'position_weights': weights,
                'diversification_score': diversification
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio stats: {e}")
            return None


# Test the portfolio manager if run directly
if __name__ == "__main__":
    portfolio_manager = PortfolioManager()
    
    # Get portfolio stats
    stats = portfolio_manager.get_portfolio_stats()
    
    if stats:
        print("Portfolio Statistics:")
        print(f"Portfolio Value: ${stats['portfolio_value']:.2f}")
        print(f"Cash: ${stats['cash']:.2f} ({stats['cash_pct']:.1f}%)")
        print(f"Invested: ${stats['invested_value']:.2f} ({stats['invested_pct']:.1f}%)")
        print(f"Positions: {stats['num_positions']}/{stats['max_positions']}")
        print(f"Diversification Score: {stats['diversification_score']:.1f}/100")
        
        if stats['position_weights']:
            print("\nPosition Weights:")
            for symbol, weight in stats['position_weights'].items():
                print(f"- {symbol}: {weight*100:.1f}%")
    else:
        print("Could not get portfolio statistics")
