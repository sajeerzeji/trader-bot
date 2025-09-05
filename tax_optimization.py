import os
import sqlite3
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tradebot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TaxOptimization')

# Load environment variables
load_dotenv()

class TaxOptimization:
    def __init__(self):
        """Initialize the Tax Optimization component"""
        # Load settings
        self.tax_aware_trading = os.getenv('TAX_AWARE_TRADING', 'True').lower() == 'true'
        self.tax_loss_harvesting = os.getenv('TAX_LOSS_HARVESTING', 'True').lower() == 'true'
        self.short_term_tax_rate = float(os.getenv('SHORT_TERM_TAX_RATE', '25.0'))
        self.long_term_tax_rate = float(os.getenv('LONG_TERM_TAX_RATE', '15.0'))
        self.wash_sale_window_days = int(os.getenv('WASH_SALE_WINDOW_DAYS', '30'))
        
        # Create tax tables if they don't exist
        self.init_database()
        
        logger.info("TaxOptimization component initialized")
    
    def init_database(self):
        """Initialize database tables for tax optimization"""
        try:
            conn = sqlite3.connect('tradebot.db')
            cursor = conn.cursor()
            
            # Create tax_lots table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS tax_lots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                quantity REAL NOT NULL,
                purchase_price REAL NOT NULL,
                purchase_date TIMESTAMP NOT NULL,
                sale_price REAL,
                sale_date TIMESTAMP,
                gain_loss REAL,
                tax_status TEXT DEFAULT 'short_term',
                wash_sale_disallowed REAL DEFAULT 0.0,
                status TEXT DEFAULT 'open'
            )
            ''')
            
            # Create tax_summary table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS tax_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                year INTEGER NOT NULL,
                short_term_gains REAL DEFAULT 0.0,
                long_term_gains REAL DEFAULT 0.0,
                wash_sale_adjustments REAL DEFAULT 0.0,
                estimated_tax_liability REAL DEFAULT 0.0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Tax optimization database tables initialized")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def record_purchase(self, symbol, quantity, price, timestamp=None):
        """Record a stock purchase as a new tax lot
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares purchased
            price: Purchase price per share
            timestamp: Purchase timestamp (default: now)
            
        Returns:
            int: Tax lot ID
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        try:
            conn = sqlite3.connect('tradebot.db')
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO tax_lots (
                symbol, quantity, purchase_price, purchase_date, status
            )
            VALUES (?, ?, ?, ?, ?)
            ''', (
                symbol, quantity, price, timestamp.isoformat(), 'open'
            ))
            
            lot_id = cursor.lastrowid
            
            conn.commit()
            conn.close()
            
            logger.info(f"Recorded purchase of {quantity} shares of {symbol} at ${price} as tax lot {lot_id}")
            
            return lot_id
            
        except Exception as e:
            logger.error(f"Error recording purchase: {e}")
            return None
    
    def record_sale(self, symbol, quantity, price, timestamp=None):
        """Record a stock sale using specific tax lot selection strategy
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares sold
            price: Sale price per share
            timestamp: Sale timestamp (default: now)
            
        Returns:
            dict: Sale information
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        try:
            # Get open tax lots for this symbol
            lots = self.get_open_lots(symbol)
            
            if not lots or sum(lot['quantity'] for lot in lots) < quantity:
                logger.error(f"Not enough open lots for {symbol} to sell {quantity} shares")
                return None
            
            # Select lots to sell based on tax optimization strategy
            selected_lots = self.select_lots_for_sale(lots, quantity, price, timestamp)
            
            # Record the sale
            conn = sqlite3.connect('tradebot.db')
            cursor = conn.cursor()
            
            total_proceeds = 0
            total_cost_basis = 0
            total_gain_loss = 0
            total_wash_sale_adjustment = 0
            
            for lot in selected_lots:
                lot_id = lot['id']
                lot_quantity = lot['sell_quantity']
                purchase_price = lot['purchase_price']
                purchase_date = datetime.fromisoformat(lot['purchase_date'])
                
                # Calculate gain/loss
                proceeds = lot_quantity * price
                cost_basis = lot_quantity * purchase_price
                gain_loss = proceeds - cost_basis
                
                # Determine if long-term or short-term
                holding_period = (timestamp - purchase_date).days
                tax_status = 'long_term' if holding_period >= 365 else 'short_term'
                
                # Check for wash sale
                wash_sale_adjustment = self.check_wash_sale(symbol, lot_id, gain_loss, timestamp)
                
                # Update the tax lot
                if lot_quantity == lot['quantity']:
                    # Full lot sale
                    cursor.execute('''
                    UPDATE tax_lots
                    SET sale_price = ?, sale_date = ?, gain_loss = ?,
                        tax_status = ?, wash_sale_disallowed = ?, status = 'closed'
                    WHERE id = ?
                    ''', (
                        price, timestamp.isoformat(), gain_loss,
                        tax_status, wash_sale_adjustment, lot_id
                    ))
                else:
                    # Partial lot sale - close the original lot and create a new one for the remainder
                    cursor.execute('''
                    UPDATE tax_lots
                    SET quantity = ?, status = 'closed', sale_price = ?, sale_date = ?,
                        gain_loss = ?, tax_status = ?, wash_sale_disallowed = ?
                    WHERE id = ?
                    ''', (
                        lot_quantity, price, timestamp.isoformat(),
                        gain_loss, tax_status, wash_sale_adjustment, lot_id
                    ))
                    
                    # Create new lot for remaining shares
                    remaining_quantity = lot['quantity'] - lot_quantity
                    cursor.execute('''
                    INSERT INTO tax_lots (
                        symbol, quantity, purchase_price, purchase_date, status
                    )
                    VALUES (?, ?, ?, ?, ?)
                    ''', (
                        symbol, remaining_quantity, purchase_price, lot['purchase_date'], 'open'
                    ))
                
                # Update totals
                total_proceeds += proceeds
                total_cost_basis += cost_basis
                total_gain_loss += gain_loss
                total_wash_sale_adjustment += wash_sale_adjustment
            
            conn.commit()
            conn.close()
            
            # Update tax summary
            self.update_tax_summary()
            
            logger.info(f"Recorded sale of {quantity} shares of {symbol} at ${price} with gain/loss ${total_gain_loss:.2f}")
            
            return {
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'proceeds': total_proceeds,
                'cost_basis': total_cost_basis,
                'gain_loss': total_gain_loss,
                'wash_sale_adjustment': total_wash_sale_adjustment,
                'net_gain_loss': total_gain_loss - total_wash_sale_adjustment
            }
            
        except Exception as e:
            logger.error(f"Error recording sale: {e}")
            return None
    
    def get_open_lots(self, symbol=None):
        """Get open tax lots
        
        Args:
            symbol: Optional stock symbol to filter by
            
        Returns:
            list: Open tax lots
        """
        try:
            conn = sqlite3.connect('tradebot.db')
            cursor = conn.cursor()
            
            if symbol:
                cursor.execute('''
                SELECT id, symbol, quantity, purchase_price, purchase_date
                FROM tax_lots
                WHERE symbol = ? AND status = 'open'
                ORDER BY purchase_date ASC
                ''', (symbol,))
            else:
                cursor.execute('''
                SELECT id, symbol, quantity, purchase_price, purchase_date
                FROM tax_lots
                WHERE status = 'open'
                ORDER BY symbol, purchase_date ASC
                ''')
            
            lots = []
            for row in cursor.fetchall():
                lots.append({
                    'id': row[0],
                    'symbol': row[1],
                    'quantity': row[2],
                    'purchase_price': row[3],
                    'purchase_date': row[4]
                })
            
            conn.close()
            
            return lots
            
        except Exception as e:
            logger.error(f"Error getting open lots: {e}")
            return []
    
    def select_lots_for_sale(self, lots, quantity, price, timestamp):
        """Select tax lots for sale based on tax optimization strategy
        
        Args:
            lots: List of open tax lots
            quantity: Number of shares to sell
            price: Sale price per share
            timestamp: Sale timestamp
            
        Returns:
            list: Selected lots with quantities to sell
        """
        if not self.tax_aware_trading:
            # If tax-aware trading is disabled, use FIFO (First In, First Out)
            return self.select_lots_fifo(lots, quantity)
        
        # Get current year's tax situation
        tax_summary = self.get_tax_summary(timestamp.year)
        
        # Determine if we have net gains or losses so far this year
        net_gains = tax_summary['short_term_gains'] + tax_summary['long_term_gains']
        
        if net_gains > 0 and self.tax_loss_harvesting:
            # If we have net gains, prioritize selling lots with losses to offset gains
            return self.select_lots_tax_loss_harvesting(lots, quantity, price)
        else:
            # Otherwise, use the optimal tax strategy based on gain/loss
            return self.select_lots_optimal(lots, quantity, price, timestamp)
    
    def select_lots_fifo(self, lots, quantity):
        """Select lots using FIFO (First In, First Out) method
        
        Args:
            lots: List of open tax lots
            quantity: Number of shares to sell
            
        Returns:
            list: Selected lots with quantities to sell
        """
        remaining = quantity
        selected = []
        
        # Sort by purchase date (oldest first)
        sorted_lots = sorted(lots, key=lambda x: x['purchase_date'])
        
        for lot in sorted_lots:
            if remaining <= 0:
                break
                
            lot_quantity = min(lot['quantity'], remaining)
            lot_copy = lot.copy()
            lot_copy['sell_quantity'] = lot_quantity
            selected.append(lot_copy)
            
            remaining -= lot_quantity
        
        return selected
    
    def select_lots_tax_loss_harvesting(self, lots, quantity, price):
        """Select lots to maximize tax loss harvesting
        
        Args:
            lots: List of open tax lots
            quantity: Number of shares to sell
            price: Sale price per share
            
        Returns:
            list: Selected lots with quantities to sell
        """
        remaining = quantity
        selected = []
        
        # Calculate potential gain/loss for each lot
        for lot in lots:
            lot['potential_gain_loss'] = (price - lot['purchase_price']) * lot['quantity']
        
        # Sort by potential gain/loss (losses first)
        sorted_lots = sorted(lots, key=lambda x: x['potential_gain_loss'])
        
        for lot in sorted_lots:
            if remaining <= 0:
                break
                
            lot_quantity = min(lot['quantity'], remaining)
            lot_copy = lot.copy()
            lot_copy['sell_quantity'] = lot_quantity
            selected.append(lot_copy)
            
            remaining -= lot_quantity
        
        return selected
    
    def select_lots_optimal(self, lots, quantity, price, timestamp):
        """Select lots using optimal tax strategy
        
        Args:
            lots: List of open tax lots
            quantity: Number of shares to sell
            price: Sale price per share
            timestamp: Sale timestamp
            
        Returns:
            list: Selected lots with quantities to sell
        """
        remaining = quantity
        selected = []
        
        # Calculate tax impact for each lot
        for lot in lots:
            purchase_date = datetime.fromisoformat(lot['purchase_date'])
            holding_period = (timestamp - purchase_date).days
            is_long_term = holding_period >= 365
            
            # Calculate potential gain/loss
            potential_gain_loss = (price - lot['purchase_price']) * lot['quantity']
            
            # Calculate tax rate
            tax_rate = self.long_term_tax_rate if is_long_term else self.short_term_tax_rate
            
            # Calculate tax impact (negative for losses, positive for gains)
            if potential_gain_loss > 0:
                tax_impact = potential_gain_loss * (tax_rate / 100)
            else:
                tax_impact = potential_gain_loss * (self.short_term_tax_rate / 100)
            
            lot['tax_impact'] = tax_impact
            lot['tax_impact_per_share'] = tax_impact / lot['quantity']
        
        # For gains: prefer long-term (lower tax rate)
        # For losses: prefer short-term (higher tax benefit)
        # Sort by tax impact per share (lowest first)
        sorted_lots = sorted(lots, key=lambda x: x['tax_impact_per_share'])
        
        for lot in sorted_lots:
            if remaining <= 0:
                break
                
            lot_quantity = min(lot['quantity'], remaining)
            lot_copy = lot.copy()
            lot_copy['sell_quantity'] = lot_quantity
            selected.append(lot_copy)
            
            remaining -= lot_quantity
        
        return selected
    
    def check_wash_sale(self, symbol, lot_id, gain_loss, sale_date):
        """Check if a sale triggers the wash sale rule
        
        Args:
            symbol: Stock symbol
            lot_id: Tax lot ID being sold
            gain_loss: Gain/loss amount
            sale_date: Sale date
            
        Returns:
            float: Wash sale adjustment amount (0 if no wash sale)
        """
        if gain_loss >= 0:
            # Wash sale rule only applies to losses
            return 0.0
        
        try:
            conn = sqlite3.connect('tradebot.db')
            cursor = conn.cursor()
            
            # Check for purchases within the wash sale window (30 days before or after)
            window_start = (sale_date - timedelta(days=self.wash_sale_window_days)).isoformat()
            window_end = (sale_date + timedelta(days=self.wash_sale_window_days)).isoformat()
            
            cursor.execute('''
            SELECT SUM(quantity)
            FROM tax_lots
            WHERE symbol = ? AND id != ? AND status = 'open'
            AND purchase_date BETWEEN ? AND ?
            ''', (symbol, lot_id, window_start, window_end))
            
            replacement_shares = cursor.fetchone()[0] or 0
            
            conn.close()
            
            if replacement_shares > 0:
                # Wash sale rule applies - disallow the loss
                wash_sale_adjustment = abs(gain_loss)
                logger.info(f"Wash sale detected for {symbol}: ${wash_sale_adjustment:.2f} loss disallowed")
                return wash_sale_adjustment
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error checking wash sale: {e}")
            return 0.0
    
    def update_tax_summary(self):
        """Update the tax summary for the current year"""
        try:
            current_year = datetime.now().year
            
            conn = sqlite3.connect('tradebot.db')
            cursor = conn.cursor()
            
            # Calculate short-term gains/losses
            cursor.execute('''
            SELECT SUM(gain_loss)
            FROM tax_lots
            WHERE tax_status = 'short_term' AND status = 'closed'
            AND strftime('%Y', sale_date) = ?
            ''', (str(current_year),))
            
            short_term_gains = cursor.fetchone()[0] or 0.0
            
            # Calculate long-term gains/losses
            cursor.execute('''
            SELECT SUM(gain_loss)
            FROM tax_lots
            WHERE tax_status = 'long_term' AND status = 'closed'
            AND strftime('%Y', sale_date) = ?
            ''', (str(current_year),))
            
            long_term_gains = cursor.fetchone()[0] or 0.0
            
            # Calculate wash sale adjustments
            cursor.execute('''
            SELECT SUM(wash_sale_disallowed)
            FROM tax_lots
            WHERE status = 'closed'
            AND strftime('%Y', sale_date) = ?
            ''', (str(current_year),))
            
            wash_sale_adjustments = cursor.fetchone()[0] or 0.0
            
            # Calculate estimated tax liability
            short_term_tax = max(0, short_term_gains) * (self.short_term_tax_rate / 100)
            long_term_tax = max(0, long_term_gains) * (self.long_term_tax_rate / 100)
            estimated_tax_liability = short_term_tax + long_term_tax
            
            # Update or insert tax summary
            cursor.execute('''
            SELECT id FROM tax_summary WHERE year = ?
            ''', (current_year,))
            
            if cursor.fetchone():
                cursor.execute('''
                UPDATE tax_summary
                SET short_term_gains = ?, long_term_gains = ?,
                    wash_sale_adjustments = ?, estimated_tax_liability = ?,
                    updated_at = ?
                WHERE year = ?
                ''', (
                    short_term_gains, long_term_gains,
                    wash_sale_adjustments, estimated_tax_liability,
                    datetime.now().isoformat(), current_year
                ))
            else:
                cursor.execute('''
                INSERT INTO tax_summary (
                    year, short_term_gains, long_term_gains,
                    wash_sale_adjustments, estimated_tax_liability
                )
                VALUES (?, ?, ?, ?, ?)
                ''', (
                    current_year, short_term_gains, long_term_gains,
                    wash_sale_adjustments, estimated_tax_liability
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Updated tax summary for {current_year}")
            
        except Exception as e:
            logger.error(f"Error updating tax summary: {e}")
    
    def get_tax_summary(self, year=None):
        """Get tax summary for a specific year
        
        Args:
            year: Year to get summary for (default: current year)
            
        Returns:
            dict: Tax summary
        """
        if year is None:
            year = datetime.now().year
        
        try:
            conn = sqlite3.connect('tradebot.db')
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT year, short_term_gains, long_term_gains,
                   wash_sale_adjustments, estimated_tax_liability
            FROM tax_summary
            WHERE year = ?
            ''', (year,))
            
            row = cursor.fetchone()
            
            conn.close()
            
            if row:
                return {
                    'year': row[0],
                    'short_term_gains': row[1],
                    'long_term_gains': row[2],
                    'wash_sale_adjustments': row[3],
                    'estimated_tax_liability': row[4],
                    'total_gains': row[1] + row[2]
                }
            else:
                return {
                    'year': year,
                    'short_term_gains': 0.0,
                    'long_term_gains': 0.0,
                    'wash_sale_adjustments': 0.0,
                    'estimated_tax_liability': 0.0,
                    'total_gains': 0.0
                }
            
        except Exception as e:
            logger.error(f"Error getting tax summary: {e}")
            return {
                'year': year,
                'short_term_gains': 0.0,
                'long_term_gains': 0.0,
                'wash_sale_adjustments': 0.0,
                'estimated_tax_liability': 0.0,
                'total_gains': 0.0
            }
    
    def generate_tax_report(self, year=None):
        """Generate a tax report for a specific year
        
        Args:
            year: Year to generate report for (default: current year)
            
        Returns:
            dict: Tax report data
        """
        if year is None:
            year = datetime.now().year
        
        try:
            conn = sqlite3.connect('tradebot.db')
            cursor = conn.cursor()
            
            # Get closed positions for the year
            cursor.execute('''
            SELECT symbol, quantity, purchase_price, purchase_date,
                   sale_price, sale_date, gain_loss, tax_status,
                   wash_sale_disallowed
            FROM tax_lots
            WHERE status = 'closed'
            AND strftime('%Y', sale_date) = ?
            ORDER BY sale_date ASC
            ''', (str(year),))
            
            trades = []
            for row in cursor.fetchall():
                trades.append({
                    'symbol': row[0],
                    'quantity': row[1],
                    'purchase_price': row[2],
                    'purchase_date': row[3],
                    'sale_price': row[4],
                    'sale_date': row[5],
                    'gain_loss': row[6],
                    'tax_status': row[7],
                    'wash_sale_disallowed': row[8],
                    'net_gain_loss': row[6] - row[8]
                })
            
            conn.close()
            
            # Get tax summary
            summary = self.get_tax_summary(year)
            
            # Calculate statistics
            total_trades = len(trades)
            profitable_trades = sum(1 for t in trades if t['gain_loss'] > 0)
            loss_trades = sum(1 for t in trades if t['gain_loss'] < 0)
            
            win_rate = profitable_trades / total_trades if total_trades > 0 else 0
            
            # Group by month
            monthly_data = {}
            for trade in trades:
                sale_date = datetime.fromisoformat(trade['sale_date'])
                month = sale_date.month
                
                if month not in monthly_data:
                    monthly_data[month] = {
                        'trades': 0,
                        'gain_loss': 0.0,
                        'wash_sale_disallowed': 0.0,
                        'net_gain_loss': 0.0
                    }
                
                monthly_data[month]['trades'] += 1
                monthly_data[month]['gain_loss'] += trade['gain_loss']
                monthly_data[month]['wash_sale_disallowed'] += trade['wash_sale_disallowed']
                monthly_data[month]['net_gain_loss'] += trade['net_gain_loss']
            
            # Format monthly data
            monthly = []
            for month in range(1, 13):
                if month in monthly_data:
                    data = monthly_data[month]
                    monthly.append({
                        'month': month,
                        'month_name': datetime(year, month, 1).strftime('%B'),
                        'trades': data['trades'],
                        'gain_loss': data['gain_loss'],
                        'wash_sale_disallowed': data['wash_sale_disallowed'],
                        'net_gain_loss': data['net_gain_loss']
                    })
                else:
                    monthly.append({
                        'month': month,
                        'month_name': datetime(year, month, 1).strftime('%B'),
                        'trades': 0,
                        'gain_loss': 0.0,
                        'wash_sale_disallowed': 0.0,
                        'net_gain_loss': 0.0
                    })
            
            return {
                'year': year,
                'summary': summary,
                'trades': trades,
                'statistics': {
                    'total_trades': total_trades,
                    'profitable_trades': profitable_trades,
                    'loss_trades': loss_trades,
                    'win_rate': win_rate
                },
                'monthly': monthly
            }
            
        except Exception as e:
            logger.error(f"Error generating tax report: {e}")
            return {
                'year': year,
                'summary': self.get_tax_summary(year),
                'trades': [],
                'statistics': {
                    'total_trades': 0,
                    'profitable_trades': 0,
                    'loss_trades': 0,
                    'win_rate': 0
                },
                'monthly': []
            }
    
    def get_tax_implications(self, symbol, quantity, current_price):
        """Get tax implications of selling a position
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares to sell
            current_price: Current price per share
            
        Returns:
            dict: Tax implications
        """
        try:
            # Get open lots for this symbol
            lots = self.get_open_lots(symbol)
            
            if not lots or sum(lot['quantity'] for lot in lots) < quantity:
                return {
                    'symbol': symbol,
                    'quantity': quantity,
                    'current_price': current_price,
                    'error': "Not enough open lots"
                }
            
            # Select lots to sell
            selected_lots = self.select_lots_for_sale(lots, quantity, current_price, datetime.now())
            
            # Calculate tax implications
            total_proceeds = 0
            total_cost_basis = 0
            total_gain_loss = 0
            short_term_gain_loss = 0
            long_term_gain_loss = 0
            
            for lot in selected_lots:
                lot_quantity = lot['sell_quantity']
                purchase_price = lot['purchase_price']
                purchase_date = datetime.fromisoformat(lot['purchase_date'])
                
                # Calculate gain/loss
                proceeds = lot_quantity * current_price
                cost_basis = lot_quantity * purchase_price
                gain_loss = proceeds - cost_basis
                
                # Determine if long-term or short-term
                holding_period = (datetime.now() - purchase_date).days
                is_long_term = holding_period >= 365
                
                if is_long_term:
                    long_term_gain_loss += gain_loss
                else:
                    short_term_gain_loss += gain_loss
                
                total_proceeds += proceeds
                total_cost_basis += cost_basis
                total_gain_loss += gain_loss
            
            # Calculate estimated tax
            short_term_tax = max(0, short_term_gain_loss) * (self.short_term_tax_rate / 100)
            long_term_tax = max(0, long_term_gain_loss) * (self.long_term_tax_rate / 100)
            estimated_tax = short_term_tax + long_term_tax
            
            # Get current year's tax summary
            tax_summary = self.get_tax_summary()
            
            # Calculate impact on annual tax situation
            new_short_term_gains = tax_summary['short_term_gains'] + short_term_gain_loss
            new_long_term_gains = tax_summary['long_term_gains'] + long_term_gain_loss
            new_total_gains = new_short_term_gains + new_long_term_gains
            
            return {
                'symbol': symbol,
                'quantity': quantity,
                'current_price': current_price,
                'proceeds': total_proceeds,
                'cost_basis': total_cost_basis,
                'gain_loss': total_gain_loss,
                'short_term_gain_loss': short_term_gain_loss,
                'long_term_gain_loss': long_term_gain_loss,
                'estimated_tax': estimated_tax,
                'after_tax_proceeds': total_proceeds - estimated_tax,
                'tax_impact': {
                    'current_year_gains': tax_summary['total_gains'],
                    'new_total_gains': new_total_gains,
                    'net_impact': new_total_gains - tax_summary['total_gains']
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting tax implications: {e}")
            return {
                'symbol': symbol,
                'quantity': quantity,
                'current_price': current_price,
                'error': str(e)
            }


# Test the tax optimization component if run directly
if __name__ == "__main__":
    tax_optimizer = TaxOptimization()
    
    # Test with sample data
    symbol = "AAPL"
    
    # Record a purchase
    purchase_date = datetime.now() - timedelta(days=180)  # 6 months ago
    lot_id = tax_optimizer.record_purchase(symbol, 10, 150.0, purchase_date)
    
    # Record another purchase
    purchase_date = datetime.now() - timedelta(days=400)  # Long-term
    lot_id2 = tax_optimizer.record_purchase(symbol, 5, 130.0, purchase_date)
    
    # Get tax implications of selling
    implications = tax_optimizer.get_tax_implications(symbol, 15, 170.0)
    
    print("Tax Implications of Selling:")
    print(f"Symbol: {implications['symbol']}")
    print(f"Quantity: {implications['quantity']}")
    print(f"Current Price: ${implications['current_price']:.2f}")
    print(f"Proceeds: ${implications['proceeds']:.2f}")
    print(f"Cost Basis: ${implications['cost_basis']:.2f}")
    print(f"Gain/Loss: ${implications['gain_loss']:.2f}")
    print(f"Short-term Gain/Loss: ${implications['short_term_gain_loss']:.2f}")
    print(f"Long-term Gain/Loss: ${implications['long_term_gain_loss']:.2f}")
    print(f"Estimated Tax: ${implications['estimated_tax']:.2f}")
    print(f"After-tax Proceeds: ${implications['after_tax_proceeds']:.2f}")
    
    # Record a sale
    sale = tax_optimizer.record_sale(symbol, 15, 170.0)
    
    # Generate tax report
    report = tax_optimizer.generate_tax_report()
    
    print("\nTax Report Summary:")
    print(f"Year: {report['year']}")
    print(f"Total Trades: {report['statistics']['total_trades']}")
    print(f"Win Rate: {report['statistics']['win_rate']:.2f}")
    print(f"Short-term Gains: ${report['summary']['short_term_gains']:.2f}")
    print(f"Long-term Gains: ${report['summary']['long_term_gains']:.2f}")
    print(f"Total Gains: ${report['summary']['total_gains']:.2f}")
    print(f"Estimated Tax Liability: ${report['summary']['estimated_tax_liability']:.2f}")
