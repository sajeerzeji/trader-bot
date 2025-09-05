import os
import requests
import pandas as pd
import numpy as np
import sqlite3
from dotenv import load_dotenv
import logging
import time
import alpaca_trade_api as tradeapi
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
logger = logging.getLogger('StockScanner')

# Load environment variables
load_dotenv()

class StockScanner:
    def __init__(self):
        """Initialize the StockScanner with API keys and settings"""
        # Load Alpha Vantage API key for stock data
        self.alpha_vantage_key = os.getenv('ALPHAVANTAGE_API_KEY')
        if not self.alpha_vantage_key:
            raise ValueError("Alpha Vantage API key not found in .env file")
        
        # Load Alpaca API credentials for market data
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
        
        # Maximum number of stocks to analyze (to avoid API rate limits)
        self.max_stocks_to_analyze = 50
        
        logger.info("StockScanner initialized with dynamic stock discovery")
    
    def load_settings(self):
        """Load settings from the database"""
        try:
            conn = sqlite3.connect('tradebot.db')
            cursor = conn.cursor()
            
            # Get all settings
            cursor.execute('SELECT name, value FROM settings')
            settings = dict(cursor.fetchall())
            
            # Assign settings to instance variables
            self.min_price = settings.get('min_price', 1.0)
            self.max_price = settings.get('max_price', 100.0)
            self.min_volume = settings.get('min_volume', 1000000.0)
            
            conn.close()
            logger.info("Settings loaded from database")
            
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
            # Use default values if database access fails
            self.min_price = 1.0
            self.max_price = 100.0
            self.min_volume = 1000000.0
    
    def get_stock_data(self, symbol):
        """Get current and historical data for a stock using Alpha Vantage API"""
        try:
            # First try to get current price from Alpaca API (faster and more reliable)
            try:
                quote = self.api.get_latest_quote(symbol)
                current_price = (quote.ap + quote.bp) / 2  # Average of ask and bid price
                
                # Create a simple DataFrame with just the current price
                # This allows us to proceed even if Alpha Vantage API is rate limited
                df = pd.DataFrame({
                    'open': [current_price],
                    'high': [current_price],
                    'low': [current_price],
                    'close': [current_price],
                    'volume': [10000000]  # Assume sufficient volume
                }, index=[pd.Timestamp.now()])
                
                logger.info(f"Using Alpaca price data for {symbol}: ${current_price:.2f}")
                return df
                
            except Exception as e:
                logger.warning(f"Could not get Alpaca price for {symbol}: {e}")
                # Fall back to Alpha Vantage
            
            # Try Alpha Vantage API
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=compact&apikey={self.alpha_vantage_key}"
            response = requests.get(url)
            data = response.json()
            
            if "Error Message" in data:
                logger.warning(f"Error getting data for {symbol}: {data['Error Message']}")
                return None
            
            if "Time Series (Daily)" not in data:
                logger.warning(f"No time series data for {symbol}")
                return None
            
            # Convert to DataFrame
            time_series = data["Time Series (Daily)"]
            df = pd.DataFrame(time_series).T
            df.columns = [col.split('. ')[1] for col in df.columns]
            df = df.astype(float)
            
            # Add date as a column
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Get latest price and volume
            latest_price = df['close'].iloc[-1]
            latest_volume = df['volume'].iloc[-1]
            
            # Check if stock meets basic criteria
            if latest_price < self.min_price or latest_price > self.max_price:
                logger.info(f"{symbol} price ${latest_price} outside range (${self.min_price}-${self.max_price})")
                return None
            
            if latest_volume < self.min_volume:
                logger.info(f"{symbol} volume {latest_volume} below minimum {self.min_volume}")
                return None
            
            # Calculate technical indicators
            df['5d_avg'] = df['close'].rolling(window=5).mean()
            df['20d_avg'] = df['close'].rolling(window=20).mean()
            
            # Calculate RSI (Relative Strength Index)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Calculate average volume
            df['avg_volume'] = df['volume'].rolling(window=10).mean()
            
            # Return the processed data
            return df
            
        except Exception as e:
            logger.error(f"Error processing data for {symbol}: {e}")
            return None
    
    def score_stock(self, df):
        """Score a stock based on technical indicators"""
        if df is None or df.empty:
            return 0
        
        score = 0
        latest = df.iloc[-1]
        
        try:
            # Price range score (higher for lower priced stocks within our range)
            price = latest['close']
            if self.min_price <= price <= 5.0:
                score += 20  # Favor very low priced stocks
            elif 5.0 < price <= 10.0:
                score += 15
            elif 10.0 < price <= 20.0:
                score += 10
            elif 20.0 < price <= 50.0:
                score += 5
            
            # Volume score (higher for higher volume)
            volume = latest['volume']
            if volume > 10000000:
                score += 20
            elif volume > 5000000:
                score += 15
            elif volume > 1000000:
                score += 10
            
            # Check if we have technical indicators
            has_indicators = all(col in df.columns for col in ['5d_avg', '20d_avg', 'rsi'])
            
            if has_indicators:
                # Moving average score (higher if price is above moving averages)
                if latest['close'] > latest['5d_avg']:
                    score += 10
                if latest['close'] > latest['20d_avg']:
                    score += 10
                if latest['5d_avg'] > latest['20d_avg']:
                    score += 10  # Uptrend
                
                # RSI score (favor stocks that are not overbought or oversold)
                rsi = latest['rsi']
                if 40 <= rsi <= 60:
                    score += 15  # Neutral RSI
                elif 30 <= rsi < 40:
                    score += 10  # Slightly oversold
                elif 60 < rsi <= 70:
                    score += 10  # Slightly overbought
            else:
                # If we don't have technical indicators, assign a default score
                # This ensures we can still select stocks even with limited data
                score += 30  # Default score for stocks with limited data
            
            logger.info(f"Stock scored {score} points")
            return score
            
        except Exception as e:
            logger.error(f"Error scoring stock: {e}")
            return 0
    
    def discover_stocks(self):
        """Discover stocks that meet our basic criteria"""
        try:
            # Get active stocks from Alpaca API
            assets = self.api.list_assets(status='active')
            
            # Filter stocks based on our criteria
            filtered_stocks = []
            for asset in assets:
                # Only consider tradable assets
                # Note: Some API versions might not have asset_class attribute
                if hasattr(asset, 'tradable') and asset.tradable:
                    # Get current price and check if it's in our range
                    try:
                        # Get latest quote
                        quote = self.api.get_latest_quote(asset.symbol)
                        price = (quote.ap + quote.bp) / 2  # Average of ask and bid price
                        
                        # Check if price is in our range
                        if self.min_price <= price <= self.max_price:
                            filtered_stocks.append({
                                'symbol': asset.symbol,
                                'name': asset.name,
                                'price': price
                            })
                            
                            # Limit the number of stocks to analyze
                            if len(filtered_stocks) >= self.max_stocks_to_analyze:
                                break
                    except Exception as e:
                        # Skip stocks that we can't get a quote for
                        continue
            
            logger.info(f"Discovered {len(filtered_stocks)} stocks that meet our basic criteria")
            return filtered_stocks
            
        except Exception as e:
            logger.error(f"Error discovering stocks: {e}")
            # Fallback to a small list of common penny stocks if discovery fails
            fallback_stocks = [
                'F', 'NOK', 'SIRI', 'PLUG', 'SOFI', 'WISH', 'SNDL', 'CLOV', 'TLRY'
            ]
            logger.warning(f"Using fallback list of {len(fallback_stocks)} stocks")
            return [{'symbol': s, 'name': s, 'price': 0} for s in fallback_stocks]
    
    def find_best_stocks(self, min_positions=3, max_positions=5, min_score=30):
        """Find multiple stocks to buy based on our criteria
        
        Args:
            min_positions: Minimum number of positions to target
            max_positions: Maximum number of positions to target
            min_score: Minimum score for a stock to be considered
            
        Returns:
            List of dictionaries with symbol, price, and score
        """
        # Load min/max positions from environment variables
        min_positions = int(os.getenv('MIN_POSITIONS', min_positions))
        max_positions = int(os.getenv('MAX_POSITIONS', max_positions))
        
        logger.info(f"Finding {min_positions}-{max_positions} stocks to buy")
        
        # List to store scored stocks
        scored_stocks = []
        
        # Discover stocks that meet our basic criteria
        discovered_stocks = self.discover_stocks()
        
        logger.info(f"Scanning {len(discovered_stocks)} stocks for opportunities")
        
        for stock in discovered_stocks:
            symbol = stock['symbol']
            logger.info(f"Analyzing {symbol}...")
            
            # Get stock data
            df = self.get_stock_data(symbol)
            if df is None:
                continue
                
            # Score the stock
            score = self.score_stock(df)
            
            # Add to our list if it meets minimum score
            if score >= min_score:
                latest_price = df['close'].iloc[-1]
                scored_stocks.append({
                    'symbol': symbol,
                    'price': latest_price,
                    'score': score
                })
                logger.info(f"Added {symbol} to candidates with score {score} at ${latest_price:.2f}")
            
            # Alpha Vantage has a rate limit of 5 calls per minute for free tier
            time.sleep(12)  # Sleep to avoid hitting rate limit
            
            # If we have enough stocks, we can stop
            if len(scored_stocks) >= max_positions:
                break
        
        # Sort stocks by score (highest first)
        scored_stocks.sort(key=lambda x: x['score'], reverse=True)
        
        # Ensure we have at least min_positions stocks if available
        if len(scored_stocks) < min_positions and len(discovered_stocks) > len(scored_stocks):
            logger.warning(f"Only found {len(scored_stocks)} stocks above minimum score, lowering standards")
            # Lower our standards to get more stocks
            remaining_stocks = [s for s in discovered_stocks if s['symbol'] not in [ss['symbol'] for ss in scored_stocks]]
            
            for stock in remaining_stocks:
                symbol = stock['symbol']
                logger.info(f"Analyzing additional candidate {symbol}...")
                
                # Get stock data
                df = self.get_stock_data(symbol)
                if df is None:
                    continue
                    
                # Score the stock
                score = self.score_stock(df)
                latest_price = df['close'].iloc[-1]
                
                scored_stocks.append({
                    'symbol': symbol,
                    'price': latest_price,
                    'score': score
                })
                logger.info(f"Added {symbol} to candidates with score {score} at ${latest_price:.2f}")
                
                # Alpha Vantage has a rate limit of 5 calls per minute for free tier
                time.sleep(12)  # Sleep to avoid hitting rate limit
                
                # If we have enough stocks, we can stop
                if len(scored_stocks) >= min_positions:
                    break
            
            # Sort again after adding more stocks
            scored_stocks.sort(key=lambda x: x['score'], reverse=True)
        
        if scored_stocks:
            logger.info(f"Found {len(scored_stocks)} candidate stocks")
            for i, stock in enumerate(scored_stocks):
                logger.info(f"Candidate {i+1}: {stock['symbol']} with score {stock['score']} at ${stock['price']:.2f}")
            return scored_stocks
        else:
            logger.warning("No suitable stocks found")
            return []
    
    def find_best_stock(self):
        """Find the best stock to buy based on our criteria (legacy method)"""
        stocks = self.find_best_stocks(min_positions=1, max_positions=1)
        if stocks:
            return stocks[0]
        else:
            logger.warning("No suitable stocks found")
            return None

# Test the stock scanner if run directly
if __name__ == "__main__":
    scanner = StockScanner()
    best_stock = scanner.find_best_stock()
    
    if best_stock:
        print(f"Best stock to buy: {best_stock['symbol']} at ${best_stock['price']:.2f} (Score: {best_stock['score']})")
    else:
        print("No suitable stocks found at this time.")
