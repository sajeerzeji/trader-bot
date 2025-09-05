import os
import logging
import json
import time
import threading
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import sqlite3
from dotenv import load_dotenv

# Import our batch manager
from ai_batch_manager import AIBatchManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ai_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('AIIntegration')

# Load environment variables
load_dotenv()

class AIIntegration:
    """
    Integrates Claude AI capabilities with the trading bot components
    using the batch manager for efficient API usage.
    """
    
    def __init__(self, db_path: str = 'tradebot.db'):
        """
        Initialize the AI integration.
        
        Args:
            db_path: Path to the database
        """
        self.db_path = db_path
        
        # Initialize batch manager with appropriate settings
        self.batch_manager = AIBatchManager(
            batch_size=5,              # Process up to 5 queries at once
            batch_interval_seconds=30,  # Wait max 30 seconds to fill a batch
            cache_expiry_minutes=60    # Cache responses for 60 minutes
        )
        
        # Initialize database connection
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
        # Create AI analysis table if it doesn't exist
        self._create_tables()
        
        logger.info("AI Integration initialized")
    
    def _create_tables(self) -> None:
        """Create necessary database tables."""
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS ai_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            analysis_type TEXT NOT NULL,
            analysis_data TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            score REAL,
            recommendation TEXT
        )
        ''')
        
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS ai_batch_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            total_queries INTEGER DEFAULT 0,
            cache_hits INTEGER DEFAULT 0,
            api_calls INTEGER DEFAULT 0,
            avg_response_time REAL,
            errors INTEGER DEFAULT 0
        )
        ''')
        
        self.conn.commit()
    
    def analyze_stock_for_purchase(self, symbol: str, stock_data: Dict, 
                                  technical_analysis: Dict, news_data: Optional[Dict] = None,
                                  social_data: Optional[Dict] = None) -> Dict:
        """
        Analyze a stock for potential purchase using Claude AI.
        
        Args:
            symbol: Stock symbol
            stock_data: Basic stock data (price, volume, etc.)
            technical_analysis: Technical indicators and analysis
            news_data: Recent news sentiment (optional)
            social_data: Social media sentiment (optional)
            
        Returns:
            Analysis results with AI-enhanced decision
        """
        # Prepare the query
        query = self._prepare_stock_purchase_query(symbol, stock_data, technical_analysis, news_data, social_data)
        
        # Generate a unique query ID
        query_id = f"purchase_{symbol}_{int(time.time())}"
        
        # Get response synchronously (we need this for the purchase decision)
        response = self.batch_manager.get_response_sync(query_id, query, timeout_seconds=45)
        
        # Process the response
        analysis = self._process_claude_response(response, "purchase")
        
        # Store the analysis
        self._store_analysis(symbol, "purchase", analysis)
        
        return analysis
    
    def analyze_position_for_sale(self, symbol: str, position_data: Dict,
                                 technical_analysis: Dict, news_data: Optional[Dict] = None,
                                 social_data: Optional[Dict] = None) -> Dict:
        """
        Analyze a position for potential sale using Claude AI.
        
        Args:
            symbol: Stock symbol
            position_data: Position data (entry price, current price, etc.)
            technical_analysis: Technical indicators and analysis
            news_data: Recent news sentiment (optional)
            social_data: Social media sentiment (optional)
            
        Returns:
            Analysis results with AI-enhanced decision
        """
        # Prepare the query
        query = self._prepare_position_sale_query(symbol, position_data, technical_analysis, news_data, social_data)
        
        # Generate a unique query ID
        query_id = f"sale_{symbol}_{int(time.time())}"
        
        # Get response synchronously (we need this for the sale decision)
        response = self.batch_manager.get_response_sync(query_id, query, timeout_seconds=45)
        
        # Process the response
        analysis = self._process_claude_response(response, "sale")
        
        # Store the analysis
        self._store_analysis(symbol, "sale", analysis)
        
        return analysis
    
    def analyze_market_conditions(self, market_data: Dict, economic_indicators: Dict,
                                 sector_performance: Dict) -> Dict:
        """
        Analyze overall market conditions using Claude AI.
        
        Args:
            market_data: Overall market data (indices, volatility, etc.)
            economic_indicators: Economic indicators (GDP, unemployment, etc.)
            sector_performance: Performance by sector
            
        Returns:
            Analysis results with market outlook
        """
        # Prepare the query
        query = self._prepare_market_conditions_query(market_data, economic_indicators, sector_performance)
        
        # Generate a unique query ID
        query_id = f"market_{int(time.time())}"
        
        # This can be processed asynchronously as it's not needed for immediate decisions
        result_ready = threading.Event()
        result_container = {}
        
        def callback(qid, response):
            result_container['response'] = response
            result_ready.set()
        
        # Add to batch queue
        self.batch_manager.add_query(query_id, query, callback)
        
        # Wait for result (with timeout)
        if result_ready.wait(timeout=60):
            # Process the response
            analysis = self._process_claude_response(result_container.get('response'), "market")
            
            # Store the analysis
            self._store_analysis("MARKET", "conditions", analysis)
            
            return analysis
        else:
            logger.warning("Timeout waiting for market analysis")
            return {
                "success": False,
                "error": "Timeout waiting for analysis",
                "recommendation": "neutral",
                "score": 50,
                "insights": []
            }
    
    def analyze_news_batch(self, news_items: List[Dict]) -> Dict:
        """
        Analyze a batch of news items for multiple stocks at once.
        
        Args:
            news_items: List of news items with symbol, title, content
            
        Returns:
            Dictionary of symbol -> sentiment analysis
        """
        # Group news by symbol
        news_by_symbol = {}
        for item in news_items:
            symbol = item.get('symbol')
            if symbol not in news_by_symbol:
                news_by_symbol[symbol] = []
            news_by_symbol[symbol].append(item)
        
        # Prepare a single query for all symbols
        query = self._prepare_news_batch_query(news_by_symbol)
        
        # Generate a unique query ID
        query_id = f"news_batch_{int(time.time())}"
        
        # Get response synchronously
        response = self.batch_manager.get_response_sync(query_id, query, timeout_seconds=60)
        
        # Process the response
        analysis = self._process_claude_response(response, "news_batch")
        
        # Store individual analyses
        for symbol, sentiment in analysis.get('sentiments', {}).items():
            self._store_analysis(symbol, "news", {
                "success": True,
                "sentiment": sentiment,
                "score": sentiment.get('score', 50),
                "insights": sentiment.get('insights', [])
            })
        
        return analysis
    
    def _prepare_stock_purchase_query(self, symbol: str, stock_data: Dict,
                                     technical_analysis: Dict, news_data: Optional[Dict] = None,
                                     social_data: Optional[Dict] = None) -> str:
        """Prepare a query for stock purchase analysis."""
        query = f"""
        You are an AI financial advisor helping with stock trading decisions. Please analyze the following stock for a potential purchase:
        
        Symbol: {symbol}
        
        Stock Data:
        - Current Price: ${stock_data.get('price', 'N/A')}
        - 52-Week Range: ${stock_data.get('52_week_low', 'N/A')} - ${stock_data.get('52_week_high', 'N/A')}
        - Volume: {stock_data.get('volume', 'N/A')}
        - Market Cap: ${stock_data.get('market_cap', 'N/A')}
        - P/E Ratio: {stock_data.get('pe_ratio', 'N/A')}
        
        Technical Analysis:
        - SMA(50): ${technical_analysis.get('sma_50', 'N/A')}
        - SMA(200): ${technical_analysis.get('sma_200', 'N/A')}
        - RSI: {technical_analysis.get('rsi', 'N/A')}
        - MACD: {technical_analysis.get('macd', 'N/A')}
        - Signal: {technical_analysis.get('macd_signal', 'N/A')}
        - Bollinger Bands: Upper=${technical_analysis.get('bb_upper', 'N/A')}, Middle=${technical_analysis.get('bb_middle', 'N/A')}, Lower=${technical_analysis.get('bb_lower', 'N/A')}
        - Technical Score: {technical_analysis.get('score', 'N/A')}/100
        """
        
        if news_data:
            query += f"""
            News Sentiment:
            - Overall Sentiment: {news_data.get('sentiment', 'N/A')}
            - Recent Headlines: {', '.join(news_data.get('headlines', ['N/A'])[:3])}
            - News Score: {news_data.get('score', 'N/A')}/100
            """
        
        if social_data:
            query += f"""
            Social Media Sentiment:
            - Overall Sentiment: {social_data.get('sentiment', 'N/A')}
            - Mention Volume: {social_data.get('mention_volume', 'N/A')}
            - Social Score: {social_data.get('score', 'N/A')}/100
            """
        
        query += """
        Based on this information, please provide:
        1. A buy/hold/avoid recommendation
        2. A confidence score from 0-100
        3. 2-3 key insights supporting your recommendation
        4. Any potential risks to be aware of
        
        Format your response as JSON with the following structure:
        {
            "recommendation": "buy|hold|avoid",
            "score": 75,
            "insights": ["insight1", "insight2", "insight3"],
            "risks": ["risk1", "risk2"]
        }
        """
        
        return query
    
    def _prepare_position_sale_query(self, symbol: str, position_data: Dict,
                                    technical_analysis: Dict, news_data: Optional[Dict] = None,
                                    social_data: Optional[Dict] = None) -> str:
        """Prepare a query for position sale analysis."""
        query = f"""
        You are an AI financial advisor helping with stock trading decisions. Please analyze the following position for a potential sale:
        
        Symbol: {symbol}
        
        Position Data:
        - Entry Price: ${position_data.get('entry_price', 'N/A')}
        - Current Price: ${position_data.get('current_price', 'N/A')}
        - Profit/Loss: {position_data.get('profit_loss_pct', 'N/A')}%
        - Hold Time: {position_data.get('hold_time', 'N/A')} days
        
        Technical Analysis:
        - SMA(50): ${technical_analysis.get('sma_50', 'N/A')}
        - SMA(200): ${technical_analysis.get('sma_200', 'N/A')}
        - RSI: {technical_analysis.get('rsi', 'N/A')}
        - MACD: {technical_analysis.get('macd', 'N/A')}
        - Signal: {technical_analysis.get('macd_signal', 'N/A')}
        - Bollinger Bands: Upper=${technical_analysis.get('bb_upper', 'N/A')}, Middle=${technical_analysis.get('bb_middle', 'N/A')}, Lower=${technical_analysis.get('bb_lower', 'N/A')}
        - Technical Score: {technical_analysis.get('score', 'N/A')}/100
        """
        
        if news_data:
            query += f"""
            News Sentiment:
            - Overall Sentiment: {news_data.get('sentiment', 'N/A')}
            - Recent Headlines: {', '.join(news_data.get('headlines', ['N/A'])[:3])}
            - News Score: {news_data.get('score', 'N/A')}/100
            """
        
        if social_data:
            query += f"""
            Social Media Sentiment:
            - Overall Sentiment: {social_data.get('sentiment', 'N/A')}
            - Mention Volume: {social_data.get('mention_volume', 'N/A')}
            - Social Score: {social_data.get('score', 'N/A')}/100
            """
        
        query += """
        Based on this information, please provide:
        1. A sell/hold recommendation
        2. A confidence score from 0-100
        3. 2-3 key insights supporting your recommendation
        
        Format your response as JSON with the following structure:
        {
            "recommendation": "sell|hold",
            "score": 75,
            "insights": ["insight1", "insight2", "insight3"]
        }
        """
        
        return query
    
    def _prepare_market_conditions_query(self, market_data: Dict, economic_indicators: Dict,
                                        sector_performance: Dict) -> str:
        """Prepare a query for market conditions analysis."""
        query = f"""
        You are an AI financial advisor helping with market analysis. Please analyze the following market conditions:
        
        Market Data:
        - S&P 500: {market_data.get('sp500', 'N/A')} ({market_data.get('sp500_change', 'N/A')}%)
        - Dow Jones: {market_data.get('dow', 'N/A')} ({market_data.get('dow_change', 'N/A')}%)
        - Nasdaq: {market_data.get('nasdaq', 'N/A')} ({market_data.get('nasdaq_change', 'N/A')}%)
        - VIX: {market_data.get('vix', 'N/A')}
        - 10Y Treasury: {market_data.get('treasury_10y', 'N/A')}%
        
        Economic Indicators:
        - GDP Growth: {economic_indicators.get('gdp_growth', 'N/A')}%
        - Unemployment: {economic_indicators.get('unemployment', 'N/A')}%
        - Inflation: {economic_indicators.get('inflation', 'N/A')}%
        - Fed Funds Rate: {economic_indicators.get('fed_rate', 'N/A')}%
        
        Top Performing Sectors:
        """
        
        # Add top 3 and bottom 3 sectors
        sectors = sorted(sector_performance.items(), key=lambda x: x[1], reverse=True)
        for sector, performance in sectors[:3]:
            query += f"- {sector}: {performance}%\n"
        
        query += "\nWorst Performing Sectors:\n"
        for sector, performance in sectors[-3:]:
            query += f"- {sector}: {performance}%\n"
        
        query += """
        Based on this information, please provide:
        1. An overall market outlook (bullish/neutral/bearish)
        2. A confidence score from 0-100
        3. 2-3 key insights about current market conditions
        4. Recommended sectors to focus on
        
        Format your response as JSON with the following structure:
        {
            "outlook": "bullish|neutral|bearish",
            "score": 75,
            "insights": ["insight1", "insight2", "insight3"],
            "recommended_sectors": ["sector1", "sector2"]
        }
        """
        
        return query
    
    def _prepare_news_batch_query(self, news_by_symbol: Dict[str, List[Dict]]) -> str:
        """Prepare a query for batch news sentiment analysis."""
        query = """
        You are an AI financial advisor helping with news sentiment analysis for multiple stocks. Please analyze the following news items and provide sentiment analysis for each stock:
        
        """
        
        for symbol, news_items in news_by_symbol.items():
            query += f"\nStock: {symbol}\nNews Items:\n"
            for i, item in enumerate(news_items[:5]):  # Limit to 5 news items per symbol
                query += f"{i+1}. {item.get('title', 'N/A')}: {item.get('summary', 'N/A')[:200]}...\n"
        
        query += """
        For each stock, please provide:
        1. Overall sentiment (positive/neutral/negative)
        2. A sentiment score from 0-100 (0 = very negative, 100 = very positive)
        3. 1-2 key insights from the news
        
        Format your response as JSON with the following structure:
        {
            "sentiments": {
                "AAPL": {
                    "sentiment": "positive",
                    "score": 75,
                    "insights": ["insight1", "insight2"]
                },
                "MSFT": {
                    "sentiment": "neutral",
                    "score": 50,
                    "insights": ["insight1", "insight2"]
                }
            }
        }
        """
        
        return query
    
    def _process_claude_response(self, response: Dict, analysis_type: str) -> Dict:
        """
        Process Claude's response into a structured format.
        
        Args:
            response: Raw API response
            analysis_type: Type of analysis performed
            
        Returns:
            Structured analysis results
        """
        if not response or 'content' not in response:
            logger.error(f"Invalid response format: {response}")
            return {
                "success": False,
                "error": "Invalid API response",
                "recommendation": "neutral",
                "score": 50,
                "insights": []
            }
        
        try:
            # Extract the text content
            content = response['content'][0]['text']
            
            # Find JSON in the response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                result = json.loads(json_str)
                result["success"] = True
                return result
            else:
                logger.warning(f"No JSON found in response: {content[:100]}...")
                return {
                    "success": False,
                    "error": "No JSON found in response",
                    "recommendation": "neutral",
                    "score": 50,
                    "insights": []
                }
                
        except Exception as e:
            logger.error(f"Error processing Claude response: {e}")
            return {
                "success": False,
                "error": str(e),
                "recommendation": "neutral",
                "score": 50,
                "insights": []
            }
    
    def _store_analysis(self, symbol: str, analysis_type: str, analysis: Dict) -> None:
        """
        Store analysis results in the database.
        
        Args:
            symbol: Stock symbol
            analysis_type: Type of analysis
            analysis: Analysis results
        """
        try:
            # Convert analysis to JSON string
            analysis_json = json.dumps(analysis)
            
            # Extract score and recommendation
            score = analysis.get('score', 50)
            recommendation = analysis.get('recommendation', analysis.get('outlook', 'neutral'))
            
            # Insert into database
            self.cursor.execute(
                '''
                INSERT INTO ai_analysis 
                (symbol, analysis_type, analysis_data, score, recommendation)
                VALUES (?, ?, ?, ?, ?)
                ''',
                (symbol, analysis_type, analysis_json, score, recommendation)
            )
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Error storing analysis: {e}")
    
    def get_recent_analysis(self, symbol: str, analysis_type: str, 
                           max_age_minutes: int = 60) -> Optional[Dict]:
        """
        Get recent analysis for a symbol if available.
        
        Args:
            symbol: Stock symbol
            analysis_type: Type of analysis
            max_age_minutes: Maximum age of analysis in minutes
            
        Returns:
            Recent analysis or None if not available
        """
        try:
            # Calculate cutoff time
            cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)
            cutoff_str = cutoff_time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Query database
            self.cursor.execute(
                '''
                SELECT analysis_data FROM ai_analysis
                WHERE symbol = ? AND analysis_type = ? AND timestamp > ?
                ORDER BY timestamp DESC LIMIT 1
                ''',
                (symbol, analysis_type, cutoff_str)
            )
            
            row = self.cursor.fetchone()
            
            if row:
                return json.loads(row[0])
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting recent analysis: {e}")
            return None
    
    def close(self) -> None:
        """Close connections and shut down."""
        if self.conn:
            self.conn.close()
        
        self.batch_manager.shutdown()
        logger.info("AI Integration shut down")


# Example usage
if __name__ == "__main__":
    # Create AI integration
    ai = AIIntegration()
    
    # Example stock data
    stock_data = {
        "price": 150.25,
        "52_week_low": 120.50,
        "52_week_high": 175.75,
        "volume": 35000000,
        "market_cap": "2.5T",
        "pe_ratio": 28.5
    }
    
    # Example technical analysis
    technical_analysis = {
        "sma_50": 148.30,
        "sma_200": 145.20,
        "rsi": 62,
        "macd": 2.5,
        "macd_signal": 1.8,
        "bb_upper": 155.20,
        "bb_middle": 150.10,
        "bb_lower": 145.00,
        "score": 65
    }
    
    # Example news data
    news_data = {
        "sentiment": "positive",
        "headlines": [
            "Company announces new product line",
            "Quarterly earnings exceed expectations",
            "Expansion into new markets planned"
        ],
        "score": 70
    }
    
    # Analyze stock for purchase
    analysis = ai.analyze_stock_for_purchase("AAPL", stock_data, technical_analysis, news_data)
    
    print("Analysis result:")
    print(json.dumps(analysis, indent=2))
    
    # Close connections
    ai.close()
