import pandas as pd
import numpy as np
import talib
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tradebot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('AdvancedTA')

@dataclass
class TechnicalIndicators:
    """Class to hold technical indicator values for a stock"""
    # Trend indicators
    sma_20: float = 0
    sma_50: float = 0
    sma_200: float = 0
    ema_9: float = 0
    ema_21: float = 0
    macd: float = 0
    macd_signal: float = 0
    macd_hist: float = 0
    adx: float = 0
    
    # Momentum indicators
    rsi: float = 0
    stoch_k: float = 0
    stoch_d: float = 0
    cci: float = 0
    
    # Volatility indicators
    bbands_upper: float = 0
    bbands_middle: float = 0
    bbands_lower: float = 0
    atr: float = 0
    
    # Volume indicators
    obv: float = 0
    vwap: float = 0
    
    # Price action
    support_level: float = 0
    resistance_level: float = 0
    distance_from_high: float = 0
    distance_from_low: float = 0

class AdvancedTechnicalAnalysis:
    def __init__(self):
        """Initialize the Advanced Technical Analysis component"""
        logger.info("Advanced Technical Analysis component initialized")
    
    def calculate_indicators(self, df):
        """Calculate all technical indicators for a given dataframe of price data
        
        Args:
            df: DataFrame with OHLCV data (open, high, low, close, volume)
            
        Returns:
            TechnicalIndicators object with calculated values
        """
        if df is None or len(df) < 200:
            logger.warning("Not enough data to calculate indicators (need at least 200 bars)")
            return None
        
        try:
            indicators = TechnicalIndicators()
            
            # Make sure we have the required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    logger.error(f"Missing required column: {col}")
                    return None
            
            # Convert to numpy arrays for talib
            open_prices = df['open'].values
            high_prices = df['high'].values
            low_prices = df['low'].values
            close_prices = df['close'].values
            volume = df['volume'].values
            
            # Calculate trend indicators
            indicators.sma_20 = talib.SMA(close_prices, timeperiod=20)[-1]
            indicators.sma_50 = talib.SMA(close_prices, timeperiod=50)[-1]
            indicators.sma_200 = talib.SMA(close_prices, timeperiod=200)[-1]
            indicators.ema_9 = talib.EMA(close_prices, timeperiod=9)[-1]
            indicators.ema_21 = talib.EMA(close_prices, timeperiod=21)[-1]
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(
                close_prices, fastperiod=12, slowperiod=26, signalperiod=9
            )
            indicators.macd = macd[-1]
            indicators.macd_signal = macd_signal[-1]
            indicators.macd_hist = macd_hist[-1]
            
            # ADX - Average Directional Index
            indicators.adx = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)[-1]
            
            # Momentum indicators
            indicators.rsi = talib.RSI(close_prices, timeperiod=14)[-1]
            
            # Stochastic
            stoch_k, stoch_d = talib.STOCH(
                high_prices, low_prices, close_prices, 
                fastk_period=14, slowk_period=3, slowk_matype=0, 
                slowd_period=3, slowd_matype=0
            )
            indicators.stoch_k = stoch_k[-1]
            indicators.stoch_d = stoch_d[-1]
            
            # CCI - Commodity Channel Index
            indicators.cci = talib.CCI(high_prices, low_prices, close_prices, timeperiod=14)[-1]
            
            # Volatility indicators
            upper, middle, lower = talib.BBANDS(
                close_prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
            )
            indicators.bbands_upper = upper[-1]
            indicators.bbands_middle = middle[-1]
            indicators.bbands_lower = lower[-1]
            
            # ATR - Average True Range
            indicators.atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)[-1]
            
            # Volume indicators
            indicators.obv = talib.OBV(close_prices, volume)[-1]
            
            # VWAP - Volume Weighted Average Price (custom calculation)
            typical_price = (high_prices + low_prices + close_prices) / 3
            indicators.vwap = np.sum(typical_price[-20:] * volume[-20:]) / np.sum(volume[-20:])
            
            # Support and resistance levels (simple implementation)
            recent_lows = pd.Series(low_prices[-30:]).nsmallest(3).mean()
            recent_highs = pd.Series(high_prices[-30:]).nlargest(3).mean()
            indicators.support_level = recent_lows
            indicators.resistance_level = recent_highs
            
            # Distance from highs and lows
            all_time_high = np.max(high_prices)
            all_time_low = np.min(low_prices)
            current_price = close_prices[-1]
            
            indicators.distance_from_high = (all_time_high - current_price) / current_price * 100
            indicators.distance_from_low = (current_price - all_time_low) / all_time_low * 100
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return None
    
    def detect_patterns(self, df):
        """Detect chart patterns in the price data
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary of detected patterns and their strengths
        """
        if df is None or len(df) < 30:
            return {}
        
        patterns = {}
        
        try:
            # Convert to numpy arrays for talib
            open_prices = df['open'].values
            high_prices = df['high'].values
            low_prices = df['low'].values
            close_prices = df['close'].values
            
            # Candlestick patterns
            # Bullish patterns
            patterns['hammer'] = talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)[-1]
            patterns['inverted_hammer'] = talib.CDLINVERTEDHAMMER(open_prices, high_prices, low_prices, close_prices)[-1]
            patterns['engulfing_bullish'] = talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)[-1]
            patterns['morning_star'] = talib.CDLMORNINGSTAR(open_prices, high_prices, low_prices, close_prices)[-1]
            patterns['piercing'] = talib.CDLPIERCING(open_prices, high_prices, low_prices, close_prices)[-1]
            
            # Bearish patterns
            patterns['hanging_man'] = talib.CDLHANGINGMAN(open_prices, high_prices, low_prices, close_prices)[-1]
            patterns['shooting_star'] = talib.CDLSHOOTINGSTAR(open_prices, high_prices, low_prices, close_prices)[-1]
            patterns['engulfing_bearish'] = -1 * (talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)[-1] < 0)
            patterns['evening_star'] = talib.CDLEVENINGSTAR(open_prices, high_prices, low_prices, close_prices)[-1]
            patterns['dark_cloud_cover'] = talib.CDLDARKCLOUDCOVER(open_prices, high_prices, low_prices, close_prices)[-1]
            
            # Filter out patterns that weren't detected (value = 0)
            patterns = {k: v for k, v in patterns.items() if v != 0}
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            return {}
    
    def score_technical_indicators(self, indicators):
        """Score a stock based on its technical indicators
        
        Args:
            indicators: TechnicalIndicators object
            
        Returns:
            score: int, technical score from 0-100
            signals: dict, bullish and bearish signals
        """
        if indicators is None:
            return 0, {'bullish': [], 'bearish': []}
        
        score = 50  # Start at neutral
        bullish_signals = []
        bearish_signals = []
        
        # Trend analysis
        current_price = indicators.bbands_middle  # Use middle bollinger band as current price approximation
        
        # Moving averages
        if indicators.sma_20 > indicators.sma_50:
            score += 5
            bullish_signals.append("SMA20 above SMA50 (bullish)")
        else:
            score -= 5
            bearish_signals.append("SMA20 below SMA50 (bearish)")
            
        if indicators.sma_50 > indicators.sma_200:
            score += 5
            bullish_signals.append("SMA50 above SMA200 (golden cross)")
        else:
            score -= 5
            bearish_signals.append("SMA50 below SMA200 (death cross)")
            
        # Price relative to moving averages
        if current_price > indicators.sma_20:
            score += 3
            bullish_signals.append("Price above SMA20")
        else:
            score -= 3
            bearish_signals.append("Price below SMA20")
            
        if current_price > indicators.sma_50:
            score += 3
            bullish_signals.append("Price above SMA50")
        else:
            score -= 3
            bearish_signals.append("Price below SMA50")
            
        # MACD
        if indicators.macd > indicators.macd_signal:
            score += 5
            bullish_signals.append("MACD above signal line")
        else:
            score -= 5
            bearish_signals.append("MACD below signal line")
            
        if indicators.macd_hist > 0 and indicators.macd_hist > 0:
            score += 3
            bullish_signals.append("MACD histogram positive and increasing")
        elif indicators.macd_hist < 0:
            score -= 3
            bearish_signals.append("MACD histogram negative")
            
        # ADX (trend strength)
        if indicators.adx > 25:
            score += 3
            bullish_signals.append("Strong trend (ADX > 25)")
        elif indicators.adx < 20:
            score -= 2
            bearish_signals.append("Weak trend (ADX < 20)")
            
        # RSI
        if 40 <= indicators.rsi <= 60:
            # Neutral
            pass
        elif 30 <= indicators.rsi < 40:
            score -= 2
            bearish_signals.append("RSI showing weakness")
        elif 60 < indicators.rsi <= 70:
            score += 2
            bullish_signals.append("RSI showing strength")
        elif indicators.rsi < 30:
            score += 5  # Oversold, potential reversal
            bullish_signals.append("RSI oversold (potential buy)")
        elif indicators.rsi > 70:
            score -= 5  # Overbought, potential reversal
            bearish_signals.append("RSI overbought (potential sell)")
            
        # Stochastic
        if indicators.stoch_k < 20 and indicators.stoch_k > indicators.stoch_d:
            score += 5
            bullish_signals.append("Stochastic oversold with bullish crossover")
        elif indicators.stoch_k > 80 and indicators.stoch_k < indicators.stoch_d:
            score -= 5
            bearish_signals.append("Stochastic overbought with bearish crossover")
            
        # Bollinger Bands
        bb_width = (indicators.bbands_upper - indicators.bbands_lower) / indicators.bbands_middle
        
        if current_price < indicators.bbands_lower:
            score += 5
            bullish_signals.append("Price below lower Bollinger Band (oversold)")
        elif current_price > indicators.bbands_upper:
            score -= 5
            bearish_signals.append("Price above upper Bollinger Band (overbought)")
            
        if bb_width < 0.1:  # Narrow bands, potential breakout
            score += 3
            bullish_signals.append("Narrow Bollinger Bands (potential breakout)")
            
        # Support and resistance
        if abs(current_price - indicators.support_level) / current_price < 0.03:
            score += 5
            bullish_signals.append("Price near support level")
        elif abs(current_price - indicators.resistance_level) / current_price < 0.03:
            score -= 5
            bearish_signals.append("Price near resistance level")
            
        # Volume indicators
        # OBV analysis would need historical values to determine trend
        
        # Ensure score is within 0-100 range
        score = max(0, min(100, score))
        
        signals = {
            'bullish': bullish_signals,
            'bearish': bearish_signals
        }
        
        return score, signals
    
    def analyze_stock(self, df):
        """Complete technical analysis of a stock
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            dict with analysis results
        """
        if df is None or len(df) < 200:
            return {
                'score': 0,
                'signals': {'bullish': [], 'bearish': ['Insufficient data']},
                'patterns': {},
                'recommendation': 'avoid'
            }
        
        # Calculate indicators
        indicators = self.calculate_indicators(df)
        
        # Detect patterns
        patterns = self.detect_patterns(df)
        
        # Score based on indicators
        score, signals = self.score_technical_indicators(indicators)
        
        # Adjust score based on patterns
        pattern_score = 0
        for pattern, value in patterns.items():
            if 'bullish' in pattern or value > 0:
                pattern_score += 5
                signals['bullish'].append(f"Bullish pattern: {pattern}")
            elif 'bearish' in pattern or value < 0:
                pattern_score -= 5
                signals['bearish'].append(f"Bearish pattern: {pattern}")
        
        # Cap pattern adjustment at +/- 15 points
        pattern_score = max(-15, min(15, pattern_score))
        score += pattern_score
        
        # Ensure score is within 0-100 range
        score = max(0, min(100, score))
        
        # Generate recommendation
        recommendation = 'neutral'
        if score >= 70:
            recommendation = 'strong_buy'
        elif score >= 60:
            recommendation = 'buy'
        elif score <= 30:
            recommendation = 'avoid'
        elif score <= 40:
            recommendation = 'weak'
        
        return {
            'score': score,
            'signals': signals,
            'patterns': patterns,
            'recommendation': recommendation,
            'indicators': indicators
        }


# Test the advanced technical analysis if run directly
if __name__ == "__main__":
    import yfinance as yf
    
    # Get sample data
    ticker = "AAPL"
    data = yf.download(ticker, period="1y")
    
    # Rename columns to match our expected format
    data.columns = [col.lower() for col in data.columns]
    data = data.rename(columns={'adj close': 'close'})
    
    # Create analyzer
    analyzer = AdvancedTechnicalAnalysis()
    
    # Analyze
    analysis = analyzer.analyze_stock(data)
    
    # Print results
    print(f"Technical Analysis for {ticker}:")
    print(f"Score: {analysis['score']}")
    print(f"Recommendation: {analysis['recommendation']}")
    print("\nBullish Signals:")
    for signal in analysis['signals']['bullish']:
        print(f"- {signal}")
    print("\nBearish Signals:")
    for signal in analysis['signals']['bearish']:
        print(f"- {signal}")
    print("\nDetected Patterns:")
    for pattern, value in analysis['patterns'].items():
        print(f"- {pattern}: {value}")
