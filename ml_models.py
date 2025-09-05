import os
import sqlite3
import logging
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score
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
logger = logging.getLogger('MLModels')

# Load environment variables
load_dotenv()

class MLModels:
    def __init__(self):
        """Initialize the Machine Learning Models component"""
        # Load Alpaca API credentials for market data
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.api_secret = os.getenv('ALPACA_API_SECRET')
        self.api_endpoint = os.getenv('ALPACA_API_ENDPOINT')
        
        if all([self.api_key, self.api_secret, self.api_endpoint]):
            # Initialize Alpaca API
            self.api = tradeapi.REST(
                self.api_key,
                self.api_secret,
                base_url=self.api_endpoint
            )
        else:
            logger.warning("Alpaca API credentials not found in .env file")
            self.api = None
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Initialize models
        self.price_prediction_model = None
        self.direction_prediction_model = None
        self.volatility_prediction_model = None
        self.feature_scaler = None
        
        # Load existing models if available
        self.load_models()
        
        logger.info("MLModels component initialized")
    
    def load_models(self):
        """Load trained models from disk"""
        try:
            if os.path.exists('models/price_prediction_model.joblib'):
                self.price_prediction_model = joblib.load('models/price_prediction_model.joblib')
                logger.info("Price prediction model loaded")
            
            if os.path.exists('models/direction_prediction_model.joblib'):
                self.direction_prediction_model = joblib.load('models/direction_prediction_model.joblib')
                logger.info("Direction prediction model loaded")
            
            if os.path.exists('models/volatility_prediction_model.joblib'):
                self.volatility_prediction_model = joblib.load('models/volatility_prediction_model.joblib')
                logger.info("Volatility prediction model loaded")
            
            if os.path.exists('models/feature_scaler.joblib'):
                self.feature_scaler = joblib.load('models/feature_scaler.joblib')
                logger.info("Feature scaler loaded")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def save_models(self):
        """Save trained models to disk"""
        try:
            if self.price_prediction_model:
                joblib.dump(self.price_prediction_model, 'models/price_prediction_model.joblib')
            
            if self.direction_prediction_model:
                joblib.dump(self.direction_prediction_model, 'models/direction_prediction_model.joblib')
            
            if self.volatility_prediction_model:
                joblib.dump(self.volatility_prediction_model, 'models/volatility_prediction_model.joblib')
            
            if self.feature_scaler:
                joblib.dump(self.feature_scaler, 'models/feature_scaler.joblib')
            
            logger.info("Models saved to disk")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def get_historical_data(self, symbol, days=365):
        """Get historical price data for a stock
        
        Args:
            symbol: Stock symbol
            days: Number of days of historical data to retrieve
            
        Returns:
            DataFrame: Historical price data
        """
        if not self.api:
            logger.error("Alpaca API not initialized")
            return None
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Format dates for Alpaca API
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Get daily bars
            bars = self.api.get_barset(symbol, 'day', start=start_str, end=end_str)
            
            if symbol not in bars or len(bars[symbol]) == 0:
                logger.warning(f"No historical data found for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame()
            symbol_bars = bars[symbol]
            
            df['date'] = [bar.t for bar in symbol_bars]
            df['open'] = [bar.o for bar in symbol_bars]
            df['high'] = [bar.h for bar in symbol_bars]
            df['low'] = [bar.l for bar in symbol_bars]
            df['close'] = [bar.c for bar in symbol_bars]
            df['volume'] = [bar.v for bar in symbol_bars]
            
            # Set date as index
            df.set_index('date', inplace=True)
            
            logger.info(f"Retrieved {len(df)} days of historical data for {symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return None
    
    def prepare_features(self, df):
        """Prepare features for machine learning models
        
        Args:
            df: DataFrame with historical price data
            
        Returns:
            DataFrame: Features for machine learning
        """
        if df is None or len(df) < 30:
            logger.warning("Not enough data to prepare features")
            return None
        
        try:
            # Create a copy to avoid modifying the original
            data = df.copy()
            
            # Calculate returns
            data['return'] = data['close'].pct_change()
            data['return_1d'] = data['return'].shift(-1)  # Next day's return (target)
            
            # Price features
            data['price_sma_5'] = data['close'].rolling(window=5).mean()
            data['price_sma_10'] = data['close'].rolling(window=10).mean()
            data['price_sma_20'] = data['close'].rolling(window=20).mean()
            data['price_sma_50'] = data['close'].rolling(window=50).mean()
            
            # Price ratios
            data['price_ratio_5'] = data['close'] / data['price_sma_5']
            data['price_ratio_10'] = data['close'] / data['price_sma_10']
            data['price_ratio_20'] = data['close'] / data['price_sma_20']
            data['price_ratio_50'] = data['close'] / data['price_sma_50']
            
            # Volatility features
            data['volatility_5'] = data['return'].rolling(window=5).std()
            data['volatility_10'] = data['return'].rolling(window=10).std()
            data['volatility_20'] = data['return'].rolling(window=20).std()
            
            # Volume features
            data['volume_sma_5'] = data['volume'].rolling(window=5).mean()
            data['volume_sma_10'] = data['volume'].rolling(window=10).mean()
            data['volume_ratio_5'] = data['volume'] / data['volume_sma_5']
            data['volume_ratio_10'] = data['volume'] / data['volume_sma_10']
            
            # Momentum features
            data['momentum_5'] = data['close'].pct_change(periods=5)
            data['momentum_10'] = data['close'].pct_change(periods=10)
            data['momentum_20'] = data['close'].pct_change(periods=20)
            
            # Mean reversion features
            data['mean_reversion_5'] = (data['close'] - data['price_sma_5']) / data['price_sma_5']
            data['mean_reversion_10'] = (data['close'] - data['price_sma_10']) / data['price_sma_10']
            
            # High-Low range
            data['high_low_ratio'] = data['high'] / data['low']
            data['high_low_ratio_5'] = data['high_low_ratio'].rolling(window=5).mean()
            
            # Direction features (for classification)
            data['direction'] = np.where(data['return_1d'] > 0, 1, 0)
            
            # Drop NaN values
            data.dropna(inplace=True)
            
            return data
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None
    
    def train_price_prediction_model(self, symbol, days=365):
        """Train a model to predict next day's price
        
        Args:
            symbol: Stock symbol
            days: Number of days of historical data to use
            
        Returns:
            bool: True if training was successful, False otherwise
        """
        # Get historical data
        df = self.get_historical_data(symbol, days)
        if df is None:
            return False
        
        # Prepare features
        data = self.prepare_features(df)
        if data is None:
            return False
        
        try:
            # Define features and target
            feature_columns = [
                'price_ratio_5', 'price_ratio_10', 'price_ratio_20', 'price_ratio_50',
                'volatility_5', 'volatility_10', 'volatility_20',
                'volume_ratio_5', 'volume_ratio_10',
                'momentum_5', 'momentum_10', 'momentum_20',
                'mean_reversion_5', 'mean_reversion_10',
                'high_low_ratio', 'high_low_ratio_5'
            ]
            
            X = data[feature_columns].values
            y = data['return_1d'].values
            
            # Scale features
            self.feature_scaler = StandardScaler()
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Train model
            self.price_prediction_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.price_prediction_model.fit(X_scaled, y)
            
            # Evaluate model
            y_pred = self.price_prediction_model.predict(X_scaled)
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            
            logger.info(f"Price prediction model trained for {symbol} with RMSE: {rmse:.6f}")
            
            # Save models
            self.save_models()
            
            return True
            
        except Exception as e:
            logger.error(f"Error training price prediction model: {e}")
            return False
    
    def train_direction_prediction_model(self, symbol, days=365):
        """Train a model to predict price direction (up/down)
        
        Args:
            symbol: Stock symbol
            days: Number of days of historical data to use
            
        Returns:
            bool: True if training was successful, False otherwise
        """
        # Get historical data
        df = self.get_historical_data(symbol, days)
        if df is None:
            return False
        
        # Prepare features
        data = self.prepare_features(df)
        if data is None:
            return False
        
        try:
            # Define features and target
            feature_columns = [
                'price_ratio_5', 'price_ratio_10', 'price_ratio_20', 'price_ratio_50',
                'volatility_5', 'volatility_10', 'volatility_20',
                'volume_ratio_5', 'volume_ratio_10',
                'momentum_5', 'momentum_10', 'momentum_20',
                'mean_reversion_5', 'mean_reversion_10',
                'high_low_ratio', 'high_low_ratio_5'
            ]
            
            X = data[feature_columns].values
            y = data['direction'].values
            
            # Scale features
            if self.feature_scaler is None:
                self.feature_scaler = StandardScaler()
                X_scaled = self.feature_scaler.fit_transform(X)
            else:
                X_scaled = self.feature_scaler.transform(X)
            
            # Train model
            self.direction_prediction_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.direction_prediction_model.fit(X_scaled, y)
            
            # Evaluate model
            y_pred = self.direction_prediction_model.predict(X_scaled)
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred)
            recall = recall_score(y, y_pred)
            
            logger.info(f"Direction prediction model trained for {symbol} with accuracy: {accuracy:.4f}, precision: {precision:.4f}, recall: {recall:.4f}")
            
            # Save models
            self.save_models()
            
            return True
            
        except Exception as e:
            logger.error(f"Error training direction prediction model: {e}")
            return False
    
    def train_volatility_prediction_model(self, symbol, days=365):
        """Train a model to predict next day's volatility
        
        Args:
            symbol: Stock symbol
            days: Number of days of historical data to use
            
        Returns:
            bool: True if training was successful, False otherwise
        """
        # Get historical data
        df = self.get_historical_data(symbol, days)
        if df is None:
            return False
        
        # Prepare features
        data = self.prepare_features(df)
        if data is None:
            return False
        
        try:
            # Define features and target
            feature_columns = [
                'price_ratio_5', 'price_ratio_10', 'price_ratio_20',
                'volatility_5', 'volatility_10', 'volatility_20',
                'volume_ratio_5', 'volume_ratio_10',
                'momentum_5', 'momentum_10',
                'high_low_ratio', 'high_low_ratio_5'
            ]
            
            # Calculate next day's volatility (absolute return)
            data['next_volatility'] = data['return_1d'].abs()
            
            X = data[feature_columns].values
            y = data['next_volatility'].values
            
            # Scale features
            if self.feature_scaler is None:
                self.feature_scaler = StandardScaler()
                X_scaled = self.feature_scaler.fit_transform(X)
            else:
                X_scaled = self.feature_scaler.transform(X)
            
            # Train model
            self.volatility_prediction_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.volatility_prediction_model.fit(X_scaled, y)
            
            # Evaluate model
            y_pred = self.volatility_prediction_model.predict(X_scaled)
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            
            logger.info(f"Volatility prediction model trained for {symbol} with RMSE: {rmse:.6f}")
            
            # Save models
            self.save_models()
            
            return True
            
        except Exception as e:
            logger.error(f"Error training volatility prediction model: {e}")
            return False
    
    def predict_next_day(self, symbol):
        """Predict next day's price movement for a stock
        
        Args:
            symbol: Stock symbol
            
        Returns:
            dict: Prediction results
        """
        # Check if models are trained
        if self.price_prediction_model is None or self.direction_prediction_model is None:
            logger.warning("Models not trained yet")
            return None
        
        # Get latest data
        df = self.get_historical_data(symbol, days=60)  # Get 60 days of data
        if df is None:
            return None
        
        # Prepare features
        data = self.prepare_features(df)
        if data is None:
            return None
        
        try:
            # Define features
            feature_columns = [
                'price_ratio_5', 'price_ratio_10', 'price_ratio_20', 'price_ratio_50',
                'volatility_5', 'volatility_10', 'volatility_20',
                'volume_ratio_5', 'volume_ratio_10',
                'momentum_5', 'momentum_10', 'momentum_20',
                'mean_reversion_5', 'mean_reversion_10',
                'high_low_ratio', 'high_low_ratio_5'
            ]
            
            # Get latest features
            latest_features = data[feature_columns].iloc[-1].values.reshape(1, -1)
            
            # Scale features
            if self.feature_scaler is None:
                logger.warning("Feature scaler not available")
                return None
            
            latest_features_scaled = self.feature_scaler.transform(latest_features)
            
            # Make predictions
            predicted_return = self.price_prediction_model.predict(latest_features_scaled)[0]
            predicted_direction = self.direction_prediction_model.predict(latest_features_scaled)[0]
            predicted_direction_prob = self.direction_prediction_model.predict_proba(latest_features_scaled)[0]
            
            # Get current price
            current_price = df['close'].iloc[-1]
            
            # Calculate predicted price
            predicted_price = current_price * (1 + predicted_return)
            
            # Get prediction confidence
            direction_confidence = predicted_direction_prob[1] if predicted_direction == 1 else predicted_direction_prob[0]
            
            # Predict volatility if model is available
            predicted_volatility = None
            if self.volatility_prediction_model:
                predicted_volatility = self.volatility_prediction_model.predict(latest_features_scaled)[0]
            
            # Create prediction result
            prediction = {
                'symbol': symbol,
                'current_price': current_price,
                'predicted_return': predicted_return * 100,  # Convert to percentage
                'predicted_price': predicted_price,
                'predicted_direction': 'up' if predicted_direction == 1 else 'down',
                'direction_confidence': direction_confidence * 100,  # Convert to percentage
                'predicted_volatility': predicted_volatility * 100 if predicted_volatility is not None else None  # Convert to percentage
            }
            
            logger.info(f"Prediction for {symbol}: {prediction['predicted_direction']} ({prediction['direction_confidence']:.1f}% confidence), return: {prediction['predicted_return']:.2f}%")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None
    
    def train_all_models(self, symbol, days=365):
        """Train all prediction models for a stock
        
        Args:
            symbol: Stock symbol
            days: Number of days of historical data to use
            
        Returns:
            bool: True if all training was successful, False otherwise
        """
        logger.info(f"Training all models for {symbol}...")
        
        # Train price prediction model
        price_success = self.train_price_prediction_model(symbol, days)
        
        # Train direction prediction model
        direction_success = self.train_direction_prediction_model(symbol, days)
        
        # Train volatility prediction model
        volatility_success = self.train_volatility_prediction_model(symbol, days)
        
        return price_success and direction_success and volatility_success
    
    def get_ml_score(self, symbol):
        """Get a trading score based on ML predictions
        
        Args:
            symbol: Stock symbol
            
        Returns:
            dict: ML score and insights
        """
        # Make prediction
        prediction = self.predict_next_day(symbol)
        
        if prediction is None:
            # Try to train models first
            logger.info(f"No models available for {symbol}, training now...")
            if self.train_all_models(symbol):
                # Try prediction again
                prediction = self.predict_next_day(symbol)
            
            if prediction is None:
                return {
                    'symbol': symbol,
                    'ml_score': 50,  # Neutral score
                    'confidence': 0,
                    'insights': ["Could not generate ML predictions"]
                }
        
        # Calculate ML score (0-100)
        # Base score is 50 (neutral)
        ml_score = 50
        
        # Adjust based on predicted return and direction
        if prediction['predicted_direction'] == 'up':
            # Scale from 50 to 100 based on return and confidence
            return_factor = min(1.0, prediction['predicted_return'] / 5.0)  # Cap at 5% return
            confidence_factor = prediction['direction_confidence'] / 100
            
            score_adjustment = 50 * return_factor * confidence_factor
            ml_score += score_adjustment
        else:
            # Scale from 50 to 0 based on return and confidence
            return_factor = min(1.0, abs(prediction['predicted_return']) / 5.0)  # Cap at 5% return
            confidence_factor = prediction['direction_confidence'] / 100
            
            score_adjustment = 50 * return_factor * confidence_factor
            ml_score -= score_adjustment
        
        # Generate insights
        insights = []
        
        insights.append(f"ML predicts {prediction['predicted_direction']} movement with {prediction['direction_confidence']:.1f}% confidence")
        insights.append(f"Expected return: {prediction['predicted_return']:.2f}%")
        
        if prediction['predicted_volatility'] is not None:
            insights.append(f"Expected volatility: {prediction['predicted_volatility']:.2f}%")
        
        # Add trading recommendation
        if ml_score >= 70:
            insights.append("ML recommendation: Strong Buy")
        elif ml_score >= 60:
            insights.append("ML recommendation: Buy")
        elif ml_score <= 30:
            insights.append("ML recommendation: Strong Sell")
        elif ml_score <= 40:
            insights.append("ML recommendation: Sell")
        else:
            insights.append("ML recommendation: Hold/Neutral")
        
        return {
            'symbol': symbol,
            'ml_score': ml_score,
            'confidence': prediction['direction_confidence'],
            'insights': insights
        }


# Test the ML models component if run directly
if __name__ == "__main__":
    ml_models = MLModels()
    
    # Test with a sample stock
    symbol = "AAPL"
    
    # Train models if needed
    if not ml_models.price_prediction_model or not ml_models.direction_prediction_model:
        print(f"Training models for {symbol}...")
        ml_models.train_all_models(symbol)
    
    # Get ML score
    score = ml_models.get_ml_score(symbol)
    
    print(f"ML Score for {symbol}: {score['ml_score']:.2f}/100 (Confidence: {score['confidence']:.1f}%)")
    
    print("\nInsights:")
    for insight in score['insights']:
        print(f"- {insight}")
