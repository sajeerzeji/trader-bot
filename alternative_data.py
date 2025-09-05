import os
import requests
import json
import sqlite3
import logging
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from datetime import datetime, timedelta
import re
import time
from bs4 import BeautifulSoup
import praw

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tradebot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('AlternativeData')

# Load environment variables
load_dotenv()

class AlternativeData:
    def __init__(self):
        """Initialize the Alternative Data component with API keys and settings"""
        # Load API keys
        self.alpha_vantage_key = os.getenv('ALPHAVANTAGE_API_KEY')
        self.newsapi_key = os.getenv('NEWSAPI_KEY')
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.reddit_user_agent = os.getenv('REDDIT_USER_AGENT', 'tradebot:v1.0 (by /u/yourusername)')
        
        # Initialize Reddit API if credentials are available
        self.reddit = None
        if all([self.reddit_client_id, self.reddit_client_secret]):
            try:
                self.reddit = praw.Reddit(
                    client_id=self.reddit_client_id,
                    client_secret=self.reddit_client_secret,
                    user_agent=self.reddit_user_agent
                )
                logger.info("Reddit API initialized")
            except Exception as e:
                logger.error(f"Error initializing Reddit API: {e}")
        
        # Create database table for alternative data if it doesn't exist
        self.init_database()
        
        logger.info("AlternativeData component initialized")
    
    def init_database(self):
        """Initialize database tables for alternative data"""
        try:
            conn = sqlite3.connect('tradebot.db')
            cursor = conn.cursor()
            
            # Create news_sentiment table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_sentiment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                title TEXT,
                source TEXT,
                url TEXT,
                published_at TIMESTAMP,
                sentiment_score REAL,
                relevance_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Create social_media_sentiment table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS social_media_sentiment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                platform TEXT,
                post_id TEXT,
                content TEXT,
                author TEXT,
                published_at TIMESTAMP,
                sentiment_score REAL,
                engagement_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Alternative data database tables initialized")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def get_news(self, symbol, days=3):
        """Get news articles for a stock symbol
        
        Args:
            symbol: Stock symbol
            days: Number of days to look back
            
        Returns:
            list: News articles
        """
        if not self.newsapi_key:
            logger.warning("NewsAPI key not found in .env file")
            return self.get_news_alpha_vantage(symbol)
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Format dates for NewsAPI
            from_param = start_date.strftime('%Y-%m-%d')
            to_param = end_date.strftime('%Y-%m-%d')
            
            # Make API request
            url = 'https://newsapi.org/v2/everything'
            params = {
                'q': f'${symbol} OR {symbol} stock',
                'from': from_param,
                'to': to_param,
                'language': 'en',
                'sortBy': 'relevancy',
                'apiKey': self.newsapi_key
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if response.status_code != 200:
                logger.error(f"Error getting news from NewsAPI: {data.get('message', 'Unknown error')}")
                return self.get_news_alpha_vantage(symbol)
            
            articles = data.get('articles', [])
            
            # Process articles
            processed_articles = []
            for article in articles[:10]:  # Limit to top 10 articles
                processed_articles.append({
                    'symbol': symbol,
                    'title': article.get('title'),
                    'source': article.get('source', {}).get('name'),
                    'url': article.get('url'),
                    'published_at': article.get('publishedAt'),
                    'description': article.get('description')
                })
            
            logger.info(f"Got {len(processed_articles)} news articles for {symbol} from NewsAPI")
            
            # Save to database
            self.save_news_to_db(processed_articles)
            
            return processed_articles
            
        except Exception as e:
            logger.error(f"Error getting news from NewsAPI: {e}")
            return self.get_news_alpha_vantage(symbol)
    
    def get_news_alpha_vantage(self, symbol):
        """Get news articles for a stock symbol using Alpha Vantage API
        
        Args:
            symbol: Stock symbol
            
        Returns:
            list: News articles
        """
        if not self.alpha_vantage_key:
            logger.warning("Alpha Vantage API key not found in .env file")
            return []
        
        try:
            # Make API request
            url = 'https://www.alphavantage.co/query'
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'apikey': self.alpha_vantage_key
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'feed' not in data:
                logger.error(f"Error getting news from Alpha Vantage: {data.get('Note', 'Unknown error')}")
                return []
            
            articles = data.get('feed', [])
            
            # Process articles
            processed_articles = []
            for article in articles:
                # Extract sentiment for this symbol
                sentiment_score = 0
                relevance_score = 0
                
                for ticker_sentiment in article.get('ticker_sentiment', []):
                    if ticker_sentiment.get('ticker') == symbol:
                        sentiment_score = float(ticker_sentiment.get('ticker_sentiment_score', 0))
                        relevance_score = float(ticker_sentiment.get('relevance_score', 0))
                        break
                
                processed_articles.append({
                    'symbol': symbol,
                    'title': article.get('title'),
                    'source': article.get('source'),
                    'url': article.get('url'),
                    'published_at': article.get('time_published'),
                    'sentiment_score': sentiment_score,
                    'relevance_score': relevance_score
                })
            
            logger.info(f"Got {len(processed_articles)} news articles for {symbol} from Alpha Vantage")
            
            # Save to database
            self.save_news_to_db(processed_articles)
            
            return processed_articles
            
        except Exception as e:
            logger.error(f"Error getting news from Alpha Vantage: {e}")
            return []
    
    def save_news_to_db(self, articles):
        """Save news articles to database
        
        Args:
            articles: List of news article dictionaries
        """
        if not articles:
            return
        
        try:
            conn = sqlite3.connect('tradebot.db')
            cursor = conn.cursor()
            
            for article in articles:
                # Check if article already exists
                cursor.execute('''
                SELECT id FROM news_sentiment
                WHERE symbol = ? AND url = ?
                ''', (article['symbol'], article['url']))
                
                if cursor.fetchone():
                    continue  # Skip if already exists
                
                # Insert article
                cursor.execute('''
                INSERT INTO news_sentiment (
                    symbol, title, source, url, published_at, 
                    sentiment_score, relevance_score
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    article['symbol'],
                    article.get('title'),
                    article.get('source'),
                    article.get('url'),
                    article.get('published_at'),
                    article.get('sentiment_score', 0),
                    article.get('relevance_score', 0)
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving news to database: {e}")
    
    def get_reddit_sentiment(self, symbol, subreddits=None, limit=25):
        """Get Reddit sentiment for a stock symbol
        
        Args:
            symbol: Stock symbol
            subreddits: List of subreddits to search (default: wallstreetbets, stocks, investing)
            limit: Maximum number of posts to retrieve per subreddit
            
        Returns:
            dict: Reddit sentiment data
        """
        if not self.reddit:
            logger.warning("Reddit API not initialized")
            return {'posts': [], 'sentiment': 0, 'mentions': 0}
        
        if subreddits is None:
            subreddits = ['wallstreetbets', 'stocks', 'investing', 'pennystocks']
        
        try:
            all_posts = []
            
            for subreddit_name in subreddits:
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)
                    
                    # Search for posts containing the symbol
                    search_query = f'{symbol}'
                    
                    for post in subreddit.search(search_query, limit=limit):
                        # Check if post is relevant
                        if self._is_relevant_post(post, symbol):
                            sentiment_score = self._analyze_text_sentiment(post.title + ' ' + post.selftext)
                            
                            post_data = {
                                'symbol': symbol,
                                'platform': 'reddit',
                                'subreddit': subreddit_name,
                                'post_id': post.id,
                                'title': post.title,
                                'content': post.selftext[:500],  # Truncate long posts
                                'author': str(post.author),
                                'published_at': datetime.fromtimestamp(post.created_utc).isoformat(),
                                'sentiment_score': sentiment_score,
                                'engagement_score': post.score,
                                'comments': post.num_comments
                            }
                            
                            all_posts.append(post_data)
                    
                    # Add a small delay to avoid rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error getting posts from r/{subreddit_name}: {e}")
                    continue
            
            # Calculate overall sentiment
            if all_posts:
                # Weight sentiment by engagement
                weighted_sentiment = sum(p['sentiment_score'] * p['engagement_score'] for p in all_posts)
                total_engagement = sum(p['engagement_score'] for p in all_posts)
                overall_sentiment = weighted_sentiment / total_engagement if total_engagement > 0 else 0
            else:
                overall_sentiment = 0
            
            logger.info(f"Got {len(all_posts)} Reddit posts for {symbol} with sentiment {overall_sentiment:.2f}")
            
            # Save to database
            self.save_social_media_to_db(all_posts)
            
            return {
                'posts': all_posts,
                'sentiment': overall_sentiment,
                'mentions': len(all_posts)
            }
            
        except Exception as e:
            logger.error(f"Error getting Reddit sentiment: {e}")
            return {'posts': [], 'sentiment': 0, 'mentions': 0}
    
    def _is_relevant_post(self, post, symbol):
        """Check if a Reddit post is relevant to a stock symbol
        
        Args:
            post: Reddit post
            symbol: Stock symbol
            
        Returns:
            bool: True if relevant, False otherwise
        """
        # Check title and content for exact symbol match
        pattern = r'\b' + re.escape(symbol) + r'\b|\$' + re.escape(symbol)
        
        if re.search(pattern, post.title, re.IGNORECASE):
            return True
        
        if re.search(pattern, post.selftext, re.IGNORECASE):
            return True
        
        return False
    
    def _analyze_text_sentiment(self, text):
        """Simple sentiment analysis for text
        
        Args:
            text: Text to analyze
            
        Returns:
            float: Sentiment score (-1 to 1)
        """
        # This is a very basic sentiment analysis
        # In a real implementation, you would use a more sophisticated model
        
        # List of positive and negative words
        positive_words = [
            'buy', 'bullish', 'up', 'upside', 'growth', 'profit', 'gain', 'positive',
            'good', 'great', 'excellent', 'strong', 'opportunity', 'undervalued',
            'potential', 'promising', 'confident', 'optimistic', 'moon', 'rocket'
        ]
        
        negative_words = [
            'sell', 'bearish', 'down', 'downside', 'decline', 'loss', 'negative',
            'bad', 'poor', 'weak', 'risk', 'overvalued', 'avoid', 'caution',
            'worried', 'pessimistic', 'crash', 'dump', 'short'
        ]
        
        # Convert to lowercase and split into words
        words = text.lower().split()
        
        # Count positive and negative words
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        # Calculate sentiment score
        total_count = positive_count + negative_count
        if total_count == 0:
            return 0
        
        return (positive_count - negative_count) / total_count
    
    def save_social_media_to_db(self, posts):
        """Save social media posts to database
        
        Args:
            posts: List of social media post dictionaries
        """
        if not posts:
            return
        
        try:
            conn = sqlite3.connect('tradebot.db')
            cursor = conn.cursor()
            
            for post in posts:
                # Check if post already exists
                cursor.execute('''
                SELECT id FROM social_media_sentiment
                WHERE platform = ? AND post_id = ?
                ''', (post['platform'], post['post_id']))
                
                if cursor.fetchone():
                    continue  # Skip if already exists
                
                # Insert post
                cursor.execute('''
                INSERT INTO social_media_sentiment (
                    symbol, platform, post_id, content, author,
                    published_at, sentiment_score, engagement_score
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    post['symbol'],
                    post['platform'],
                    post['post_id'],
                    post.get('content', ''),
                    post.get('author', ''),
                    post.get('published_at'),
                    post.get('sentiment_score', 0),
                    post.get('engagement_score', 0)
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving social media to database: {e}")
    
    def get_alternative_data_score(self, symbol):
        """Get combined alternative data score for a stock
        
        Args:
            symbol: Stock symbol
            
        Returns:
            dict: Alternative data scores and insights
        """
        # Get news and social media data
        news = self.get_news(symbol)
        reddit = self.get_reddit_sentiment(symbol)
        
        # Calculate news sentiment
        if news:
            # Use pre-calculated sentiment if available, otherwise use simple analysis
            news_with_sentiment = [
                n for n in news 
                if 'sentiment_score' in n and n['sentiment_score'] != 0
            ]
            
            if news_with_sentiment:
                # Weight by relevance if available
                weighted_sentiment = sum(
                    n['sentiment_score'] * n.get('relevance_score', 1) 
                    for n in news_with_sentiment
                )
                total_relevance = sum(n.get('relevance_score', 1) for n in news_with_sentiment)
                news_sentiment = weighted_sentiment / total_relevance if total_relevance > 0 else 0
            else:
                # Analyze sentiment ourselves
                news_sentiment = sum(
                    self._analyze_text_sentiment(n.get('title', '') + ' ' + n.get('description', ''))
                    for n in news
                ) / len(news) if news else 0
        else:
            news_sentiment = 0
        
        # Get social media sentiment
        social_sentiment = reddit['sentiment']
        
        # Calculate news volume score (0-1)
        news_volume_score = min(1.0, len(news) / 10)
        
        # Calculate social media volume score (0-1)
        social_volume_score = min(1.0, reddit['mentions'] / 20)
        
        # Calculate combined sentiment score (-1 to 1)
        # Weight news more heavily than social media
        if news_volume_score > 0 or social_volume_score > 0:
            combined_sentiment = (
                (news_sentiment * news_volume_score * 0.7) + 
                (social_sentiment * social_volume_score * 0.3)
            ) / (news_volume_score * 0.7 + social_volume_score * 0.3 or 1)
        else:
            combined_sentiment = 0
        
        # Calculate volume score (0-1)
        volume_score = (news_volume_score * 0.7) + (social_volume_score * 0.3)
        
        # Generate insights
        insights = []
        
        if news:
            insights.append(f"Found {len(news)} recent news articles")
            
            # Add most relevant positive and negative news
            news_sorted = sorted(
                [n for n in news if 'sentiment_score' in n],
                key=lambda x: x.get('relevance_score', 0) * abs(x.get('sentiment_score', 0)),
                reverse=True
            )
            
            if news_sorted:
                positive_news = [n for n in news_sorted if n.get('sentiment_score', 0) > 0.2][:2]
                negative_news = [n for n in news_sorted if n.get('sentiment_score', 0) < -0.2][:2]
                
                for n in positive_news:
                    insights.append(f"Positive news: {n.get('title')}")
                
                for n in negative_news:
                    insights.append(f"Negative news: {n.get('title')}")
        
        if reddit['mentions'] > 0:
            insights.append(f"Found {reddit['mentions']} Reddit mentions")
            
            # Add most engaged posts
            posts_sorted = sorted(reddit['posts'], key=lambda x: x.get('engagement_score', 0), reverse=True)
            
            for post in posts_sorted[:2]:
                sentiment = "positive" if post.get('sentiment_score', 0) > 0.2 else (
                    "negative" if post.get('sentiment_score', 0) < -0.2 else "neutral"
                )
                insights.append(f"{sentiment.capitalize()} Reddit post: {post.get('title')} ({post.get('engagement_score', 0)} upvotes)")
        
        # Convert sentiment to a 0-100 score for consistency with other components
        sentiment_score = (combined_sentiment + 1) * 50
        
        return {
            'symbol': symbol,
            'sentiment_score': sentiment_score,
            'volume_score': volume_score * 100,
            'news_count': len(news),
            'social_mentions': reddit['mentions'],
            'insights': insights
        }


# Test the alternative data component if run directly
if __name__ == "__main__":
    alt_data = AlternativeData()
    
    # Test with a sample stock
    symbol = "AAPL"
    
    # Get alternative data score
    score = alt_data.get_alternative_data_score(symbol)
    
    print(f"Alternative Data Score for {symbol}:")
    print(f"Sentiment Score: {score['sentiment_score']:.2f}/100")
    print(f"Volume Score: {score['volume_score']:.2f}/100")
    print(f"News Count: {score['news_count']}")
    print(f"Social Media Mentions: {score['social_mentions']}")
    
    print("\nInsights:")
    for insight in score['insights']:
        print(f"- {insight}")
