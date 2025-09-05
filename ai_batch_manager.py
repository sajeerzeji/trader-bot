import os
import time
import json
import logging
import threading
import queue
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ai_batch_manager.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('AIBatchManager')

# Load environment variables
load_dotenv()

class AIBatchManager:
    """
    Manages batched API calls to Anthropic's Claude API to optimize usage and costs.
    
    This class:
    1. Collects multiple queries into batches
    2. Processes batches at regular intervals or when threshold is reached
    3. Handles rate limiting and retries
    4. Provides caching for recent responses
    """
    
    def __init__(self, batch_size: int = 5, batch_interval_seconds: int = 60, 
                 cache_expiry_minutes: int = 60, max_retries: int = 3):
        """
        Initialize the batch manager.
        
        Args:
            batch_size: Maximum number of queries to include in a single batch
            batch_interval_seconds: Maximum time to wait before processing a batch
            cache_expiry_minutes: How long to keep cached responses
            max_retries: Maximum number of retries for failed API calls
        """
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        self.model = os.getenv('ANTHROPIC_MODEL', 'claude-3-opus-20240229')
        
        if not self.api_key:
            logger.error("Anthropic API key not found in environment variables")
            raise ValueError("Anthropic API key not found")
        
        self.batch_size = batch_size
        self.batch_interval_seconds = batch_interval_seconds
        self.cache_expiry_minutes = cache_expiry_minutes
        self.max_retries = max_retries
        
        # Queue for incoming queries
        self.query_queue = queue.Queue()
        
        # Cache for responses
        self.response_cache = {}
        
        # Tracking for rate limiting
        self.last_api_call = datetime.now() - timedelta(minutes=1)
        self.min_interval_seconds = 1  # Minimum time between API calls
        
        # Start the background processing thread
        self.processing_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.processing_thread.start()
        
        # Flag to track if the manager is running
        self.running = True
        
        logger.info(f"AIBatchManager initialized with batch_size={batch_size}, "
                   f"batch_interval={batch_interval_seconds}s")
    
    def add_query(self, query_id: str, query_text: str, 
                  callback: Optional[Callable[[str, Any], None]] = None,
                  priority: int = 1) -> None:
        """
        Add a query to the batch queue.
        
        Args:
            query_id: Unique identifier for the query
            query_text: The text to send to Claude
            callback: Function to call with results when available
            priority: Priority level (higher numbers = higher priority)
        """
        # Check cache first
        cache_key = self._generate_cache_key(query_text)
        if cache_key in self.response_cache:
            cache_entry = self.response_cache[cache_key]
            if datetime.now() < cache_entry['expiry']:
                logger.info(f"Cache hit for query {query_id}")
                if callback:
                    callback(query_id, cache_entry['response'])
                return
        
        # Add to queue
        self.query_queue.put({
            'id': query_id,
            'text': query_text,
            'callback': callback,
            'priority': priority,
            'timestamp': datetime.now()
        })
        logger.debug(f"Added query {query_id} to queue")
    
    def get_response_sync(self, query_id: str, query_text: str, 
                         timeout_seconds: int = 30) -> Optional[Any]:
        """
        Get a response synchronously, waiting for the result.
        
        Args:
            query_id: Unique identifier for the query
            query_text: The text to send to Claude
            timeout_seconds: Maximum time to wait for a response
            
        Returns:
            The API response or None if timeout occurs
        """
        result_queue = queue.Queue()
        
        def callback(qid, response):
            if qid == query_id:
                result_queue.put(response)
        
        self.add_query(query_id, query_text, callback, priority=2)
        
        try:
            return result_queue.get(timeout=timeout_seconds)
        except queue.Empty:
            logger.warning(f"Timeout waiting for response to query {query_id}")
            return None
    
    def _process_queue(self) -> None:
        """Background thread that processes the query queue."""
        while self.running:
            try:
                # Collect queries for a batch
                current_batch = []
                batch_start_time = datetime.now()
                
                # Try to fill the batch
                while (len(current_batch) < self.batch_size and 
                       (datetime.now() - batch_start_time).total_seconds() < self.batch_interval_seconds):
                    
                    try:
                        # Wait for a short time for new queries
                        query = self.query_queue.get(timeout=1)
                        current_batch.append(query)
                    except queue.Empty:
                        # No new queries, continue waiting if batch has items
                        if current_batch:
                            continue
                        else:
                            break
                
                # Process the batch if we have any queries
                if current_batch:
                    self._process_batch(current_batch)
                
                # Sleep briefly to prevent CPU spinning
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in queue processing: {e}")
                time.sleep(5)  # Back off on errors
    
    def _process_batch(self, batch: List[Dict]) -> None:
        """
        Process a batch of queries.
        
        Args:
            batch: List of query dictionaries
        """
        logger.info(f"Processing batch of {len(batch)} queries")
        
        # Sort batch by priority (higher first)
        batch.sort(key=lambda x: (-x['priority'], x['timestamp']))
        
        # Process each query in the batch
        for query in batch:
            query_id = query['id']
            query_text = query['text']
            callback = query['callback']
            
            # Check cache again (might have been added since queuing)
            cache_key = self._generate_cache_key(query_text)
            if cache_key in self.response_cache:
                cache_entry = self.response_cache[cache_key]
                if datetime.now() < cache_entry['expiry']:
                    logger.info(f"Cache hit for query {query_id} during processing")
                    if callback:
                        callback(query_id, cache_entry['response'])
                    continue
            
            # Call the API
            response = self._call_claude_api(query_text)
            
            # Cache the response
            if response:
                self.response_cache[cache_key] = {
                    'response': response,
                    'expiry': datetime.now() + timedelta(minutes=self.cache_expiry_minutes)
                }
                
                # Call the callback if provided
                if callback:
                    callback(query_id, response)
    
    def _call_claude_api(self, query_text: str) -> Optional[Any]:
        """
        Call the Claude API with rate limiting and retries.
        
        Args:
            query_text: The text to send to Claude
            
        Returns:
            The API response or None if all retries fail
        """
        # Respect rate limits
        time_since_last_call = (datetime.now() - self.last_api_call).total_seconds()
        if time_since_last_call < self.min_interval_seconds:
            sleep_time = self.min_interval_seconds - time_since_last_call
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        # Update last call time
        self.last_api_call = datetime.now()
        
        # Prepare the API request
        headers = {
            "x-api-key": self.api_key,
            "content-type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": query_text}
            ],
            "max_tokens": 1024
        }
        
        # Try the API call with retries
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limited
                    retry_after = int(response.headers.get('retry-after', 5))
                    logger.warning(f"Rate limited by Anthropic API. Retrying after {retry_after} seconds")
                    time.sleep(retry_after)
                else:
                    logger.error(f"API error: {response.status_code} - {response.text}")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
            except Exception as e:
                logger.error(f"Error calling Claude API: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
        
        logger.error(f"Failed to get response after {self.max_retries} attempts")
        return None
    
    def _generate_cache_key(self, query_text: str) -> str:
        """Generate a cache key for a query."""
        # Simple hash of the query text
        return f"query_{hash(query_text)}"
    
    def clear_cache(self) -> None:
        """Clear the response cache."""
        self.response_cache = {}
        logger.info("Response cache cleared")
    
    def shutdown(self) -> None:
        """Shutdown the batch manager."""
        self.running = False
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)
        logger.info("AIBatchManager shut down")


# Example usage
if __name__ == "__main__":
    # Create batch manager
    batch_manager = AIBatchManager(batch_size=3, batch_interval_seconds=10)
    
    # Define a callback function
    def print_response(query_id, response):
        print(f"Response for query {query_id}:")
        if response and 'content' in response:
            print(response['content'][0]['text'])
        else:
            print("No valid response")
    
    # Add some test queries
    batch_manager.add_query(
        "stock_analysis_1", 
        "Analyze the following stock: AAPL. Recent price movement shows a 5% increase over the last week with increasing volume. RSI is 68. What's your assessment?",
        print_response
    )
    
    batch_manager.add_query(
        "stock_analysis_2", 
        "Analyze the following stock: MSFT. Recent price movement shows a 2% decrease over the last week with decreasing volume. RSI is 42. What's your assessment?",
        print_response
    )
    
    # Get a response synchronously
    response = batch_manager.get_response_sync(
        "stock_analysis_3",
        "Analyze the following stock: AMZN. Recent price movement shows a 1% increase over the last week with stable volume. RSI is 55. What's your assessment?"
    )
    
    if response:
        print(f"Sync response: {response}")
    
    # Keep the main thread alive to allow background processing
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        batch_manager.shutdown()
