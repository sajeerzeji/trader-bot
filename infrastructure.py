import os
import sys
import time
import logging
import sqlite3
import traceback
import threading
import queue
import signal
import json
from datetime import datetime, timedelta
import requests
from functools import wraps
import asyncio
import aiohttp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tradebot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('Infrastructure')

class DatabaseManager:
    """Manages database connections and provides utility functions"""
    
    def __init__(self, db_path='tradebot.db'):
        """Initialize the database manager"""
        self.db_path = db_path
        self.connection_pool = queue.Queue(maxsize=5)
        self._initialize_pool()
        
        # Create health check table if it doesn't exist
        self._create_health_table()
        
        logger.info("DatabaseManager initialized")
    
    def _initialize_pool(self):
        """Initialize the connection pool"""
        for _ in range(5):
            conn = sqlite3.connect(self.db_path)
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON")
            self.connection_pool.put(conn)
    
    def _create_health_table(self):
        """Create health check table if it doesn't exist"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_health (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                component TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            conn.commit()
    
    def get_connection(self):
        """Get a connection from the pool"""
        try:
            return self.connection_pool.get(timeout=5)
        except queue.Empty:
            logger.warning("Connection pool empty, creating new connection")
            return sqlite3.connect(self.db_path)
    
    def release_connection(self, conn):
        """Release a connection back to the pool"""
        try:
            self.connection_pool.put(conn, block=False)
        except queue.Full:
            conn.close()
    
    def execute_query(self, query, params=None):
        """Execute a query and return results"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            result = cursor.fetchall()
            conn.commit()
            return result
        except Exception as e:
            logger.error(f"Database error: {e}")
            conn.rollback()
            raise
        finally:
            self.release_connection(conn)
    
    def execute_many(self, query, params_list):
        """Execute many queries with different parameters"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            conn.commit()
        except Exception as e:
            logger.error(f"Database error in execute_many: {e}")
            conn.rollback()
            raise
        finally:
            self.release_connection(conn)
    
    def backup_database(self, backup_path=None):
        """Create a backup of the database"""
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"backup/tradebot_{timestamp}.db"
        
        # Create backup directory if it doesn't exist
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        
        # Connect to source database
        source_conn = self.get_connection()
        try:
            # Connect to backup database
            backup_conn = sqlite3.connect(backup_path)
            
            # Backup
            source_conn.backup(backup_conn)
            
            # Close backup connection
            backup_conn.close()
            
            logger.info(f"Database backed up to {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Error backing up database: {e}")
            return False
        finally:
            self.release_connection(source_conn)
    
    def optimize_database(self):
        """Optimize the database by running VACUUM and ANALYZE"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("VACUUM")
            cursor.execute("ANALYZE")
            conn.commit()
            logger.info("Database optimized")
            return True
        except Exception as e:
            logger.error(f"Error optimizing database: {e}")
            return False
        finally:
            self.release_connection(conn)

class CacheManager:
    """Manages caching of API responses and other data"""
    
    def __init__(self, max_size=100, ttl=300):
        """Initialize the cache manager
        
        Args:
            max_size: Maximum number of items in cache
            ttl: Time to live in seconds
        """
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl
        self.lock = threading.Lock()
        
        # Start cache cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        
        logger.info("CacheManager initialized")
    
    def get(self, key):
        """Get a value from the cache"""
        with self.lock:
            if key in self.cache:
                item = self.cache[key]
                if datetime.now() < item['expiry']:
                    return item['value']
                else:
                    # Remove expired item
                    del self.cache[key]
        return None
    
    def set(self, key, value, ttl=None):
        """Set a value in the cache"""
        if ttl is None:
            ttl = self.ttl
        
        expiry = datetime.now() + timedelta(seconds=ttl)
        
        with self.lock:
            # Check if cache is full
            if len(self.cache) >= self.max_size and key not in self.cache:
                # Remove oldest item
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['expiry'])
                del self.cache[oldest_key]
            
            # Add new item
            self.cache[key] = {
                'value': value,
                'expiry': expiry
            }
    
    def delete(self, key):
        """Delete a value from the cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
    
    def clear(self):
        """Clear the entire cache"""
        with self.lock:
            self.cache.clear()
    
    def _cleanup_loop(self):
        """Background thread to clean up expired cache items"""
        while True:
            time.sleep(60)  # Check every minute
            self._cleanup()
    
    def _cleanup(self):
        """Remove expired items from the cache"""
        now = datetime.now()
        with self.lock:
            expired_keys = [k for k, v in self.cache.items() if now > v['expiry']]
            for key in expired_keys:
                del self.cache[key]

class RateLimiter:
    """Rate limiter for API calls"""
    
    def __init__(self, calls_per_second=1, calls_per_minute=60):
        """Initialize the rate limiter
        
        Args:
            calls_per_second: Maximum calls per second
            calls_per_minute: Maximum calls per minute
        """
        self.calls_per_second = calls_per_second
        self.calls_per_minute = calls_per_minute
        
        self.second_bucket = []
        self.minute_bucket = []
        
        self.lock = threading.Lock()
        
        logger.info(f"RateLimiter initialized: {calls_per_second}/s, {calls_per_minute}/min")
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        with self.lock:
            now = datetime.now()
            
            # Clean up old timestamps
            self.second_bucket = [t for t in self.second_bucket if (now - t).total_seconds() < 1]
            self.minute_bucket = [t for t in self.minute_bucket if (now - t).total_seconds() < 60]
            
            # Check if we need to wait
            if len(self.second_bucket) >= self.calls_per_second:
                sleep_time = 1.0 - (now - self.second_bucket[0]).total_seconds()
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            if len(self.minute_bucket) >= self.calls_per_minute:
                sleep_time = 60.0 - (now - self.minute_bucket[0]).total_seconds()
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            # Add current timestamp
            now = datetime.now()
            self.second_bucket.append(now)
            self.minute_bucket.append(now)
    
    def __call__(self, func):
        """Decorator for rate-limited functions"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.wait_if_needed()
            return func(*args, **kwargs)
        return wrapper

class HealthMonitor:
    """Monitors system health and components"""
    
    def __init__(self, db_manager):
        """Initialize the health monitor
        
        Args:
            db_manager: DatabaseManager instance
        """
        self.db_manager = db_manager
        self.components = {}
        self.lock = threading.Lock()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("HealthMonitor initialized")
    
    def register_component(self, name, check_function):
        """Register a component to be monitored
        
        Args:
            name: Component name
            check_function: Function that returns (status, message)
        """
        with self.lock:
            self.components[name] = {
                'check_function': check_function,
                'status': 'unknown',
                'message': 'Not checked yet',
                'last_check': None
            }
    
    def check_component(self, name):
        """Check a specific component's health
        
        Args:
            name: Component name
            
        Returns:
            tuple: (status, message)
        """
        if name not in self.components:
            return 'unknown', f"Component {name} not registered"
        
        try:
            component = self.components[name]
            status, message = component['check_function']()
            
            with self.lock:
                component['status'] = status
                component['message'] = message
                component['last_check'] = datetime.now()
            
            # Record in database
            self.db_manager.execute_query(
                "INSERT INTO system_health (component, status, message) VALUES (?, ?, ?)",
                (name, status, message)
            )
            
            return status, message
        except Exception as e:
            error_msg = f"Error checking component {name}: {e}"
            logger.error(error_msg)
            
            with self.lock:
                self.components[name]['status'] = 'error'
                self.components[name]['message'] = error_msg
                self.components[name]['last_check'] = datetime.now()
            
            # Record in database
            self.db_manager.execute_query(
                "INSERT INTO system_health (component, status, message) VALUES (?, ?, ?)",
                (name, 'error', error_msg)
            )
            
            return 'error', error_msg
    
    def check_all_components(self):
        """Check health of all registered components
        
        Returns:
            dict: Component health status
        """
        results = {}
        
        with self.lock:
            component_names = list(self.components.keys())
        
        for name in component_names:
            status, message = self.check_component(name)
            results[name] = {
                'status': status,
                'message': message
            }
        
        return results
    
    def get_system_status(self):
        """Get overall system status
        
        Returns:
            dict: System status
        """
        with self.lock:
            components = {name: {
                'status': comp['status'],
                'message': comp['message'],
                'last_check': comp['last_check']
            } for name, comp in self.components.items()}
        
        # Determine overall status
        statuses = [comp['status'] for comp in components.values()]
        
        if 'error' in statuses:
            overall_status = 'error'
        elif 'warning' in statuses:
            overall_status = 'warning'
        elif all(status == 'ok' for status in statuses):
            overall_status = 'ok'
        else:
            overall_status = 'unknown'
        
        return {
            'overall_status': overall_status,
            'components': components,
            'timestamp': datetime.now()
        }
    
    def _monitor_loop(self):
        """Background thread to periodically check component health"""
        while True:
            time.sleep(300)  # Check every 5 minutes
            try:
                self.check_all_components()
            except Exception as e:
                logger.error(f"Error in health monitor loop: {e}")

class AsyncAPIClient:
    """Asynchronous API client for making parallel requests"""
    
    def __init__(self, rate_limiter=None, cache_manager=None):
        """Initialize the async API client
        
        Args:
            rate_limiter: Optional RateLimiter instance
            cache_manager: Optional CacheManager instance
        """
        self.rate_limiter = rate_limiter
        self.cache_manager = cache_manager
        self.session = None
        
        logger.info("AsyncAPIClient initialized")
    
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close(self):
        """Close the session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def get(self, url, params=None, headers=None, use_cache=True, cache_ttl=None):
        """Make an async GET request
        
        Args:
            url: URL to request
            params: Query parameters
            headers: Request headers
            use_cache: Whether to use cache
            cache_ttl: Cache TTL in seconds
            
        Returns:
            dict: Response data
        """
        # Check cache
        cache_key = f"GET:{url}:{json.dumps(params) if params else ''}"
        if use_cache and self.cache_manager:
            cached = self.cache_manager.get(cache_key)
            if cached:
                return cached
        
        # Apply rate limiting
        if self.rate_limiter:
            self.rate_limiter.wait_if_needed()
        
        # Make request
        session = await self._get_session()
        try:
            async with session.get(url, params=params, headers=headers) as response:
                response.raise_for_status()
                data = await response.json()
                
                # Cache response
                if use_cache and self.cache_manager:
                    self.cache_manager.set(cache_key, data, ttl=cache_ttl)
                
                return data
        except aiohttp.ClientError as e:
            logger.error(f"API request error: {e}")
            raise
    
    async def fetch_multiple(self, requests):
        """Fetch multiple URLs in parallel
        
        Args:
            requests: List of request dicts with url, params, headers
            
        Returns:
            list: Response data for each request
        """
        tasks = []
        for req in requests:
            url = req['url']
            params = req.get('params')
            headers = req.get('headers')
            use_cache = req.get('use_cache', True)
            cache_ttl = req.get('cache_ttl')
            
            task = asyncio.create_task(
                self.get(url, params, headers, use_cache, cache_ttl)
            )
            tasks.append(task)
        
        return await asyncio.gather(*tasks, return_exceptions=True)

class ErrorHandler:
    """Handles and logs errors"""
    
    def __init__(self, db_manager=None, notify_url=None):
        """Initialize the error handler
        
        Args:
            db_manager: Optional DatabaseManager instance
            notify_url: Optional URL for error notifications
        """
        self.db_manager = db_manager
        self.notify_url = notify_url
        
        # Create errors table if it doesn't exist
        if db_manager:
            self._create_errors_table()
        
        logger.info("ErrorHandler initialized")
    
    def _create_errors_table(self):
        """Create errors table if it doesn't exist"""
        try:
            self.db_manager.execute_query('''
            CREATE TABLE IF NOT EXISTS errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                component TEXT NOT NULL,
                error_type TEXT NOT NULL,
                message TEXT,
                traceback TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
        except Exception as e:
            logger.error(f"Error creating errors table: {e}")
    
    def handle_error(self, component, error, context=None):
        """Handle an error
        
        Args:
            component: Component name
            error: Exception object
            context: Optional context dict
            
        Returns:
            bool: True if handled successfully
        """
        error_type = type(error).__name__
        message = str(error)
        tb = traceback.format_exc()
        
        # Log error
        logger.error(f"Error in {component}: {error_type} - {message}\n{tb}")
        
        # Record in database
        if self.db_manager:
            try:
                self.db_manager.execute_query(
                    "INSERT INTO errors (component, error_type, message, traceback) VALUES (?, ?, ?, ?)",
                    (component, error_type, message, tb)
                )
            except Exception as e:
                logger.error(f"Error recording error in database: {e}")
        
        # Send notification
        if self.notify_url:
            try:
                notification_data = {
                    'component': component,
                    'error_type': error_type,
                    'message': message,
                    'context': context
                }
                requests.post(self.notify_url, json=notification_data, timeout=5)
            except Exception as e:
                logger.error(f"Error sending notification: {e}")
        
        return True
    
    def __call__(self, component):
        """Decorator for error handling
        
        Args:
            component: Component name
            
        Returns:
            decorator: Error handling decorator
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self.handle_error(component, e)
                    raise
            return wrapper
        return decorator

class GracefulShutdown:
    """Handles graceful shutdown of the application"""
    
    def __init__(self):
        """Initialize the graceful shutdown handler"""
        self.shutdown_event = threading.Event()
        self.handlers = []
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("GracefulShutdown initialized")
    
    def _signal_handler(self, sig, frame):
        """Handle termination signals"""
        logger.info(f"Received signal {sig}, initiating graceful shutdown")
        self.initiate_shutdown()
    
    def register_handler(self, handler, *args, **kwargs):
        """Register a shutdown handler
        
        Args:
            handler: Function to call on shutdown
            args, kwargs: Arguments to pass to the handler
        """
        self.handlers.append((handler, args, kwargs))
    
    def initiate_shutdown(self):
        """Initiate graceful shutdown"""
        if self.shutdown_event.is_set():
            logger.info("Shutdown already in progress")
            return
        
        self.shutdown_event.set()
        
        # Call all handlers
        for handler, args, kwargs in self.handlers:
            try:
                handler(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in shutdown handler: {e}")
        
        logger.info("Graceful shutdown complete")
    
    def is_shutting_down(self):
        """Check if shutdown is in progress
        
        Returns:
            bool: True if shutting down
        """
        return self.shutdown_event.is_set()

# Infrastructure manager class
class InfrastructureManager:
    """Manages all infrastructure components"""
    
    def __init__(self):
        """Initialize the infrastructure manager"""
        # Create components
        self.db_manager = DatabaseManager()
        self.cache_manager = CacheManager()
        self.rate_limiter = RateLimiter()
        self.health_monitor = HealthMonitor(self.db_manager)
        self.error_handler = ErrorHandler(self.db_manager)
        self.shutdown_manager = GracefulShutdown()
        
        # Register health checks
        self._register_health_checks()
        
        # Register shutdown handlers
        self._register_shutdown_handlers()
        
        logger.info("InfrastructureManager initialized")
    
    def _register_health_checks(self):
        """Register component health checks"""
        # Database health check
        def check_db():
            try:
                self.db_manager.execute_query("SELECT 1")
                return 'ok', "Database connection successful"
            except Exception as e:
                return 'error', f"Database connection failed: {e}"
        
        self.health_monitor.register_component('database', check_db)
        
        # Disk space check
        def check_disk_space():
            try:
                # Get disk usage of current directory
                total, used, free = self._get_disk_usage()
                
                # Calculate percentage used
                percent_used = (used / total) * 100
                
                if percent_used > 90:
                    return 'error', f"Disk space critical: {percent_used:.1f}% used"
                elif percent_used > 80:
                    return 'warning', f"Disk space low: {percent_used:.1f}% used"
                else:
                    return 'ok', f"Disk space ok: {percent_used:.1f}% used"
            except Exception as e:
                return 'error', f"Disk space check failed: {e}"
        
        self.health_monitor.register_component('disk_space', check_disk_space)
        
        # Memory usage check
        def check_memory():
            try:
                # Get memory usage
                memory_info = self._get_memory_usage()
                
                # Calculate percentage used
                percent_used = memory_info['percent']
                
                if percent_used > 90:
                    return 'error', f"Memory usage critical: {percent_used:.1f}% used"
                elif percent_used > 80:
                    return 'warning', f"Memory usage high: {percent_used:.1f}% used"
                else:
                    return 'ok', f"Memory usage ok: {percent_used:.1f}% used"
            except Exception as e:
                return 'error', f"Memory check failed: {e}"
        
        self.health_monitor.register_component('memory', check_memory)
    
    def _register_shutdown_handlers(self):
        """Register handlers for graceful shutdown"""
        # Database backup handler
        self.shutdown_manager.register_handler(self.db_manager.backup_database)
        
        # Cache cleanup handler
        self.shutdown_manager.register_handler(self.cache_manager.clear)
    
    def _get_disk_usage(self):
        """Get disk usage of current directory
        
        Returns:
            tuple: (total, used, free) in bytes
        """
        if os.name == 'posix':
            # Unix/Linux/MacOS
            st = os.statvfs('.')
            free = st.f_bavail * st.f_frsize
            total = st.f_blocks * st.f_frsize
            used = (st.f_blocks - st.f_bfree) * st.f_frsize
            return (total, used, free)
        else:
            # Windows
            import ctypes
            free_bytes = ctypes.c_ulonglong(0)
            total_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                ctypes.c_wchar_p('.'), None, ctypes.pointer(total_bytes),
                ctypes.pointer(free_bytes)
            )
            used = total_bytes.value - free_bytes.value
            return (total_bytes.value, used, free_bytes.value)
    
    def _get_memory_usage(self):
        """Get memory usage
        
        Returns:
            dict: Memory usage information
        """
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            return {
                'rss': memory_info.rss,  # Resident Set Size
                'vms': memory_info.vms,  # Virtual Memory Size
                'percent': process.memory_percent()
            }
        except ImportError:
            # Fallback if psutil is not available
            return {
                'rss': 0,
                'vms': 0,
                'percent': 0
            }
    
    def get_system_status(self):
        """Get system status
        
        Returns:
            dict: System status
        """
        return self.health_monitor.get_system_status()
    
    def optimize_database(self):
        """Optimize the database"""
        return self.db_manager.optimize_database()
    
    def backup_database(self):
        """Backup the database"""
        return self.db_manager.backup_database()
    
    def shutdown(self):
        """Initiate graceful shutdown"""
        self.shutdown_manager.initiate_shutdown()

# Test the infrastructure components if run directly
if __name__ == "__main__":
    infra = InfrastructureManager()
    
    # Check system status
    status = infra.get_system_status()
    
    print("System Status:")
    print(f"Overall: {status['overall_status']}")
    print("\nComponents:")
    for name, component in status['components'].items():
        print(f"- {name}: {component['status']} - {component['message']}")
    
    # Optimize database
    print("\nOptimizing database...")
    infra.optimize_database()
    
    # Backup database
    print("\nBacking up database...")
    infra.backup_database()
    
    print("\nInfrastructure test complete")
