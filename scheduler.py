import os
import sys
import time
import logging
import schedule
import subprocess
from datetime import datetime, timedelta
import sqlite3
from dotenv import load_dotenv

# Configure logging
# Create logs directory if it doesn't exist
os.makedirs('data/logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/logs/scheduler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('Scheduler')

# Load environment variables
load_dotenv()

# Get database path from environment or use default
DB_PATH = os.getenv('DATABASE_PATH', 'data/db/tradebot.db')

def run_stock_scanner():
    """Run stock scanner to find potential stocks"""
    logger.info("Running stock scanner...")
    try:
        subprocess.run(["python", "stock_scanner.py"], check=True)
        logger.info("Stock scanner completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Stock scanner failed: {e}")

def run_buy_engine():
    """Run buy engine to execute buy orders"""
    logger.info("Running buy engine...")
    try:
        subprocess.run(["python", "buy_engine.py"], check=True)
        logger.info("Buy engine completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Buy engine failed: {e}")

def run_monitor_engine():
    """Run monitor engine to update positions"""
    logger.info("Running monitor engine...")
    try:
        subprocess.run(["python", "monitor_engine.py"], check=True)
        logger.info("Monitor engine completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Monitor engine failed: {e}")

def run_sell_engine():
    """Run sell engine to execute sell orders"""
    logger.info("Running sell engine...")
    try:
        subprocess.run(["python", "sell_engine.py"], check=True)
        logger.info("Sell engine completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Sell engine failed: {e}")

def optimize_database():
    """Optimize the database"""
    logger.info("Optimizing database...")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("VACUUM")
        cursor.execute("ANALYZE")
        conn.commit()
        conn.close()
        logger.info("Database optimization completed")
    except Exception as e:
        logger.error(f"Database optimization failed: {e}")

def train_ml_models():
    """Train machine learning models"""
    logger.info("Training ML models...")
    try:
        subprocess.run(["python", "-c", "from ml_models import MLModels; MLModels().train_all_models('AAPL')"], check=True)
        logger.info("ML model training completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"ML model training failed: {e}")

def run_health_check():
    """Run health check"""
    logger.info("Running health check...")
    try:
        # Check database connection
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        conn.close()
        
        # Check disk space
        data_dir = "data"  # Use relative path
        disk = os.statvfs(data_dir)
        free_space = disk.f_bavail * disk.f_frsize
        total_space = disk.f_blocks * disk.f_frsize
        used_percent = (1 - free_space / total_space) * 100
        
        if used_percent > 90:
            logger.warning(f"Disk space critical: {used_percent:.1f}% used")
        
        logger.info("Health check completed successfully")
    except Exception as e:
        logger.error(f"Health check failed: {e}")

def is_market_open():
    """Check if the market is open"""
    # This is a simplified check - in production you would use the broker API
    now = datetime.now()
    
    # Check if it's a weekday
    if now.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return False
    
    # Check if it's between 9:30 AM and 4:00 PM EST
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= now <= market_close

def schedule_jobs():
    """Schedule all jobs"""
    # Market hours jobs
    schedule.every().day.at("09:35").do(run_stock_scanner)
    schedule.every().day.at("10:00").do(run_buy_engine)
    schedule.every(15).minutes.do(lambda: run_monitor_engine() if is_market_open() else None)
    schedule.every(30).minutes.do(lambda: run_sell_engine() if is_market_open() else None)
    
    # After-hours jobs
    schedule.every().day.at("16:30").do(optimize_database)
    schedule.every().sunday.at("12:00").do(train_ml_models)
    
    # Health check
    schedule.every(1).hours.do(run_health_check)
    
    logger.info("All jobs scheduled")

def main():
    """Main function"""
    logger.info("Starting scheduler...")
    
    # Schedule jobs
    schedule_jobs()
    
    # Run health check immediately
    run_health_check()
    
    # Run loop
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    main()
