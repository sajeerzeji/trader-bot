#!/usr/bin/env python
"""
Simple test script to verify trading bot components
"""
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TestComponents')

def check_dependencies():
    """Check if required dependencies are installed"""
    missing_packages = []
    
    # Core dependencies
    dependencies = [
        'pandas', 'numpy', 'alpaca_trade_api', 'python_dotenv', 
        'schedule', 'sqlite3'
    ]
    
    for package in dependencies:
        try:
            if package == 'python_dotenv':
                # Special case for python-dotenv
                import dotenv
                logger.info(f"✅ python-dotenv is installed")
            else:
                __import__(package)
                logger.info(f"✅ {package} is installed")
        except ImportError as e:
            missing_packages.append(package)
            logger.error(f"❌ {package} is NOT installed: {e}")
    
    # Optional dependencies
    optional_deps = ['talib', 'matplotlib', 'seaborn', 'scikit_learn']
    for package in optional_deps:
        try:
            __import__(package)
            logger.info(f"✅ {package} is installed (optional)")
        except ImportError as e:
            logger.warning(f"⚠️ {package} is NOT installed (optional): {e}")
    
    return missing_packages

def check_environment():
    """Check if environment variables are set"""
    required_env_vars = [
        'ALPACA_API_KEY', 
        'ALPACA_API_SECRET', 
        'ALPACA_API_ENDPOINT'
    ]
    
    missing_env_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_env_vars.append(var)
            logger.error(f"❌ Environment variable {var} is NOT set")
        else:
            logger.info(f"✅ Environment variable {var} is set")
    
    return missing_env_vars

def check_database():
    """Check if database exists and is accessible"""
    try:
        import sqlite3
        conn = sqlite3.connect('data/db/tradebot.db')
        cursor = conn.cursor()
        
        # Check if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if tables:
            logger.info(f"✅ Database exists with {len(tables)} tables")
            for table in tables:
                logger.info(f"  - Table: {table[0]}")
        else:
            logger.warning("⚠️ Database exists but has no tables")
        
        conn.close()
        return True
    except Exception as e:
        logger.error(f"❌ Database error: {e}")
        return False

def main():
    """Main function"""
    logger.info("Starting component tests")
    
    # Check Python version
    logger.info(f"Python version: {sys.version}")
    
    # Check dependencies
    missing_packages = check_dependencies()
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.error("Please install required packages: pip install -r requirements.txt")
    else:
        logger.info("All required packages are installed")
    
    # Check environment variables
    missing_env_vars = check_environment()
    if missing_env_vars:
        logger.error(f"Missing environment variables: {', '.join(missing_env_vars)}")
        logger.error("Please set these variables in your .env file")
    else:
        logger.info("All required environment variables are set")
    
    # Check database
    db_ok = check_database()
    if not db_ok:
        logger.error("Database check failed")
    
    # Overall status
    if not missing_packages and not missing_env_vars and db_ok:
        logger.info("✅ All component tests passed!")
        return 0
    else:
        logger.error("❌ Some component tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
