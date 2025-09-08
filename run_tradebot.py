#!/usr/bin/env python3
"""
Direct execution script for TradeBot - runs without Docker
"""

import os
import sys
import time
import logging
import argparse
import subprocess
import signal
import atexit
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tradebot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TradeBot')

# Create necessary directories
def create_directories():
    """Create necessary directories for data persistence"""
    directories = [
        'data/db',
        'data/logs',
        'data/models',
        'data/backtest',
        'backups'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

# Check environment
def check_environment():
    """Check if the environment is properly set up"""
    # Check if .env file exists
    if not os.path.exists('.env'):
        logger.error("No .env file found. Please create one with your API keys and settings.")
        sys.exit(1)
    
    # Check if database exists, if not create it
    if not os.path.exists('data/db/tradebot.db'):
        logger.info("Database not found. Creating database...")
        try:
            subprocess.run([sys.executable, 'create_database.py'], check=True)
            logger.info("Database created successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create database: {e}")
            sys.exit(1)
    
    # Check Python packages
    try:
        import pandas
        import alpaca_trade_api
        import dotenv
        import schedule
        import numpy
        import ta
        logger.info("All required packages are installed")
    except ImportError as e:
        logger.error(f"Missing required package: {e}")
        logger.error("Please install required packages: pip install -r requirements.txt")
        sys.exit(1)

# Process management
running_processes = {}

def start_process(name, script, args=None):
    """Start a Python script as a subprocess"""
    cmd = [sys.executable, script]
    if args:
        cmd.extend(args)
    
    logger.info(f"Starting {name}...")
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        running_processes[name] = process
        logger.info(f"{name} started with PID {process.pid}")
        return process
    except Exception as e:
        logger.error(f"Failed to start {name}: {e}")
        return None

def stop_process(name):
    """Stop a running process"""
    if name in running_processes:
        process = running_processes[name]
        logger.info(f"Stopping {name} (PID {process.pid})...")
        try:
            process.terminate()
            process.wait(timeout=5)
            logger.info(f"{name} stopped")
        except subprocess.TimeoutExpired:
            logger.warning(f"{name} did not terminate gracefully, killing...")
            process.kill()
        except Exception as e:
            logger.error(f"Error stopping {name}: {e}")
        
        del running_processes[name]

def stop_all_processes():
    """Stop all running processes"""
    for name in list(running_processes.keys()):
        stop_process(name)

# Register cleanup function
atexit.register(stop_all_processes)

# Handle signals
def signal_handler(sig, frame):
    """Handle termination signals"""
    logger.info("Received termination signal, shutting down...")
    stop_all_processes()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Backup database
def backup_database():
    """Create a backup of the database"""
    logger.info("Creating database backup...")
    backup_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"backups/tradebot_{backup_date}.db"
    
    try:
        conn = sqlite3.connect('data/db/tradebot.db')
        cursor = conn.cursor()
        
        # Create a backup
        with open(backup_file, 'wb') as f:
            for line in conn.iterdump():
                f.write(f"{line}\n".encode('utf-8'))
        
        conn.close()
        logger.info(f"Backup created: {backup_file}")
        return backup_file
    except Exception as e:
        logger.error(f"Backup failed: {e}")
        return None

# Restore database
def restore_database(backup_file):
    """Restore database from backup"""
    if not os.path.exists(backup_file):
        logger.error(f"Backup file not found: {backup_file}")
        return False
    
    logger.info(f"Restoring database from {backup_file}...")
    
    try:
        # Stop all processes
        stop_all_processes()
        
        # Remove current database
        if os.path.exists('data/db/tradebot.db'):
            os.rename('data/db/tradebot.db', f'data/db/tradebot_old_{datetime.now().strftime("%Y%m%d_%H%M%S")}.db')
        
        # Create new database
        conn = sqlite3.connect('data/db/tradebot.db')
        cursor = conn.cursor()
        
        # Restore from backup
        with open(backup_file, 'r') as f:
            conn.executescript(f.read())
        
        conn.close()
        logger.info("Database restored successfully")
        return True
    except Exception as e:
        logger.error(f"Restore failed: {e}")
        return False

# Run backtest
def run_backtest():
    """Run backtesting"""
    logger.info("Running backtest...")
    try:
        process = subprocess.run([sys.executable, 'backtesting.py'], check=True)
        logger.info("Backtest completed")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Backtest failed: {e}")
        return False

# Run tests
def run_tests():
    """Run integration tests"""
    logger.info("Running tests...")
    try:
        process = subprocess.run([sys.executable, '-m', 'pytest', 'integration_tests.py', '-v'], check=True)
        logger.info("Tests completed")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Tests failed: {e}")
        return False

# Start trading bot
def start_trading(auto_restart=False, restart_interval=15):
    """Start the trading bot components
    
    Args:
        auto_restart (bool): Whether to automatically restart the bot
        restart_interval (int): Restart interval in minutes
    """
    start_time = datetime.now()
    restart_time = start_time + timedelta(minutes=restart_interval)
    
    # Start main controller
    main_process = start_process("Main Controller", "main.py")
    
    # Start scheduler
    scheduler_process = start_process("Scheduler", "scheduler.py")
    
    if auto_restart:
        logger.info(f"Auto-restart enabled. Bot will restart every {restart_interval} minutes.")
        logger.info(f"Next restart scheduled at: {restart_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Monitor logs
    while main_process.poll() is None and scheduler_process.poll() is None:
        # Check if it's time to restart
        if auto_restart and datetime.now() >= restart_time:
            logger.info(f"Auto-restart triggered after {restart_interval} minutes of runtime")
            
            # Generate dashboard before restart to capture latest data
            try:
                logger.info("Generating dashboard before restart...")
                subprocess.run([sys.executable, 'generate_html_dashboard.py'], 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except Exception as e:
                logger.error(f"Failed to generate dashboard: {e}")
            
            # Stop current processes
            stop_all_processes()
            
            # Start new processes
            logger.info("Restarting trading bot...")
            main_process = start_process("Main Controller", "main.py")
            scheduler_process = start_process("Scheduler", "scheduler.py")
            
            # Reset restart time
            restart_time = datetime.now() + timedelta(minutes=restart_interval)
            logger.info(f"Next restart scheduled at: {restart_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Process logs
        line = main_process.stdout.readline() if main_process and main_process.stdout else None
        if line:
            print(f"[Main] {line.strip()}")
        
        line = scheduler_process.stdout.readline() if scheduler_process and scheduler_process.stdout else None
        if line:
            print(f"[Scheduler] {line.strip()}")
        
        time.sleep(0.1)

# Command-line interface
def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='TradeBot Direct Execution')
    parser.add_argument('command', choices=['start', 'stop', 'backup', 'restore', 'backtest', 'test'],
                        help='Command to execute')
    parser.add_argument('--backup-file', help='Backup file for restore command')
    parser.add_argument('--auto-restart', action='store_true', help='Automatically restart the bot periodically')
    parser.add_argument('--restart-interval', type=int, default=15, help='Restart interval in minutes (default: 15)')
    parser.add_argument('--generate-dashboard', action='store_true', help='Generate dashboard on each restart')
    
    args = parser.parse_args()
    
    # Create directories
    create_directories()
    
    # Check environment
    check_environment()
    
    # Execute command
    if args.command == 'start':
        # Enable auto-restart by default
        auto_restart = True
        start_trading(auto_restart=auto_restart, restart_interval=args.restart_interval)
    elif args.command == 'stop':
        stop_all_processes()
    elif args.command == 'backup':
        backup_database()
    elif args.command == 'restore':
        if not args.backup_file:
            logger.error("Please specify a backup file with --backup-file")
            sys.exit(1)
        restore_database(args.backup_file)
    elif args.command == 'backtest':
        run_backtest()
    elif args.command == 'test':
        run_tests()

if __name__ == "__main__":
    main()
