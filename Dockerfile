FROM python:3.9-slim

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# No need to reinstall build dependencies

# Install TA-Lib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib-0.4.0-src.tar.gz ta-lib/

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
COPY requirements_fixed.txt requirements_fixed.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir numpy && \
    pip install --no-cache-dir -r requirements_fixed.txt

# Copy application code
COPY . .

# Create directory for data persistence
RUN mkdir -p /data/db /data/logs /data/models /data/backtest

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATABASE_PATH=/data/db/tradebot.db \
    LOG_PATH=/data/logs/tradebot.log \
    MODELS_PATH=/data/models \
    BACKTEST_PATH=/data/backtest

# Create a non-root user
RUN useradd -m tradebot
RUN chown -R tradebot:tradebot /app /data

# Switch to non-root user
USER tradebot

# Command to run the application
CMD ["python", "main.py"]
