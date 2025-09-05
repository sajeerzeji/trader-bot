FROM ubuntu:22.04

# Install Python and build dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    python3.9-dev \
    build-essential \
    wget \
    && ln -sf /usr/bin/python3.9 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install system dependencies for TA-Lib
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

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
RUN pip install --no-cache-dir -r requirements.txt

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
