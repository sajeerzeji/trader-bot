#!/usr/bin/env python3
"""
Generate HTML Dashboard for TradeBot
Creates a static HTML dashboard from trading data in the database
"""

import sqlite3
import pandas as pd
import json
import os
import datetime
from pathlib import Path

# Database and output paths
DB_PATH = os.path.join('data', 'db', 'tradebot.db')
OUTPUT_PATH = 'dashboard.html'

def load_data():
    """Load data from the database"""
    try:
        if not os.path.exists(DB_PATH):
            print(f"Database not found at {DB_PATH}. Cannot generate dashboard.")
            return None, None
        
        conn = sqlite3.connect(DB_PATH)
        
        # Load positions
        positions_query = "SELECT * FROM positions ORDER BY buy_date DESC"
        positions_df = pd.read_sql_query(positions_query, conn)
        
        # Load orders
        orders_query = "SELECT * FROM orders ORDER BY timestamp DESC"
        orders_df = pd.read_sql_query(orders_query, conn)
        
        conn.close()
        
        print(f"Loaded {len(positions_df)} positions and {len(orders_df)} orders from database.")
        return positions_df, orders_df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def calculate_profit_loss(positions_df):
    """Calculate profit/loss for closed positions"""
    if positions_df is None or positions_df.empty:
        return pd.DataFrame()
    
    # Filter for closed positions
    closed_positions = positions_df[positions_df['status'] == 'closed'].copy()
    
    if closed_positions.empty:
        return pd.DataFrame()
    
    # Calculate profit/loss
    closed_positions['profit_loss'] = (closed_positions['sell_price'] - closed_positions['buy_price']) * closed_positions['quantity']
    closed_positions['profit_loss_percent'] = (closed_positions['sell_price'] - closed_positions['buy_price']) / closed_positions['buy_price'] * 100
    
    return closed_positions

def generate_dashboard(positions_df, orders_df):
    """Generate HTML dashboard from trading data"""
    if positions_df is None or orders_df is None:
        return "<html><body><h1>Error loading data</h1></body></html>"
    
    # Calculate profit/loss for closed positions
    closed_positions_df = calculate_profit_loss(positions_df)
    
    # Calculate summary metrics
    total_positions = len(positions_df)
    open_positions = len(positions_df[positions_df['status'] == 'open'])
    closed_position_count = len(closed_positions_df)
    
    # Calculate total amount spent (sum of buy_price * quantity for all positions)
    total_amount_spent = (positions_df['buy_price'] * positions_df['quantity']).sum()
    
    # Calculate total current value of all positions
    total_current_value = (positions_df['current_price'] * positions_df['quantity']).sum()
    
    total_profit_loss = closed_positions_df['profit_loss'].sum() if not closed_positions_df.empty else 0
    avg_profit_loss_percent = closed_positions_df['profit_loss_percent'].mean() if not closed_positions_df.empty else 0
    
    # Prepare data for charts
    profit_by_symbol = closed_positions_df.groupby('symbol')['profit_loss'].sum().to_dict() if not closed_positions_df.empty else {}
    
    # Prepare data for trading activity chart
    if not orders_df.empty:
        orders_df['date'] = pd.to_datetime(orders_df['timestamp']).dt.date
        activity_data = orders_df.groupby(['date', 'side']).size().unstack(fill_value=0).reset_index()
        activity_data['date'] = activity_data['date'].astype(str)
        
        # Convert to format needed for chart
        dates = activity_data['date'].tolist()
        buy_counts = activity_data['buy'].tolist() if 'buy' in activity_data else []
        sell_counts = activity_data['sell'].tolist() if 'sell' in activity_data else []
    else:
        dates = []
        buy_counts = []
        sell_counts = []
    
    # Prepare data for asset allocation chart
    open_positions_df = positions_df[positions_df['status'] == 'open']
    if not open_positions_df.empty:
        open_positions_df['current_value'] = open_positions_df['current_price'] * open_positions_df['quantity']
        allocation_data = open_positions_df.groupby('symbol')['current_value'].sum().to_dict()
    else:
        allocation_data = {}
    
    # Generate HTML
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta http-equiv="refresh" content="900">  <!-- Auto-refresh every 15 minutes (900 seconds) -->
        <title>TradeBot Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{ padding: 20px; }}
            .card {{ margin-bottom: 20px; }}
            .metric-card {{ text-align: center; padding: 15px; }}
            .positive {{ color: green; }}
            .negative {{ color: red; }}
            .table-container {{ overflow-x: auto; }}
            .nav-tabs {{ margin-bottom: 15px; }}
        </style>
    </head>
    <body>
        <div class="container-fluid">
            <h1 class="mb-4">📊 TradeBot Dashboard</h1>
            <p class="lead">Real-time visualization of trading activities</p>
            
            <!-- Summary Metrics -->
            <div class="row mb-4">
                <div class="col-md-2">
                    <div class="card metric-card">
                        <h5>Total Positions</h5>
                        <h2>{total_positions}</h2>
                    </div>
                </div>
                <div class="col-md-2">
                    <div class="card metric-card">
                        <h5>Open Positions</h5>
                        <h2>{open_positions}</h2>
                    </div>
                </div>
                <div class="col-md-2">
                    <div class="card metric-card">
                        <h5>Closed Positions</h5>
                        <h2>{closed_position_count}</h2>
                    </div>
                </div>
                <div class="col-md-2">
                    <div class="card metric-card">
                        <h5>Total Spent</h5>
                        <h2>${total_amount_spent:.2f}</h2>
                    </div>
                </div>
                <div class="col-md-2">
                    <div class="card metric-card">
                        <h5>Current Value</h5>
                        <h2>${total_current_value:.2f}</h2>
                    </div>
                </div>
                <div class="col-md-2">
                    <div class="card metric-card">
                        <h5>Total Profit/Loss</h5>
                        <h2 class="{'positive' if total_profit_loss >= 0 else 'negative'}">
                            ${total_profit_loss:.2f}
                        </h2>
                        <p class="{'positive' if avg_profit_loss_percent >= 0 else 'negative'}">
                            {'+' if avg_profit_loss_percent >= 0 else ''}{avg_profit_loss_percent:.2f}%
                        </p>
                    </div>
                </div>
            </div>
            
            <!-- Transactions Tabs -->
            <div class="card mb-4">
                <div class="card-header">
                    <h4>📝 Transactions</h4>
                </div>
                <div class="card-body">
                    <ul class="nav nav-tabs" id="transactionTabs" role="tablist">
                        <li class="nav-item" role="presentation">
                            <button class="nav-link active" id="all-positions-tab" data-bs-toggle="tab" 
                                data-bs-target="#all-positions" type="button" role="tab" aria-selected="true">
                                All Positions
                            </button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="buy-sell-tab" data-bs-toggle="tab" 
                                data-bs-target="#buy-sell" type="button" role="tab" aria-selected="false">
                                Buy/Sell Pairs
                            </button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="recent-orders-tab" data-bs-toggle="tab" 
                                data-bs-target="#recent-orders" type="button" role="tab" aria-selected="false">
                                Recent Orders
                            </button>
                        </li>
                    </ul>
                    
                    <div class="tab-content" id="transactionTabsContent">
                        <!-- All Positions Tab -->
                        <div class="tab-pane fade show active" id="all-positions" role="tabpanel">
                            <div class="table-container">
                                <table class="table table-striped table-hover">
                                    <thead>
                                        <tr>
                                            <th>Symbol</th>
                                            <th>Quantity</th>
                                            <th>Buy Price</th>
                                            <th>Current Price</th>
                                            <th>Investment</th>
                                            <th>Current Value</th>
                                            <th>Status</th>
                                            <th>Buy Date</th>
                                            <th>Sell Price</th>
                                            <th>Sell Date</th>
                                        </tr>
                                    </thead>
                                    <tbody>
    """
    
    # Add all positions to the table
    for _, row in positions_df.iterrows():
        buy_date = pd.to_datetime(row['buy_date']).strftime('%Y-%m-%d %H:%M') if pd.notna(row['buy_date']) else ''
        sell_date = pd.to_datetime(row['sell_date']).strftime('%Y-%m-%d %H:%M') if pd.notna(row['sell_date']) else ''
        investment = row['buy_price'] * row['quantity']
        current_value = row['current_price'] * row['quantity']
        
        sell_price_display = f"${row['sell_price']:.2f}" if pd.notna(row['sell_price']) else ''
        
        html += f"""
                                        <tr>
                                            <td>{row['symbol']}</td>
                                            <td>{row['quantity']}</td>
                                            <td>${row['buy_price']:.2f}</td>
                                            <td>${row['current_price']:.2f}</td>
                                            <td>${investment:.2f}</td>
                                            <td>${current_value:.2f}</td>
                                            <td>{row['status']}</td>
                                            <td>{buy_date}</td>
                                            <td>{sell_price_display}</td>
                                            <td>{sell_date}</td>
                                        </tr>
        """
    
    html += """
                                    </tbody>
                                </table>
                            </div>
                        </div>
                        
                        <!-- Buy/Sell Pairs Tab -->
                        <div class="tab-pane fade" id="buy-sell" role="tabpanel">
                            <div class="table-container">
                                <table class="table table-striped table-hover">
                                    <thead>
                                        <tr>
                                            <th>Symbol</th>
                                            <th>Quantity</th>
                                            <th>Buy Price</th>
                                            <th>Sell Price</th>
                                            <th>Investment</th>
                                            <th>Return</th>
                                            <th>Profit/Loss</th>
                                            <th>P/L %</th>
                                            <th>Buy Date</th>
                                            <th>Sell Date</th>
                                            <th>Holding Period</th>
                                        </tr>
                                    </thead>
                                    <tbody>
    """
    
    # Add closed positions to the table
    if not closed_positions_df.empty:
        for _, row in closed_positions_df.iterrows():
            buy_date = pd.to_datetime(row['buy_date']).strftime('%Y-%m-%d %H:%M') if pd.notna(row['buy_date']) else ''
            sell_date = pd.to_datetime(row['sell_date']).strftime('%Y-%m-%d %H:%M') if pd.notna(row['sell_date']) else ''
            investment = row['buy_price'] * row['quantity']
            return_value = row['sell_price'] * row['quantity']
            
            # Calculate holding period
            if pd.notna(row['buy_date']) and pd.notna(row['sell_date']):
                buy_datetime = pd.to_datetime(row['buy_date'])
                sell_datetime = pd.to_datetime(row['sell_date'])
                holding_hours = (sell_datetime - buy_datetime).total_seconds() / 3600
                holding_period = f"{holding_hours:.2f} hours"
            else:
                holding_period = "N/A"
            
            profit_class = 'positive' if row['profit_loss'] >= 0 else 'negative'
            profit_sign = '+' if row['profit_loss'] >= 0 else ''
            
            html += f"""
                                        <tr>
                                            <td>{row['symbol']}</td>
                                            <td>{row['quantity']}</td>
                                            <td>${row['buy_price']:.2f}</td>
                                            <td>${row['sell_price']:.2f}</td>
                                            <td>${investment:.2f}</td>
                                            <td>${return_value:.2f}</td>
                                            <td class="{profit_class}">${profit_sign}{row['profit_loss']:.2f}</td>
                                            <td class="{profit_class}">{profit_sign}{row['profit_loss_percent']:.2f}%</td>
                                            <td>{buy_date}</td>
                                            <td>{sell_date}</td>
                                            <td>{holding_period}</td>
                                        </tr>
            """
    
    html += """
                                    </tbody>
                                </table>
                            </div>
                        </div>
                        
                        <!-- Recent Orders Tab -->
                        <div class="tab-pane fade" id="recent-orders" role="tabpanel">
                            <div class="table-container">
                                <table class="table table-striped table-hover">
                                    <thead>
                                        <tr>
                                            <th>Symbol</th>
                                            <th>Side</th>
                                            <th>Quantity</th>
                                            <th>Price</th>
                                            <th>Total Value</th>
                                            <th>Status</th>
                                            <th>Timestamp</th>
                                            <th>Source</th>
                                        </tr>
                                    </thead>
                                    <tbody>
    """
    
    # Add orders to the table
    for _, row in orders_df.iterrows():
        timestamp = pd.to_datetime(row['timestamp']).strftime('%Y-%m-%d %H:%M:%S') if pd.notna(row['timestamp']) else ''
        total_value = row['price'] * row['quantity']
        side_class = 'positive' if row['side'] == 'buy' else 'negative'
        
        html += f"""
                                        <tr>
                                            <td>{row['symbol']}</td>
                                            <td class="{side_class}">{row['side']}</td>
                                            <td>{row['quantity']}</td>
                                            <td>${row['price']:.2f}</td>
                                            <td>${total_value:.2f}</td>
                                            <td>{row['status']}</td>
                                            <td>{timestamp}</td>
                                            <td>{row['source']}</td>
                                        </tr>
        """
    
    html += """
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Bootstrap and Chart.js -->
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            // Only initialize charts if elements exist
            document.addEventListener('DOMContentLoaded', function() {
                // Profit/Loss by Symbol Chart
                const profitLossCtx = document.getElementById('profitLossChart');
                if (profitLossCtx) {
                    new Chart(profitLossCtx.getContext('2d'), {
                        type: 'bar',
                        data: {
                            labels: {json.dumps(list(profit_by_symbol.keys()))},
                            datasets: [{
                                label: 'Profit/Loss ($)',
                                data: {json.dumps(list(profit_by_symbol.values()))},
                                backgroundColor: {json.dumps(['#4CAF50' if v >= 0 else '#F44336' for v in profit_by_symbol.values()])},
                                borderColor: {json.dumps(['#388E3C' if v >= 0 else '#D32F2F' for v in profit_by_symbol.values()])},
                                borderWidth: 1
                            }]
                        },
                        options: {
                            responsive: true,
                            scales: {
                                y: {
                                    beginAtZero: true
                                }
                            }
                        }
                    });
                }
                
                // Trading Activity Chart
                const activityCtx = document.getElementById('activityChart');
                if (activityCtx) {
                    new Chart(activityCtx.getContext('2d'), {
                        type: 'line',
                        data: {
                            labels: {json.dumps(dates)},
                            datasets: [
                                {
                                    label: 'Buy Orders',
                                    data: {json.dumps(buy_counts)},
                                    backgroundColor: 'rgba(76, 175, 80, 0.2)',
                                    borderColor: '#4CAF50',
                                    borderWidth: 2,
                                    tension: 0.1
                                },
                                {
                                    label: 'Sell Orders',
                                    data: {json.dumps(sell_counts)},
                                    backgroundColor: 'rgba(244, 67, 54, 0.2)',
                                    borderColor: '#F44336',
                                    borderWidth: 2,
                                    tension: 0.1
                                }
                            ]
                        },
                        options: {
                            responsive: true,
                            scales: {
                                y: {
                                    beginAtZero: true
                                }
                            }
                        }
                    });
                }
                
                // Asset Allocation Chart
                const allocationCtx = document.getElementById('allocationChart');
                if (allocationCtx) {
                    new Chart(allocationCtx.getContext('2d'), {
                        type: 'pie',
                        data: {
                            labels: {json.dumps(list(allocation_data.keys()))},
                            datasets: [{
                                label: 'Asset Value ($)',
                                data: {json.dumps(list(allocation_data.values()))},
                                backgroundColor: [
                                    '#4CAF50', '#2196F3', '#FFC107', '#F44336', '#9C27B0',
                                    '#00BCD4', '#FF9800', '#795548', '#607D8B', '#E91E63'
                                ],
                                borderWidth: 1
                            }]
                        },
                        options: {
                            responsive: true
                        }
                    });
                }
            });
            
            // Initialize tabs and countdown timer
            document.addEventListener('DOMContentLoaded', function() {
                // Initialize tabs
                const triggerTabList = [].slice.call(document.querySelectorAll('#transactionTabs button'));
                triggerTabList.forEach(function(triggerEl) {
                    const tabTrigger = new bootstrap.Tab(triggerEl);
                    triggerEl.addEventListener('click', function(event) {
                        event.preventDefault();
                        tabTrigger.show();
                    });
                });
                
                // Initialize countdown timer for auto-refresh
                const countdownEl = document.getElementById('countdown');
                let minutes = 15;
                let seconds = 0;
                
                function updateCountdown() {
                    countdownEl.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
                    
                    if (minutes === 0 && seconds === 0) {
                        // Time to refresh
                        location.reload();
                        return;
                    }
                    
                    if (seconds === 0) {
                        minutes--;
                        seconds = 59;
                    } else {
                        seconds--;
                    }
                    
                    setTimeout(updateCountdown, 1000);
                }
                
                // Start the countdown
                updateCountdown();
            });
        </script>
    </body>
    </html>
    """
    
    return html

def main():
    """Main function to generate the dashboard"""
    print("Generating TradeBot Dashboard...")
    
    # Load data
    positions_df, orders_df = load_data()
    
    if positions_df is None or orders_df is None:
        print("Failed to load data. Cannot generate dashboard.")
        return
    
    # Generate dashboard
    html = generate_dashboard(positions_df, orders_df)
    
    # Write to file
    with open(OUTPUT_PATH, 'w') as f:
        f.write(html)
    
    print(f"Dashboard generated successfully at {OUTPUT_PATH}")
    print(f"Open {OUTPUT_PATH} in your web browser to view the dashboard")

if __name__ == "__main__":
    main()
