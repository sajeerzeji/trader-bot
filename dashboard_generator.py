#!/usr/bin/env python3
"""
Dashboard Generator for TradeBot
Creates a static HTML dashboard from trading data in the database
"""

import sqlite3
import pandas as pd
import json
import os
import datetime
import random
from pathlib import Path

# Set up paths
DB_PATH = os.path.join('data', 'db', 'tradebot.db')
OUTPUT_PATH = 'dashboard.html'

# Create mock data for demonstration
def create_mock_data():
    """Create mock data for demonstration purposes"""
    # Mock positions data
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'ADBE', 'CRM']
    positions = []
    
    for i, symbol in enumerate(symbols):
        buy_date = datetime.datetime.now() - datetime.timedelta(days=i)
        status = 'closed' if i % 3 == 0 else 'open'
        quantity = round(random.uniform(0.5, 3.0), 2)
        buy_price = round(random.uniform(100, 500), 2)
        current_price = round(buy_price * random.uniform(0.9, 1.1), 2)
        
        position = {
            'id': i + 1,
            'symbol': symbol,
            'quantity': quantity,
            'buy_price': buy_price,
            'current_price': current_price,
            'buy_date': buy_date.isoformat(),
            'status': status
        }
        
        # Add sell data for closed positions
        if status == 'closed':
            sell_date = buy_date + datetime.timedelta(hours=random.randint(4, 24))
            sell_price = round(buy_price * random.uniform(0.95, 1.15), 2)
            position['sell_date'] = sell_date.isoformat()
            position['sell_price'] = sell_price
        
        positions.append(position)
    
    # Mock orders data
    orders = []
    order_id = 1
    
    for position in positions:
        # Buy order
        buy_order = {
            'id': order_id,
            'symbol': position['symbol'],
            'side': 'buy',
            'quantity': position['quantity'],
            'price': position['buy_price'],
            'timestamp': position['buy_date'],
            'status': 'filled'
        }
        orders.append(buy_order)
        order_id += 1
        
        # Sell order for closed positions
        if position['status'] == 'closed':
            sell_order = {
                'id': order_id,
                'symbol': position['symbol'],
                'side': 'sell',
                'quantity': position['quantity'],
                'price': position['sell_price'],
                'timestamp': position['sell_date'],
                'status': 'filled'
            }
            orders.append(sell_order)
            order_id += 1
    
    return positions, orders

# Set up paths
DB_PATH = os.path.join('data', 'db', 'tradebot.db')
OUTPUT_PATH = 'dashboard.html'

# Load data from database
def load_data():
    """Load data from the database or generate mock data if not available"""
    try:
        if not os.path.exists(DB_PATH):
            print(f"Database not found at {DB_PATH}. Using mock data.")
            return create_mock_data()
        
        conn = sqlite3.connect(DB_PATH)
        
        # Load positions
        positions_query = "SELECT * FROM positions ORDER BY buy_date DESC"
        positions_df = pd.read_sql_query(positions_query, conn)
        
        # Load orders
        orders_query = "SELECT * FROM orders ORDER BY timestamp DESC"
        orders_df = pd.read_sql_query(orders_query, conn)
        
        conn.close()
        
        # Convert to list of dictionaries
        positions = positions_df.to_dict('records')
        orders = orders_df.to_dict('records')
        
        # Convert datetime objects to strings and handle NaN values
        for position in positions:
            for key, value in position.items():
                if isinstance(value, pd.Timestamp):
                    position[key] = value.isoformat()
                elif pd.isna(value):
                    position[key] = None
        
        for order in orders:
            for key, value in order.items():
                if isinstance(value, pd.Timestamp):
                    order[key] = value.isoformat()
                elif pd.isna(value):
                    order[key] = None
        
        print(f"Loaded {len(positions)} positions and {len(orders)} orders from database.")
        return positions, orders
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return create_mock_data()

# Calculate profit/loss for closed positions
def calculate_profit_loss(positions):
    """Calculate profit/loss for closed positions"""
    closed_positions = []
    
    for position in positions:
        if position['status'] == 'closed':
            # Calculate profit/loss
            buy_price = position['buy_price']
            sell_price = position['sell_price']
            quantity = position['quantity']
            
            profit_loss = (sell_price - buy_price) * quantity
            profit_loss_percent = (sell_price - buy_price) / buy_price * 100
            
            position['profit_loss'] = round(profit_loss, 2)
            position['profit_loss_percent'] = round(profit_loss_percent, 2)
            closed_positions.append(position)
    
    return closed_positions

# Generate HTML dashboard
def generate_dashboard(positions, orders):
    """Generate HTML dashboard from trading data"""
    # Calculate profit/loss for closed positions
    closed_positions = calculate_profit_loss([p for p in positions if p['status'] == 'closed'])
    
    # Calculate summary metrics
    total_positions = len(positions)
    open_positions = len([p for p in positions if p['status'] == 'open'])
    closed_position_count = len(closed_positions)
    
    total_profit_loss = sum(p['profit_loss'] for p in closed_positions) if closed_positions else 0
    avg_profit_loss_percent = sum(p['profit_loss_percent'] for p in closed_positions) / len(closed_positions) if closed_positions else 0
    
    # Prepare data for charts
    profit_by_symbol = {}
    for position in closed_positions:
        symbol = position['symbol']
        profit_loss = position['profit_loss']
        
        if symbol in profit_by_symbol:
            profit_by_symbol[symbol] += profit_loss
        else:
            profit_by_symbol[symbol] = profit_loss
    
    # Sort profit_by_symbol by value
    profit_by_symbol = {k: v for k, v in sorted(profit_by_symbol.items(), key=lambda item: item[1], reverse=True)}
    
    # Prepare data for trading activity chart
    activity_data = {}
    for order in orders:
        date = order['timestamp'].split('T')[0]
        side = order['side']
        
        if date not in activity_data:
            activity_data[date] = {'buy': 0, 'sell': 0}
        
        activity_data[date][side] += 1
    
    # Sort activity_data by date
    activity_data = {k: v for k, v in sorted(activity_data.items())}
    
    # Prepare data for asset allocation chart
    open_position_values = {}
    for position in positions:
        if position['status'] == 'open':
            symbol = position['symbol']
            value = position['current_price'] * position['quantity']
            
            open_position_values[symbol] = round(value, 2)
    
    # Generate HTML
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
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
                <div class="col-md-3">
                    <div class="card metric-card">
                        <h5>Total Positions</h5>
                        <h2>{total_positions}</h2>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card metric-card">
                        <h5>Open Positions</h5>
                        <h2>{open_positions}</h2>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card metric-card">
                        <h5>Closed Positions</h5>
                        <h2>{closed_position_count}</h2>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card metric-card">
                        <h5>Total Profit/Loss</h5>
                        <h2 class="{('positive' if total_profit_loss >= 0 else 'negative')}">
                            ${total_profit_loss:.2f}
                        </h2>
                        <p class="{('positive' if avg_profit_loss_percent >= 0 else 'negative')}">
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
    for position in positions:
        buy_date = position['buy_date'].split('T')[0] if 'buy_date' in position else ''
        sell_date = position['sell_date'].split('T')[0] if 'sell_date' in position and position['sell_date'] else ''
        investment = position['buy_price'] * position['quantity']
        current_value = position['current_price'] * position['quantity']
        
        html += f"""
                                        <tr>
                                            <td>{position['symbol']}</td>
                                            <td>{position['quantity']}</td>
                                            <td>${position['buy_price']:.2f}</td>
                                            <td>${position['current_price']:.2f}</td>
                                            <td>${investment:.2f}</td>
                                            <td>${current_value:.2f}</td>
                                            <td>{position['status']}</td>
                                            <td>{buy_date}</td>
                                            <td>${position.get('sell_price', ''):.2f if 'sell_price' in position else ''}</td>
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
    for position in closed_positions:
        buy_date = position['buy_date'].split('T')[0] if 'buy_date' in position else ''
        sell_date = position['sell_date'].split('T')[0] if 'sell_date' in position else ''
        investment = position['buy_price'] * position['quantity']
        return_value = position['sell_price'] * position['quantity']
        
        # Calculate holding period
        buy_datetime = datetime.datetime.fromisoformat(position['buy_date'])
        sell_datetime = datetime.datetime.fromisoformat(position['sell_date'])
        holding_hours = (sell_datetime - buy_datetime).total_seconds() / 3600
        
        profit_class = 'positive' if position['profit_loss'] >= 0 else 'negative'
        profit_sign = '+' if position['profit_loss'] >= 0 else ''
        
        html += f"""
                                        <tr>
                                            <td>{position['symbol']}</td>
                                            <td>{position['quantity']}</td>
                                            <td>${position['buy_price']:.2f}</td>
                                            <td>${position['sell_price']:.2f}</td>
                                            <td>${investment:.2f}</td>
                                            <td>${return_value:.2f}</td>
                                            <td class="{profit_class}">${profit_sign}{position['profit_loss']:.2f}</td>
                                            <td class="{profit_class}">{profit_sign}{position['profit_loss_percent']:.2f}%</td>
                                            <td>{buy_date}</td>
                                            <td>{sell_date}</td>
                                            <td>{holding_hours:.2f} hours</td>
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
                                        </tr>
                                    </thead>
                                    <tbody>
    """
    
    # Add orders to the table
    for order in orders:
        timestamp = order['timestamp'].replace('T', ' ').split('.')[0] if 'timestamp' in order else ''
        total_value = order['price'] * order['quantity']
        side_class = 'positive' if order['side'] == 'buy' else 'negative'
        
        html += f"""
                                        <tr>
                                            <td>{order['symbol']}</td>
                                            <td class="{side_class}">{order['side']}</td>
                                            <td>{order['quantity']}</td>
                                            <td>${order['price']:.2f}</td>
                                            <td>${total_value:.2f}</td>
                                            <td>{order['status']}</td>
                                            <td>{timestamp}</td>
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
            
            <!-- Visualizations -->
            <div class="row">
                <!-- Profit/Loss by Symbol -->
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-header">
                            <h4>Profit/Loss by Symbol</h4>
                        </div>
                        <div class="card-body">
                            <canvas id="profitLossChart"></canvas>
                        </div>
                    </div>
                </div>
                
                <!-- Trading Activity -->
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-header">
                            <h4>Trading Activity</h4>
                        </div>
                        <div class="card-body">
                            <canvas id="activityChart"></canvas>
                        </div>
                    </div>
                </div>
                
                <!-- Asset Allocation -->
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-header">
                            <h4>Asset Allocation</h4>
                        </div>
                        <div class="card-body">
                            <canvas id="allocationChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Footer -->
            <footer class="mt-4 text-center">
                <p>TradeBot Dashboard - Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><small>Data refreshes when page is reloaded</small></p>
            </footer>
        </div>
        
        <!-- Bootstrap and Chart.js -->
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            // Profit/Loss by Symbol Chart
            const profitLossCtx = document.getElementById('profitLossChart').getContext('2d');
            const profitLossChart = new Chart(profitLossCtx, {
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
            
            // Trading Activity Chart
            const activityCtx = document.getElementById('activityChart').getContext('2d');
            const activityChart = new Chart(activityCtx, {
                type: 'line',
                data: {
                    labels: {json.dumps(list(activity_data.keys()))},
                    datasets: [
                        {
                            label: 'Buy Orders',
                            data: {json.dumps([d['buy'] for d in activity_data.values()])},
                            backgroundColor: 'rgba(76, 175, 80, 0.2)',
                            borderColor: '#4CAF50',
                            borderWidth: 2,
                            tension: 0.1
                        },
                        {
                            label: 'Sell Orders',
                            data: {json.dumps([d['sell'] for d in activity_data.values()])},
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
            
            // Asset Allocation Chart
            const allocationCtx = document.getElementById('allocationChart').getContext('2d');
            const allocationChart = new Chart(allocationCtx, {
                type: 'pie',
                data: {
                    labels: {json.dumps(list(open_position_values.keys()))},
                    datasets: [{
                        label: 'Asset Value ($)',
                        data: {json.dumps(list(open_position_values.values()))},
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
            
            // Initialize tabs
            document.addEventListener('DOMContentLoaded', function() {
                const triggerTabList = [].slice.call(document.querySelectorAll('#transactionTabs button'));
                triggerTabList.forEach(function(triggerEl) {
                    const tabTrigger = new bootstrap.Tab(triggerEl);
                    triggerEl.addEventListener('click', function(event) {
                        event.preventDefault();
                        tabTrigger.show();
                    });
                });
            });
        </script>
    </body>
    </html>
    """
    
    return html

# Main function
def main():
    """Main function to generate the dashboard"""
    print("Generating TradeBot Dashboard...")
    
    # Load data
    positions, orders = load_data()
    
    # Generate dashboard
    html = generate_dashboard(positions, orders)
    
    # Write to file
    with open(OUTPUT_PATH, 'w') as f:
        f.write(html)
    
    print(f"Dashboard generated successfully at {OUTPUT_PATH}")
    print(f"Open {OUTPUT_PATH} in your web browser to view the dashboard")

if __name__ == "__main__":
    main()
