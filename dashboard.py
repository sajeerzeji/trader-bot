import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="TradeBot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("📊 TradeBot Dashboard")
st.markdown("### Real-time visualization of trading activities")

# Connect to the database
@st.cache_resource
def get_connection():
    """Create a connection to the SQLite database"""
    db_path = os.path.join('data', 'db', 'tradebot.db')
    if not os.path.exists(db_path):
        st.warning(f"Database not found at {db_path}. Using mock data for demonstration.")
        return None
    return sqlite3.connect(db_path)

# Load data from database
@st.cache_data(ttl=60)  # Cache for 60 seconds
def load_positions_data():
    """Load positions data from the database"""
    conn = get_connection()
    if conn is None:
        return pd.DataFrame()
    
    query = """
    SELECT * FROM positions
    ORDER BY buy_date DESC
    """
    try:
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Convert date strings to datetime objects
        if 'buy_date' in df.columns:
            df['buy_date'] = pd.to_datetime(df['buy_date'])
        if 'sell_date' in df.columns:
            df['sell_date'] = pd.to_datetime(df['sell_date'])
            
        return df
    except Exception as e:
        st.error(f"Error loading positions data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)  # Cache for 60 seconds
def load_orders_data():
    """Load orders data from the database"""
    conn = get_connection()
    if conn is None:
        return pd.DataFrame()
    
    query = """
    SELECT * FROM orders
    ORDER BY timestamp DESC
    """
    try:
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Convert date strings to datetime objects
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        return df
    except Exception as e:
        st.error(f"Error loading orders data: {e}")
        return pd.DataFrame()

# Calculate profit/loss
def calculate_profit_loss(positions_df):
    """Calculate profit/loss for closed positions"""
    if positions_df.empty:
        return pd.DataFrame()
    
    # Filter for closed positions
    closed_positions = positions_df[positions_df['status'] == 'closed'].copy()
    
    if closed_positions.empty:
        return pd.DataFrame()
    
    # Calculate profit/loss
    closed_positions['profit_loss'] = (closed_positions['sell_price'] - closed_positions['buy_price']) * closed_positions['quantity']
    closed_positions['profit_loss_percent'] = (closed_positions['sell_price'] - closed_positions['buy_price']) / closed_positions['buy_price'] * 100
    
    return closed_positions

# Create mock data if database doesn't exist yet
def create_mock_data():
    """Create mock data for demonstration purposes"""
    # Mock positions data
    positions_data = {
        'id': list(range(1, 11)),
        'symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'ADBE', 'CRM'],
        'quantity': [1.0, 2.0, 0.5, 0.3, 1.5, 2.5, 1.0, 0.8, 0.6, 1.2],
        'buy_price': [150.0, 250.0, 2000.0, 3000.0, 200.0, 400.0, 300.0, 500.0, 450.0, 200.0],
        'current_price': [160.0, 260.0, 2100.0, 3100.0, 210.0, 420.0, 310.0, 520.0, 460.0, 210.0],
        'buy_date': [datetime.now() - timedelta(days=i) for i in range(10)],
        'status': ['closed', 'closed', 'open', 'open', 'closed', 'open', 'closed', 'open', 'closed', 'open']
    }
    
    # Add sell data for closed positions
    positions_data['sell_price'] = [160.0, 260.0, None, None, 190.0, None, 310.0, None, 460.0, None]
    positions_data['sell_date'] = [datetime.now() - timedelta(days=i, hours=4) for i in range(10)]
    for i in range(len(positions_data['status'])):
        if positions_data['status'][i] == 'open':
            positions_data['sell_price'][i] = None
            positions_data['sell_date'][i] = None
    
    positions_df = pd.DataFrame(positions_data)
    
    # Mock orders data
    orders_data = {
        'id': list(range(1, 21)),
        'symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'ADBE', 'CRM'] * 2,
        'side': ['buy'] * 10 + ['sell'] * 10,
        'quantity': [1.0, 2.0, 0.5, 0.3, 1.5, 2.5, 1.0, 0.8, 0.6, 1.2] * 2,
        'price': [150.0, 250.0, 2000.0, 3000.0, 200.0, 400.0, 300.0, 500.0, 450.0, 200.0] + 
                 [160.0, 260.0, 2100.0, 3100.0, 190.0, 420.0, 310.0, 520.0, 460.0, 210.0],
        'timestamp': [datetime.now() - timedelta(days=i) for i in range(10)] + 
                     [datetime.now() - timedelta(days=i, hours=4) for i in range(10)],
        'status': ['filled'] * 20
    }
    
    orders_df = pd.DataFrame(orders_data)
    
    return positions_df, orders_df

# Load data
try:
    positions_df = load_positions_data()
    orders_df = load_orders_data()
    
    # If database doesn't exist yet, use mock data
    if positions_df.empty and orders_df.empty:
        st.warning("No data found in database. Using mock data for demonstration purposes.")
        positions_df, orders_df = create_mock_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    positions_df, orders_df = create_mock_data()

# Sidebar filters
st.sidebar.header("Filters")

# Date range filter
if not positions_df.empty and 'buy_date' in positions_df.columns:
    min_date = positions_df['buy_date'].min().date()
    max_date = datetime.now().date()
    
    start_date = st.sidebar.date_input("Start Date", min_date)
    end_date = st.sidebar.date_input("End Date", max_date)
    
    # Convert to datetime for filtering
    start_datetime = pd.Timestamp(start_date)
    end_datetime = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    
    # Filter dataframes by date
    positions_df = positions_df[(positions_df['buy_date'] >= start_datetime) & 
                               (positions_df['buy_date'] <= end_datetime)]

# Symbol filter
if not positions_df.empty and 'symbol' in positions_df.columns:
    all_symbols = sorted(positions_df['symbol'].unique())
    selected_symbols = st.sidebar.multiselect("Select Symbols", all_symbols, default=all_symbols)
    
    if selected_symbols:
        positions_df = positions_df[positions_df['symbol'].isin(selected_symbols)]
        if not orders_df.empty:
            orders_df = orders_df[orders_df['symbol'].isin(selected_symbols)]

# Status filter
if not positions_df.empty and 'status' in positions_df.columns:
    all_statuses = sorted(positions_df['status'].unique())
    selected_statuses = st.sidebar.multiselect("Select Status", all_statuses, default=all_statuses)
    
    if selected_statuses:
        positions_df = positions_df[positions_df['status'].isin(selected_statuses)]

# Calculate profit/loss for closed positions
closed_positions = calculate_profit_loss(positions_df)

# Dashboard layout
col1, col2, col3 = st.columns(3)

# Summary metrics
with col1:
    st.metric("Total Positions", len(positions_df) if not positions_df.empty else 0)
    
with col2:
    open_positions = len(positions_df[positions_df['status'] == 'open']) if not positions_df.empty else 0
    st.metric("Open Positions", open_positions)
    
with col3:
    closed_count = len(closed_positions) if not closed_positions.empty else 0
    st.metric("Closed Positions", closed_count)

# Profit/Loss metrics
if not closed_positions.empty:
    total_profit_loss = closed_positions['profit_loss'].sum()
    avg_profit_loss_percent = closed_positions['profit_loss_percent'].mean()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Profit/Loss", f"${total_profit_loss:.2f}", 
                 delta=f"{'+' if total_profit_loss > 0 else ''}{total_profit_loss:.2f}")
        
    with col2:
        st.metric("Average Profit/Loss %", f"{avg_profit_loss_percent:.2f}%", 
                 delta=f"{'+' if avg_profit_loss_percent > 0 else ''}{avg_profit_loss_percent:.2f}%")

# Transactions table
st.header("📝 Transactions")

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["All Positions", "Buy/Sell Pairs", "Recent Orders"])

with tab1:
    if not positions_df.empty:
        # Format the dataframe for display
        display_df = positions_df.copy()
        
        # Format currency columns
        currency_cols = ['buy_price', 'current_price', 'sell_price']
        for col in currency_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else None)
        
        # Calculate investment amount
        if 'buy_price' in display_df.columns and 'quantity' in display_df.columns:
            display_df['investment'] = positions_df['buy_price'] * positions_df['quantity']
            display_df['investment'] = display_df['investment'].apply(lambda x: f"${x:.2f}")
        
        # Calculate current value
        if 'current_price' in display_df.columns and 'quantity' in display_df.columns:
            display_df['current_value'] = positions_df['current_price'] * positions_df['quantity']
            display_df['current_value'] = display_df['current_value'].apply(lambda x: f"${x:.2f}")
        
        # Select columns to display
        cols_to_display = ['symbol', 'quantity', 'buy_price', 'current_price', 'investment', 
                          'current_value', 'status', 'buy_date']
        
        # Add sell columns for closed positions
        if 'sell_price' in display_df.columns and 'sell_date' in display_df.columns:
            cols_to_display.extend(['sell_price', 'sell_date'])
        
        # Display the table
        st.dataframe(display_df[cols_to_display], use_container_width=True)
    else:
        st.info("No position data available.")

with tab2:
    if not closed_positions.empty:
        # Format the dataframe for display
        display_df = closed_positions.copy()
        
        # Format currency columns
        currency_cols = ['buy_price', 'sell_price', 'profit_loss']
        for col in currency_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else None)
        
        # Format percentage columns
        if 'profit_loss_percent' in display_df.columns:
            display_df['profit_loss_percent'] = display_df['profit_loss_percent'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else None)
        
        # Calculate investment and return
        if 'buy_price' in display_df.columns and 'quantity' in display_df.columns:
            display_df['investment'] = closed_positions['buy_price'] * closed_positions['quantity']
            display_df['investment'] = display_df['investment'].apply(lambda x: f"${x:.2f}")
        
        if 'sell_price' in display_df.columns and 'quantity' in display_df.columns:
            display_df['return'] = closed_positions['sell_price'] * closed_positions['quantity']
            display_df['return'] = display_df['return'].apply(lambda x: f"${x:.2f}")
        
        # Calculate holding period
        if 'buy_date' in display_df.columns and 'sell_date' in display_df.columns:
            display_df['holding_period'] = (closed_positions['sell_date'] - closed_positions['buy_date']).dt.total_seconds() / 3600
            display_df['holding_period'] = display_df['holding_period'].apply(lambda x: f"{x:.2f} hours" if pd.notnull(x) else None)
        
        # Select columns to display
        cols_to_display = ['symbol', 'quantity', 'buy_price', 'sell_price', 'investment', 
                          'return', 'profit_loss', 'profit_loss_percent', 'holding_period',
                          'buy_date', 'sell_date']
        
        # Display the table
        st.dataframe(display_df[cols_to_display], use_container_width=True)
    else:
        st.info("No closed positions available.")

with tab3:
    if not orders_df.empty:
        # Format the dataframe for display
        display_df = orders_df.copy()
        
        # Format currency columns
        if 'price' in display_df.columns:
            display_df['price'] = display_df['price'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else None)
        
        # Calculate total value
        if 'price' in orders_df.columns and 'quantity' in orders_df.columns:
            display_df['total_value'] = orders_df['price'] * orders_df['quantity']
            display_df['total_value'] = display_df['total_value'].apply(lambda x: f"${x:.2f}")
        
        # Select columns to display
        cols_to_display = ['symbol', 'side', 'quantity', 'price', 'total_value', 'status', 'timestamp']
        
        # Display the table
        st.dataframe(display_df[cols_to_display], use_container_width=True)
    else:
        st.info("No order data available.")

# Visualizations
st.header("📊 Visualizations")

# Create tabs for different visualizations
viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Profit/Loss by Symbol", "Trading Activity", "Asset Allocation"])

with viz_tab1:
    if not closed_positions.empty:
        # Group by symbol and calculate total profit/loss
        profit_by_symbol = closed_positions.groupby('symbol')['profit_loss'].sum().reset_index()
        profit_by_symbol = profit_by_symbol.sort_values('profit_loss', ascending=False)
        
        # Create a bar chart
        fig = px.bar(profit_by_symbol, x='symbol', y='profit_loss',
                    title='Profit/Loss by Symbol',
                    labels={'profit_loss': 'Profit/Loss ($)', 'symbol': 'Symbol'},
                    color='profit_loss',
                    color_continuous_scale=['red', 'green'],
                    color_continuous_midpoint=0)
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No closed positions available for visualization.")

with viz_tab2:
    if not orders_df.empty:
        # Group by date and side, count orders
        orders_df['date'] = orders_df['timestamp'].dt.date
        activity_df = orders_df.groupby(['date', 'side']).size().reset_index(name='count')
        
        # Create a line chart
        fig = px.line(activity_df, x='date', y='count', color='side',
                     title='Trading Activity Over Time',
                     labels={'count': 'Number of Orders', 'date': 'Date', 'side': 'Side'},
                     color_discrete_map={'buy': 'green', 'sell': 'red'})
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No order data available for visualization.")

with viz_tab3:
    if not positions_df.empty:
        # Filter for open positions
        open_positions = positions_df[positions_df['status'] == 'open']
        
        if not open_positions.empty:
            # Calculate current value for each position
            open_positions['current_value'] = open_positions['current_price'] * open_positions['quantity']
            
            # Create a pie chart
            fig = px.pie(open_positions, values='current_value', names='symbol',
                        title='Current Asset Allocation',
                        hole=0.4)
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No open positions available for visualization.")
    else:
        st.info("No position data available for visualization.")

# Footer
st.markdown("---")
st.markdown("### 📈 TradeBot Dashboard")
st.markdown("Data refreshes automatically every 60 seconds.")
