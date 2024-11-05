import pandas as pd
import yfinance as yf
import numpy as np

import yfinance as yf
import pandas as pd

# List of cryptocurrency tickers
tickers = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD"]

# List to store each ticker's historical data
all_data = []

# Loop through each ticker and fetch its history
for ticker in tickers:
    crypto = yf.Ticker(ticker)
    history = crypto.history(start="2022-11-01", end="2024-11-01")  # Set your date range
   # print(f'{ticker}: {history.index.min()}')
    history['Ticker'] = ticker  # Add a column for the ticker symbol
    all_data.append(history)

# Concatenate all data into a single DataFrame
df = pd.concat(all_data)

# Optional: Reset index if you want a clean numerical index
df.reset_index(inplace=True)


def calculate_portfolio_metrics(df, tickers, weights):
    selected_data = df[df['Ticker'].isin(tickers)]
    selected_data = selected_data.pivot(index='Date', columns='Ticker', values='Close')
    daily_returns = selected_data.pct_change().dropna()
    #  print(daily_returns.head(4))
    weights = np.array(weights) / np.sum(weights)
    # weights = np.array(weights) / np.sum(weights)
    daily_returns['portfolio_return'] = daily_returns.dot(weights)
    # print(daily_returns.head(4))
    #  initial_value = 100  # Starting portfolio value, adjust if needed
    daily_returns['cumprod_portfolio_return'] = (1 + daily_returns['portfolio_return']).cumprod()
    # print(daily_returns.head(4))

    # Calculate the rolling 30-day peak of the cumulative portfolio return
    daily_returns['monthly_peak'] = daily_returns['cumprod_portfolio_return'].rolling('30d').max()

    # Calculate the drawdown as the current cumulative return divided by the 30-day rolling peak
    daily_returns['drawdown'] = daily_returns['cumprod_portfolio_return'] / daily_returns['monthly_peak']

    # Calculate the monthly drawdown (minimum drawdown over the last 30 days)
    daily_returns['monthly_drawdown'] = daily_returns['drawdown'].rolling('30d').min()

    # Calculate the 5% Value at Risk (VaR) over a rolling 250-day window for portfolio returns
    daily_returns['var95'] = daily_returns['portfolio_return'].rolling(250).quantile(0.05)

    return daily_returns



import streamlit as st
import plotly.graph_objects as go
#st.set_page_config(layout="wide")
# Define the function to calculate portfolio metrics
def calculate_portfolio_metrics(df, tickers, weights):
    selected_data = df[df['Ticker'].isin(tickers)]
    selected_data = selected_data.pivot(index='Date', columns='Ticker', values='Close')
    daily_returns = selected_data.pct_change().dropna()

    # Normalize weights
    weights = np.array(weights) / np.sum(weights)
    daily_returns['portfolio_return'] = daily_returns[tickers].dot(weights)
    daily_returns['cumprod_portfolio_return'] = (1 + daily_returns['portfolio_return']).cumprod()
    daily_returns['monthly_peak'] = daily_returns['cumprod_portfolio_return'].rolling('30d').max()
    daily_returns['drawdown'] = daily_returns['cumprod_portfolio_return'] / daily_returns['monthly_peak'] - 1
    daily_returns['monthly_drawdown'] = daily_returns['drawdown'].rolling('30d').min()
    daily_returns['var95'] = daily_returns['portfolio_return'].rolling(250).quantile(0.05)
    return daily_returns


# Streamlit app layout
#st.title("Choose the Strategy")

# Sidebar for filters
st.sidebar.header("Crypto NewWealth Management")
st.sidebar.image("pngwing.com.png", use_column_width=True)
st.sidebar.header("")
st.sidebar.header("Strategy Filters")
initial_value = st.sidebar.number_input("Initial Investment Amount", min_value=0.0, value=1000.0, step=10.0)
# Get unique tickers from data and add a multiselect for tickers in the sidebar
unique_tickers = df['Ticker'].unique().tolist()
selected_tickers = st.sidebar.multiselect("Select Tickers for Portfolio", unique_tickers, default=unique_tickers[:2])

weights = []
for ticker in selected_tickers:
    weight = st.sidebar.number_input(f"Weight for {ticker}", min_value=0.0, max_value=1.0, value=1.0 / len(selected_tickers), key=ticker)
    weights.append(weight)

# Normalize weights to ensure they sum up to 1
weights = np.array(weights) / np.sum(weights)

# Normalize weights to ensure they sum up to 1
weights = np.array(weights) / np.sum(weights)

st.title("Choose the Strategy")

# Create tabs
tabs = st.tabs(["Choose the Strategy", "Ready Strategies"])

# Portfolio Dashboard Tab
with tabs[0]:
    # Calculate portfolio metrics with selected tickers and weights
    if selected_tickers and len(weights) == len(selected_tickers):
        daily_returns = calculate_portfolio_metrics(df, selected_tickers, weights)

        # Plot cumulative portfolio return
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_returns.index,
            y=daily_returns['cumprod_portfolio_return'] * 100,  # Assuming initial value of 100
            mode='lines',
            name='Cumulative Portfolio Return'
        ))
        fig.update_layout(
            title="Cumulative Portfolio Return Over Time",
            xaxis_title="Date",
            yaxis_title="Portfolio Value"
        )
        st.plotly_chart(fig)

        # Display summary metrics
        pnl = daily_returns['cumprod_portfolio_return'].iloc[-1] - 1
        min_drawdown = daily_returns['monthly_drawdown'].min()
        var_95 = daily_returns['var95'].iloc[-1]
        summary_data = {
            'PnL': [round(pnl * 100, 2)],  # Percentage format
            'Min DrawDown': [round(min_drawdown * 100, 2)],  # Percentage format
            'VaR 95%': [round(var_95 * 100, 2)]  # Percentage format
        }
        summary_df = pd.DataFrame(summary_data)
        st.table(summary_df)

    # Button to buy portfolio
    st.markdown(
        """
        <style>
        .big-button {
            display: inline-block;
            background-color: #4CAF50; /* Green background */
            color: white; /* White text */
            padding: 15px 32px; /* Padding around the text */
            text-align: center; /* Center the text */
            text-decoration: none; /* Remove underline */
            font-size: 20px; /* Large font size */
            margin: 4px 2px; /* Margin around the button */
            cursor: pointer; /* Pointer cursor on hover */
            border: none; /* No border */
            border-radius: 4px; /* Rounded corners */
        }
        </style>
        <button class="big-button" onclick="document.getElementById('buy-portfolio').click();">Buy Portfolio</button>
        <input type="button" id="buy-portfolio" value="Buy Portfolio" style="display: none;" />
        """,
        unsafe_allow_html=True
    )

    # if st.button("Buy Portfolio", key='buy_portfolio'):
    #     st.success("Portfolio purchase successful!")  # Confirmation message

# Ready Strategies Tab
with tabs[1]:
    st.header("Ready Strategies")
  #  st.write("Here you can add information about ready-to-use strategies.")
    # Add your strategy details here. For example:
    st.subheader("Strategy 1: Top 1000 Coins")
    st.write("Широкий спред токенов, купленные пропорционально их ликвидности")

    st.subheader("Strategy 2: MobyDik")
    st.write("Стратегия, следующая сигналам китовых кошельков")

    st.subheader("Strategy 3: 150 Tech Indicators")
    st.write("Роботизированная стратегия, основанная на популярных тех индикатарах")

    st.subheader("Strategy 4: ChatGPT")
    st.write("Top-100 ликвидных компаний , разрабтывающих LLM модели")
