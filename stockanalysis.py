import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta

# Stock ticker (changeable)
stock = "SPY"

# Timeframes and candle counts
timeframes = {
    "1m": "1m",
    "5m": "5m",
    "30m": "30m"
}
candle_counts = [10, 50, 200]

# Function to fetch data
def fetch_data(ticker, period, interval):
    try:
        stock_data = yf.Ticker(ticker)
        # Adjust period to ensure enough data (yfinance has limits on intraday data)
        if interval == "1min":
            period = "7d"  # Max for 1min data
        elif interval == "5min":
            period = "60d"
        else:
            period = "60d"
        df = stock_data.history(period=period, interval=interval)
        return df
    except Exception as e:
        print(f"Error fetching data for {interval}: {e}")
        return None

# Function to calculate stats
def calculate_stats(data, count):
    if data is None or len(data) < count:
        return None
    data = data.tail(count)
    stats = {
        "High": {
            "Mean": data["High"].mean(),
            "Median": data["High"].median(),
            "Variance": data["High"].var()
        },
        "Low": {
            "Mean": data["Low"].mean(),
            "Median": data["Low"].median(),
            "Variance": data["Low"].var()
        },
        "Close": {
            "Mean": data["Close"].mean(),
            "Median": data["Close"].median(),
            "Variance": data["Close"].var()
        },
        "Volume": {
            "Mean": data["Volume"].mean(),
            "Median": data["Volume"].median(),
            "Variance": data["Volume"].var()
        }
    }
    return stats

# Function to calculate probabilities
def calculate_probabilities(data, count, current_close):
    if data is None or len(data) < count:
        return None
    data = data.tail(count)
    # Calculate close-to-close price changes
    price_changes = data["Close"].diff().dropna()
    mean_change = price_changes.mean()
    std_change = price_changes.std()
    
    # Use Gaussian distribution
    # Probability of increase: P(next close > current close)
    prob_increase = 1 - norm.cdf(0, mean_change, std_change)
    # Probability of decrease: P(next close < current close)
    prob_decrease = norm.cdf(0, mean_change, std_change)
    # Range thresholds: ±1.5 standard deviations from current close
    range_lower = current_close - 1.5 * std_change
    range_upper = current_close + 1.5 * std_change
    # Probability of staying in range: P(range_lower < next close < range_upper)
    prob_range = norm.cdf(range_upper - current_close, mean_change, std_change) - norm.cdf(range_lower - current_close, mean_change, std_change)
    
    return {
        "Increase": prob_increase * 100,
        "Decrease": prob_decrease * 100,
        "Range": prob_range * 100,
        "Range_Lower": range_lower,
        "Range_Upper": range_upper
    }

# Main analysis
print(f"Stock Analysis for {stock}\n")
for tf_name, tf_interval in timeframes.items():
    print(f"\nTimeframe: {tf_name}")
    data = fetch_data(stock, "60d", tf_interval)
    if data is None:
        print("No data available.")
        continue
    current_close = data["Close"].iloc[-1] if not data.empty else None
    
    for count in candle_counts:
        print(f"\nCandle Count: {count}")
        # Calculate stats
        stats = calculate_stats(data, count)
        if stats:
            print("Statistics:")
            for metric, values in stats.items():
                print(f"  {metric}:")
                print(f"    Mean: {values['Mean']:.2f}")
                print(f"    Median: {values['Median']:.2f}")
                print(f"    Variance: {values['Variance']:.2f}")
        else:
            print("  Insufficient data for stats.")
        
        # Calculate probabilities
        probs = calculate_probabilities(data, count, current_close)
        if probs:
            print("Probabilities:")
            print(f"  Increase: {probs['Increase']:.2f}%")
            print(f"  Decrease: {probs['Decrease']:.2f}%")
            print(f"  Range: {probs['Range']:.2f}%")
            print(f"  Range Thresholds: {probs['Range_Lower']:.2f} - {probs['Range_Upper']:.2f}")
        else:
            print("  Insufficient data for probabilities.")