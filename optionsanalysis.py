import requests
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import time

# Ticker (changeable: SPY or QQQ)
ticker = "SPY"
api_key = "YOUR_MARKETDATA_API_KEY"  # Get free key at https://marketdata.app/

# Analysis periods (trading days, similar to candle counts)
periods = [10, 50, 200]

# Function to fetch options chain data
def fetch_options_data(ticker, api_key, date=None):
    try:
        url = f"https://api.marketdata.app/v1/options/chain/{ticker}/"
        params = {"token": api_key}
        if date:
            params["date"] = date
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124",
            "Accept": "application/json"
        }
        
        print(f"Fetching URL: {url} with params: {params}")
        response = requests.get(url, params=params, headers=headers)
        print(f"Status Code: {response.status_code}")
        response.raise_for_status()
        
        data = response.json()
        if not data or "optionSymbol" not in data:
            print(f"No valid data for {ticker} on {date or 'today'}.")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(data)
        required_cols = ["optionSymbol", "strike", "expiration", "side", "delta", "gamma", 
                        "theta", "vega", "rho", "volume", "mid"]
        if not all(col in df.columns for col in required_cols):
            print(f"Missing required columns for {ticker}.")
            return None
        
        return df[required_cols]
    except requests.exceptions.HTTPError as e:
        if response.status_code == 429:
            print(f"Rate limit exceeded for {ticker}. Retry later.")
        elif response.status_code == 401:
            print(f"401 Unauthorized for {ticker}. Check API key at https://marketdata.app/.")
        return None
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

# Function to fetch historical data for a period
def fetch_historical_data(ticker, api_key, days):
    historical_data = []
    end_date = datetime.now()
    days_fetched = 0
    max_requests = 100  # Free tier limit
    
    while days_fetched < days and len(historical_data) < max_requests:
        date_str = (end_date - timedelta(days=days_fetched)).strftime("%Y-%m-%d")
        data = fetch_options_data(ticker, api_key, date_str)
        if data is not None:
            data["date"] = date_str
            historical_data.append(data)
        days_fetched += 1
        time.sleep(1)  # Avoid rate limits
        
        # Skip weekends (options markets closed)
        while (end_date - timedelta(days=days_fetched)).weekday() >= 5:
            days_fetched += 1
    
    return pd.concat(historical_data, ignore_index=True) if historical_data else None

# Function to calculate stats for Greeks and volume
def calculate_stats(data, days):
    if data is None or len(data["date"].unique()) < days:
        return None
    
    # Get last N trading days
    unique_dates = sorted(data["date"].unique())[-days:]
    data = data[data["date"].isin(unique_dates)]
    
    stats = {
        "Delta": {
            "Mean": data["delta"].mean(),
            "Median": data["delta"].median(),
            "Variance": data["delta"].var()
        },
        "Gamma": {
            "Mean": data["gamma"].mean(),
            "Median": data["gamma"].median(),
            "Variance": data["gamma"].var()
        },
        "Theta": {
            "Mean": data["theta"].mean(),
            "Median": data["theta"].median(),
            "Variance": data["theta"].var()
        },
        "Vega": {
            "Mean": data["vega"].mean(),
            "Median": data["vega"].median(),
            "Variance": data["vega"].var()
        },
        "Rho": {
            "Mean": data["rho"].mean(),
            "Median": data["rho"].median(),
            "Variance": data["rho"].var()
        },
        "Volume": {
            "Mean": data["volume"].mean(),
            "Median": data["volume"].median(),
            "Variance": data["volume"].var()
        }
    }
    return stats

# Function to calculate probabilities
def calculate_probabilities(data, days, current_mid):
    if data is None or len(data["date"].unique()) < days:
        return None
    
    # Get mid-price changes across chain
    unique_dates = sorted(data["date"].unique())[-days:]
    data = data[data["date"].isin(unique_dates)]
    mid_changes = data.groupby("date")["mid"].mean().diff().dropna()
    
    mean_change = mid_changes.mean()
    std_change = mid_changes.std()
    
    prob_increase = (1 - norm.cdf(0, mean_change, std_change)) * 100
    prob_decrease = norm.cdf(0, mean_change, std_change) * 100
    range_lower = current_mid - 1.5 * std_change
    range_upper = current_mid + 1.5 * std_change
    prob_range = (norm.cdf(range_upper - current_mid, mean_change, std_change) - 
                  norm.cdf(range_lower - current_mid, mean_change, std_change)) * 100
    
    return {
        "Increase": prob_increase,
        "Decrease": prob_decrease,
        "Range": prob_range,
        "Range_Lower": range_lower,
        "Range_Upper": range_upper
    }

# Function to suggest strategies
def suggest_strategies(data):
    if data is None:
        return "No data for strategy suggestions."
    
    avg_vega = data["vega"].mean()
    avg_delta = data["delta"].mean()
    avg_volume = data["volume"].mean()
    
    strategies = []
    if avg_vega > 0.1:  # High volatility sensitivity
        strategies.append("Long Straddle: Buy ATM call + put to capitalize on large price swings (high vega).")
    if abs(avg_delta) < 0.3:  # Low directional bias
        strategies.append("Iron Condor: Sell OTM call/put spreads for range-bound markets (low delta).")
    if avg_volume > 1000:  # Liquid options
        strategies.append("Covered Call: Own underlying, sell OTM calls for income (high volume).")
    if avg_delta > 0.5:  # Bullish bias
        strategies.append("Protective Put: Own underlying, buy OTM puts to hedge downside (positive delta).")
    
    return strategies if strategies else ["No clear strategy based on current Greeks/volume."]

# Main analysis
if ticker not in ["SPY", "QQQ"]:
    print(f"Unsupported ticker: {ticker}. Available: SPY, QQQ")
else:
    print(f"Options Analysis for {ticker}\n")
    for period in periods:
        print(f"Period: Last {period} Trading Days")
        # Fetch historical data
        data = fetch_historical_data(ticker, api_key, period)
        if data is None:
            print("  No data available.")
            continue
        
        # Current mid-price (average across chain)
        current_data = fetch_options_data(ticker, api_key)
        current_mid = current_data["mid"].mean() if current_data is not None else None
        
        # Calculate stats
        stats = calculate_stats(data, period)
        if stats:
            print("  Statistics:")
            for metric, values in stats.items():
                print(f"    {metric}: Mean={values['Mean']:.4f}, Median={values['Median']:.4f}, Variance={values['Variance']:.4f}")
        else:
            print("    Insufficient data for stats.")
        
        # Calculate probabilities
        probs = calculate_probabilities(data, period, current_mid)
        if probs:
            print("  Probabilities (Mid-Price):")
            print(f"    Increase: {probs['Increase']:.2f}%")
            print(f"    Decrease: {probs['Decrease']:.2f}%")
            print(f"    Range: {probs['Range']:.2f}%")
            print(f"    Range Thresholds: {probs['Range_Lower']:.2f} - {probs['Range_Upper']:.2f}")
        else:
            print("    Insufficient data for probabilities.")
        
        # Suggest strategies
        strategies = suggest_strategies(current_data)
        print("  Strategy Suggestions:")
        for strategy in strategies:
            print(f"    - {strategy}")
        
        time.sleep(2)  # Avoid rate limits