import requests
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime
import time

# Crypto ticker (changeable)
crypto_pair = "BTC-USD"
base_currency = "USD"
api_key = "aa7062059e695152e5c69ecb8ec99e51fef3a5f5cbe1f72300dfb39d50eb7e6c"  # Get free key at https://min-api.cryptocompare.com/

# Map tickers to CryptoCompare symbols
ticker_map = {
    "BTC-USD": "BTC",
    "ETH-USD": "ETH",
    "SOL-USD": "SOL",
    "DOGE-USD": "DOGE",
    "TRON-USD": "TRX",
    "BNB-USD": "BNB"
}

# Timeframes (seconds per candle) and candle counts
timeframes = {
    "1m": 60,
    "5m": 300,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
    "1w": 604800
}
candle_counts = [10, 50, 200]

# Function to fetch data
def fetch_data(coin, vs_currency, timeframe_seconds, max_candles):
    try:
        # Select endpoint based on timeframe
        if timeframe_seconds <= 1800:  # 1m, 5m, 30m
            endpoint = "histominute"
            aggregate = timeframe_seconds // 60  # Minutes
        elif timeframe_seconds <= 14400:  # 1h, 4h
            endpoint = "histohour"
            aggregate = timeframe_seconds // 3600  # Hours
        else:  # 1d, 1w
            endpoint = "histoday"
            aggregate = 1 if timeframe_seconds == 86400 else 7  # Days or weeks (resampled)

        url = f"https://min-api.cryptocompare.com/data/v2/{endpoint}"
        params = {
            "fsym": coin,
            "tsym": vs_currency,
            "limit": max_candles,
            "aggregate": aggregate,
            "api_key": api_key
        }
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124",
            "Accept": "application/json"
        }
        
        print(f"Fetching URL: {url} with params: {params}")
        for attempt in range(3):
            try:
                response = requests.get(url, params=params, headers=headers)
                print(f"Status Code: {response.status_code}")
                response.raise_for_status()
                break
            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:
                    print(f"Rate limit exceeded for {timeframe_seconds}s. Retrying in 5s...")
                    time.sleep(5)
                elif response.status_code == 401:
                    print(f"401 Unauthorized for {timeframe_seconds}s. Check API key at https://min-api.cryptocompare.com/.")
                    return None
                else:
                    raise e
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {timeframe_seconds}s: {e}")
                if attempt == 2:
                    return None
                time.sleep(2)
        
        try:
            data = response.json()
        except ValueError:
            print(f"Invalid JSON response for {timeframe_seconds}s timeframe.")
            return None
        
        candles = data.get("Data", {}).get("Data", [])
        if not candles or len(candles) < 2:
            print(f"No valid data returned for {timeframe_seconds}s timeframe.")
            return None
        
        # Create DataFrame
        timestamps = [datetime.fromtimestamp(c["time"]) for c in candles]
        df = pd.DataFrame({
            "High": [c["high"] for c in candles],
            "Low": [c["low"] for c in candles],
            "Close": [c["close"] for c in candles],
            "Volume": [c["volumeto"] for c in candles]  # volumeto is in quote currency (USD)
        }, index=timestamps)
        
        # Resample for 1w if needed (daily data aggregated to weekly)
        if timeframe_seconds == 604800:
            df = df.resample("7D").agg({
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum"
            }).dropna()
        
        return df.tail(max_candles)
    except Exception as e:
        print(f"Error fetching data for {timeframe_seconds}s: {e}")
        return None

# Function to calculate stats
def calculate_stats(data, count):
    if data is None or len(data) < count:
        return None
    data = data.tail(count)
    return {
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

# Function to calculate probabilities
def calculate_probabilities(data, count, current_close):
    if data is None or len(data) < count:
        return None
    data = data.tail(count)
    price_changes = data["Close"].diff().dropna()
    mean_change = price_changes.mean()
    std_change = price_changes.std()
    
    prob_increase = (1 - norm.cdf(0, mean_change, std_change)) * 100
    prob_decrease = norm.cdf(0, mean_change, std_change) * 100
    range_lower = current_close - 1.5 * std_change
    range_upper = current_close + 1.5 * std_change
    prob_range = (norm.cdf(range_upper - current_close, mean_change, std_change) - 
                  norm.cdf(range_lower - current_close, mean_change, std_change)) * 100
    
    return {
        "Increase": prob_increase,
        "Decrease": prob_decrease,
        "Range": prob_range,
        "Range_Lower": range_lower,
        "Range_Upper": range_upper
    }

# Main analysis
if crypto_pair not in ticker_map:
    print(f"Unsupported pair: {crypto_pair}. Available: {list(ticker_map.keys())}")
else:
    coin = ticker_map[crypto_pair]
    print(f"Cryptocurrency Analysis for {crypto_pair}\n")
    for tf_name, tf_seconds in timeframes.items():
        print(f"Timeframe: {tf_name}")
        data = fetch_data(coin, base_currency, tf_seconds, max(candle_counts))
        if data is None:
            print("  No data available.")
            continue
        current_close = data["Close"].iloc[-1] if not data.empty else None
        
        for count in candle_counts:
            print(f"\n  Candle Count: {count}")
            stats = calculate_stats(data, count)
            if stats:
                print("  Statistics:")
                for metric, values in stats.items():
                    print(f"    {metric}: Mean={values['Mean']:.2f}, Median={values['Median']:.2f}, Variance={values['Variance']:.2f}")
            else:
                print("    Insufficient data for stats.")
            
            probs = calculate_probabilities(data, count, current_close)
            if probs:
                print("  Probabilities:")
                print(f"    Increase: {probs['Increase']:.2f}%")
                print(f"    Decrease: {probs['Decrease']:.2f}%")
                print(f"    Range: {probs['Range']:.2f}%")
                print(f"    Range Thresholds: {probs['Range_Lower']:.2f} - {probs['Range_Upper']:.2f}")
            else:
                print("    Insufficient data for probabilities.")
        
        time.sleep(2)  # Avoid rate limits