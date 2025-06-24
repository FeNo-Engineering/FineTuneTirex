import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os

# Binance API credentials
BINANCE_API_KEY = "ndGxgEs4dN7NlTVOdUwt19CUnN2hECm0CAzfodKewWWZ73WUxCgTFc6Cwn5Yiqhg"
BINANCE_API_SECRET = "HdlPNtJOFOLAZmAy8XsVg5NOiU57QCHjql5pxTLBBFfYgkQsyImEc0aFlBJkd9aw"

def get_binance_klines(symbol, interval, start_time, end_time, limit=1000):
    """
    Fetch historical klines data from Binance API
    """
    url = "https://api.binance.com/api/v3/klines"
    
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": int(start_time.timestamp() * 1000),
        "endTime": int(end_time.timestamp() * 1000),
        "limit": limit
    }
    
    headers = {
        "X-MBX-APIKEY": BINANCE_API_KEY
    }
    
    response = requests.get(url, params=params, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def fetch_historical_data(symbol, interval, start_date, end_date):
    """
    Fetch historical data in chunks and combine them
    """
    all_data = []
    current_start = start_date
    
    while current_start < end_date:
        # Calculate end time for this chunk (max 1000 intervals)
        # For 5m intervals, 1000 records = ~3.5 days
        chunk_end = min(current_start + timedelta(minutes=5000), end_date)
        
        print(f"Fetching data from {current_start} to {chunk_end}")
        
        data = get_binance_klines(symbol, interval, current_start, chunk_end)
        
        if data:
            all_data.extend(data)
            print(f"Fetched {len(data)} records. Total so far: {len(all_data)}")
            
            # Update start time for next chunk
            if len(data) > 0:
                last_timestamp = data[-1][0]  # Get timestamp of last record
                current_start = datetime.fromtimestamp(last_timestamp / 1000) + timedelta(minutes=5)
            else:
                current_start = chunk_end
                
            # Save progress periodically
            if len(all_data) % 10000 == 0:
                print(f"Progress: {len(all_data)} records fetched")
        else:
            print("Failed to fetch data, stopping...")
            break
        
        # Rate limiting - wait a bit between requests
        time.sleep(0.2)
    
    return all_data

def convert_to_dataframe(data):
    """
    Convert raw klines data to pandas DataFrame
    """
    columns = [
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ]
    
    df = pd.DataFrame(data, columns=columns)
    
    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Convert numeric columns
    numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 
                      'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
    
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col])
    
    # Select and reorder columns
    df = df[['datetime', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 
             'number_of_trades', 'quote_asset_volume', 'taker_buy_base_asset_volume', 
             'taker_buy_quote_asset_volume']]
    
    return df

def main():
    symbol = "BTCUSDT"
    interval = "5m"
    
    # Fetch data for 2022-2024
    print("Fetching Bitcoin data for 2022-2024...")
    start_date_2022 = datetime(2022, 1, 1)
    end_date_2024 = datetime(2024, 12, 31, 23, 59, 59)
    
    data_2022_2024 = fetch_historical_data(symbol, interval, start_date_2022, end_date_2024)
    
    if data_2022_2024:
        df_2022_2024 = convert_to_dataframe(data_2022_2024)
        df_2022_2024.to_csv('bitcoin_2022_2024_5min.csv', index=False)
        print(f"Saved 2022-2024 data: {len(df_2022_2024)} records to bitcoin_2022_2024_5min.csv")
    
    # Fetch data for 2025
    print("Fetching Bitcoin data for 2025...")
    start_date_2025 = datetime(2025, 1, 1)
    end_date_2025 = datetime(2025, 6, 23, 23, 59, 59)
    
    data_2025 = fetch_historical_data(symbol, interval, start_date_2025, end_date_2025)
    
    if data_2025:
        df_2025 = convert_to_dataframe(data_2025)
        df_2025.to_csv('bitcoin_2025_5min.csv', index=False)
        print(f"Saved 2025 data: {len(df_2025)} records to bitcoin_2025_5min.csv")

if __name__ == "__main__":
    main()