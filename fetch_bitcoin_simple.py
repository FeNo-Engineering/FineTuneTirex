import requests
import pandas as pd
from datetime import datetime, timedelta
import time

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
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def fetch_and_save_data(symbol, interval, start_date, end_date, filename):
    """
    Fetch data for a specific period and save immediately
    """
    print(f"Fetching {symbol} data from {start_date} to {end_date}")
    
    all_data = []
    current_start = start_date
    
    while current_start < end_date:
        # For 5m intervals, 1000 records = ~3.5 days
        chunk_end = min(current_start + timedelta(days=3), end_date)
        
        print(f"  Chunk: {current_start} to {chunk_end}")
        
        data = get_binance_klines(symbol, interval, current_start, chunk_end)
        
        if data and len(data) > 0:
            all_data.extend(data)
            print(f"    Fetched {len(data)} records. Total: {len(all_data)}")
            
            # Update start time for next chunk
            last_timestamp = data[-1][0]
            current_start = datetime.fromtimestamp(last_timestamp / 1000) + timedelta(minutes=5)
        else:
            current_start = chunk_end
        
        time.sleep(0.1)  # Rate limiting
    
    if all_data:
        # Convert to DataFrame
        columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
        
        df = pd.DataFrame(all_data, columns=columns)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Convert numeric columns
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 
                          'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col])
        
        # Select final columns
        df = df[['datetime', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 
                 'number_of_trades', 'quote_asset_volume', 'taker_buy_base_asset_volume', 
                 'taker_buy_quote_asset_volume']]
        
        # Save to CSV
        df.to_csv(filename, index=False)
        print(f"Saved {len(df)} records to {filename}")
        
        return len(df)
    
    return 0

def main():
    symbol = "BTCUSDT"
    interval = "5m"
    
    # Fetch 2022-2024 data
    start_2022 = datetime(2022, 1, 1)
    end_2024 = datetime(2024, 12, 31, 23, 59, 59)
    
    records_2022_2024 = fetch_and_save_data(symbol, interval, start_2022, end_2024, 'bitcoin_2022_2024_5min.csv')
    
    print(f"\nCompleted 2022-2024: {records_2022_2024} records")
    
    # Fetch 2025 data
    start_2025 = datetime(2025, 1, 1)
    end_2025 = datetime(2025, 6, 23, 23, 59, 59)
    
    records_2025 = fetch_and_save_data(symbol, interval, start_2025, end_2025, 'bitcoin_2025_5min.csv')
    
    print(f"\nCompleted 2025: {records_2025} records")
    print("\nAll data fetched successfully!")

if __name__ == "__main__":
    main()