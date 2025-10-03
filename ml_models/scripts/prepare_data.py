"""
Data preparation script for LSTM training.
Downloads historical data and engineers features.
"""

import argparse
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def download_data(ticker: str, years: int = 5) -> pd.DataFrame:
    """
    Download historical price data.

    Args:
        ticker: Stock symbol
        years: Number of years of historical data

    Returns:
        DataFrame with OHLCV data
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)

    print(f"Downloading {years} years of data for {ticker}...")
    data = yf.download(ticker, start=start_date, end=end_date, progress=True)

    if data.empty:
        raise ValueError(f"No data downloaded for {ticker}")

    # Flatten multi-index columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Rename columns to lowercase
    data.columns = [col.lower() for col in data.columns]

    print(f"Downloaded {len(data)} days of data")
    return data


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators as features.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with added technical indicators
    """
    print("Calculating technical indicators...")

    # Returns
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # Volume features
    df['volume_change'] = df['volume'].pct_change()
    df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

    # Moving averages
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()

    # Price relative to moving averages
    df['price_to_sma_5'] = df['close'] / df['sma_5']
    df['price_to_sma_20'] = df['close'] / df['sma_20']
    df['price_to_sma_50'] = df['close'] / df['sma_50']

    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD (Moving Average Convergence Divergence)
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

    # Momentum indicators
    df['momentum_5'] = df['close'].pct_change(5)
    df['momentum_10'] = df['close'].pct_change(10)
    df['momentum_20'] = df['close'].pct_change(20)

    # Volatility (standard deviation of returns)
    df['volatility_10'] = df['returns'].rolling(10).std()
    df['volatility_20'] = df['returns'].rolling(20).std()

    # Average True Range (ATR)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(14).mean()
    df['atr_ratio'] = df['atr'] / df['close']

    # Price position in daily range
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)

    print(f"Added {len([c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume']])} features")

    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create target variable: 1 if next day closes higher, 0 otherwise.

    Args:
        df: DataFrame with price data

    Returns:
        DataFrame with target variable
    """
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    return df


def prepare_dataset(ticker: str, years: int = 5, save: bool = True) -> pd.DataFrame:
    """
    Complete data preparation pipeline.

    Args:
        ticker: Stock symbol
        years: Years of historical data
        save: Whether to save to disk

    Returns:
        Prepared DataFrame
    """
    # Download data
    df = download_data(ticker, years)

    # Add technical indicators
    df = calculate_technical_indicators(df)

    # Create target
    df = create_target(df)

    # Drop rows with NaN (from indicator calculations)
    initial_rows = len(df)
    df = df.dropna()
    print(f"Dropped {initial_rows - len(df)} rows with NaN values")
    print(f"Final dataset: {len(df)} rows")

    # Save to disk
    if save:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        os.makedirs(data_dir, exist_ok=True)

        filepath = os.path.join(data_dir, f'{ticker}_prepared.csv')
        df.to_csv(filepath)
        print(f"Saved prepared data to {filepath}")

    return df


def main():
    parser = argparse.ArgumentParser(description='Prepare data for LSTM training')
    parser.add_argument('--ticker', type=str, default='SPY',
                       help='Stock ticker symbol (default: SPY)')
    parser.add_argument('--years', type=int, default=5,
                       help='Years of historical data (default: 5)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save to disk')

    args = parser.parse_args()

    try:
        df = prepare_dataset(args.ticker, args.years, save=not args.no_save)

        print("\n" + "="*50)
        print("Data Preparation Complete!")
        print("="*50)
        print(f"\nDataset shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"\nTarget distribution:")
        print(df['target'].value_counts(normalize=True))
        print(f"\nFeature columns: {len(df.columns)}")
        print("\nReady for training! Run:")
        print(f"  python scripts/train_lstm.py --ticker {args.ticker}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
