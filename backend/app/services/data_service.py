import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from app.core.config import settings


class DataService:
    """Service for fetching market data using Alpaca API"""

    def __init__(self):
        if settings.ALPACA_API_KEY and settings.ALPACA_API_SECRET:
            self.client = StockHistoricalDataClient(
                api_key=settings.ALPACA_API_KEY,
                secret_key=settings.ALPACA_API_SECRET
            )
            print(f"✓ Alpaca API client initialized with key: {settings.ALPACA_API_KEY[:8]}...")
        else:
            self.client = None
            print("✗ No Alpaca API keys found - will use simulated data")

    def fetch_historical_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch historical price data for a given ticker and date range.

        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """

        # Validate dates are not in the future
        today = datetime.now().date()
        end = datetime.fromisoformat(end_date).date()
        if end > today:
            print(f"Warning: End date {end_date} is in the future. Using today's date instead.")
            end_date = today.strftime('%Y-%m-%d')

        if not self.client:
            # Fallback to simulated data if no API keys configured
            print(f"Warning: No Alpaca API client configured. Using simulated data for {ticker}")
            return self._generate_simulated_data(ticker, start_date, end_date)

        try:
            # Parse dates - localize to America/New_York timezone like test.py
            start = pd.to_datetime(start_date).tz_localize('America/New_York')

            # Create request params - match test.py format
            # Use 'all' adjustment to account for both splits and dividends
            request_params = StockBarsRequest(
                symbol_or_symbols=[ticker],  # Use list format like test.py
                timeframe=TimeFrame.Day,
                start=start,
                adjustment='all'  # Adjust for splits and dividends
            )

            # Fetch data
            print(f"Fetching data for {ticker} from {start_date}...")
            bars_df = self.client.get_stock_bars(request_params).df.tz_convert('America/New_York', level=1)

            if bars_df.empty:
                raise ValueError(f"No data returned for {ticker}")

            print(f"✓ Successfully fetched {len(bars_df)} days of data for {ticker}")

            # Reset index to get date as column
            bars_df = bars_df.reset_index()

            # Rename columns to match expected format
            bars_df = bars_df.rename(columns={
                'timestamp': 'date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })

            # Select relevant columns
            bars_df = bars_df[['date', 'open', 'high', 'low', 'close', 'volume']]

            # Convert date to string format
            bars_df['date'] = bars_df['date'].dt.strftime('%Y-%m-%d')

            # Filter to end_date if specified
            end = datetime.fromisoformat(end_date)
            bars_df = bars_df[bars_df['date'] <= end_date]

            return bars_df

        except Exception as e:
            print(f"Error fetching data from Alpaca for {ticker}: {type(e).__name__}: {e}")
            print(f"Falling back to simulated data for {ticker}...")
            import traceback
            traceback.print_exc()
            return self._generate_simulated_data(ticker, start_date, end_date)

    def _generate_simulated_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Generate simulated price data for testing purposes.
        Uses a random walk with drift to create realistic-looking price movements.

        WARNING: This is FAKE data and should only be used for testing!
        """
        print(f"⚠️  WARNING: Generating SIMULATED data for {ticker} - NOT REAL MARKET DATA!")

        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)

        # Prevent future dates
        today = datetime.now().date()
        if end.date() > today:
            end = datetime.combine(today, datetime.min.time())

        # Generate date range (business days only)
        dates = pd.bdate_range(start=start, end=end)
        n_days = len(dates)

        # Set random seed based on ticker for reproducibility
        np.random.seed(sum(ord(c) for c in ticker))

        # Generate price data using geometric Brownian motion
        initial_price = 100.0
        drift = 0.0005  # Small positive drift
        volatility = 0.02  # Daily volatility

        returns = np.random.normal(drift, volatility, n_days)
        price_multipliers = np.exp(returns)
        close_prices = initial_price * np.cumprod(price_multipliers)

        # Generate OHLC data
        df = pd.DataFrame({
            'date': dates.strftime('%Y-%m-%d'),
            'close': close_prices
        })

        # Generate realistic open, high, low based on close
        df['open'] = df['close'].shift(1).fillna(initial_price)

        # High and low with some randomness
        daily_range = np.abs(np.random.normal(0, volatility * 0.5, n_days))
        df['high'] = df[['open', 'close']].max(axis=1) * (1 + daily_range)
        df['low'] = df[['open', 'close']].min(axis=1) * (1 - daily_range)

        # Generate volume
        df['volume'] = np.random.randint(1000000, 10000000, n_days)

        # Reorder columns
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]

        return df

    def get_latest_price(self, ticker: str) -> Optional[float]:
        """Get the latest price for a ticker"""
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

            df = self.fetch_historical_data(ticker, start_date, end_date)

            if not df.empty:
                return float(df.iloc[-1]['close'])

            return None
        except Exception as e:
            print(f"Error fetching latest price: {e}")
            return None
