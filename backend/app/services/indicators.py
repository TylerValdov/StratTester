import pandas as pd
import numpy as np
from typing import Dict, Any, List


class IndicatorService:
    """
    Service for calculating technical indicators using pandas-ta.
    Supports both pre-built indicators and custom combinations.
    """

    @staticmethod
    def get_available_indicators() -> List[Dict[str, Any]]:
        """
        Get list of available indicators with their parameters.

        Returns:
            List of indicator definitions
        """
        return [
            {
                "id": "sma",
                "name": "Simple Moving Average (SMA)",
                "category": "trend",
                "params": [
                    {"name": "length", "type": "int", "default": 20, "min": 1, "max": 500}
                ]
            },
            {
                "id": "ema",
                "name": "Exponential Moving Average (EMA)",
                "category": "trend",
                "params": [
                    {"name": "length", "type": "int", "default": 20, "min": 1, "max": 500}
                ]
            },
            {
                "id": "rsi",
                "name": "Relative Strength Index (RSI)",
                "category": "momentum",
                "params": [
                    {"name": "length", "type": "int", "default": 14, "min": 2, "max": 100}
                ]
            },
            {
                "id": "macd",
                "name": "MACD",
                "category": "momentum",
                "params": [
                    {"name": "fast", "type": "int", "default": 12, "min": 2, "max": 100},
                    {"name": "slow", "type": "int", "default": 26, "min": 2, "max": 200},
                    {"name": "signal", "type": "int", "default": 9, "min": 2, "max": 50}
                ]
            },
            {
                "id": "bbands",
                "name": "Bollinger Bands",
                "category": "volatility",
                "params": [
                    {"name": "length", "type": "int", "default": 20, "min": 2, "max": 100},
                    {"name": "std", "type": "float", "default": 2.0, "min": 0.5, "max": 5.0}
                ]
            },
            {
                "id": "stoch",
                "name": "Stochastic Oscillator",
                "category": "momentum",
                "params": [
                    {"name": "k", "type": "int", "default": 14, "min": 2, "max": 100},
                    {"name": "d", "type": "int", "default": 3, "min": 2, "max": 50},
                    {"name": "smooth_k", "type": "int", "default": 3, "min": 1, "max": 20}
                ]
            },
            {
                "id": "atr",
                "name": "Average True Range (ATR)",
                "category": "volatility",
                "params": [
                    {"name": "length", "type": "int", "default": 14, "min": 2, "max": 100}
                ]
            },
            {
                "id": "adx",
                "name": "Average Directional Index (ADX)",
                "category": "trend",
                "params": [
                    {"name": "length", "type": "int", "default": 14, "min": 2, "max": 100}
                ]
            },
            {
                "id": "cci",
                "name": "Commodity Channel Index (CCI)",
                "category": "momentum",
                "params": [
                    {"name": "length", "type": "int", "default": 20, "min": 2, "max": 100}
                ]
            },
            {
                "id": "obv",
                "name": "On-Balance Volume (OBV)",
                "category": "volume",
                "params": []
            }
        ]

    @staticmethod
    def calculate_indicator(
        price_data: pd.DataFrame,
        indicator_id: str,
        params: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Calculate a specific indicator.

        Args:
            price_data: DataFrame with OHLCV data
            indicator_id: Indicator ID (e.g., 'rsi', 'macd')
            params: Indicator parameters

        Returns:
            DataFrame with indicator columns added
        """
        df = price_data.copy()

        # Ensure we have the right column names for pandas-ta
        if 'open' in df.columns:
            df['Open'] = df['open']
        if 'high' in df.columns:
            df['High'] = df['high']
        if 'low' in df.columns:
            df['Low'] = df['low']
        if 'close' in df.columns:
            df['Close'] = df['close']
        if 'volume' in df.columns:
            df['Volume'] = df['volume']

        if indicator_id == "sma":
            length = params.get('length', 20)
            df[f'SMA_{length}'] = df['Close'].rolling(window=length).mean()

        elif indicator_id == "ema":
            length = params.get('length', 20)
            df[f'EMA_{length}'] = df['Close'].ewm(span=length, adjust=False).mean()

        elif indicator_id == "rsi":
            length = params.get('length', 14)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

        elif indicator_id == "macd":
            fast = params.get('fast', 12)
            slow = params.get('slow', 26)
            signal = params.get('signal', 9)
            ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
            ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
            df['MACD'] = ema_fast - ema_slow
            df['MACDs'] = df['MACD'].ewm(span=signal, adjust=False).mean()
            df['MACDh'] = df['MACD'] - df['MACDs']

        elif indicator_id == "bbands":
            length = params.get('length', 20)
            std = params.get('std', 2.0)
            sma = df['Close'].rolling(window=length).mean()
            rolling_std = df['Close'].rolling(window=length).std()
            df[f'BBL_{length}_{std}'] = sma - (rolling_std * std)
            df[f'BBM_{length}_{std}'] = sma
            df[f'BBU_{length}_{std}'] = sma + (rolling_std * std)

        elif indicator_id == "stoch":
            k = params.get('k', 14)
            d = params.get('d', 3)
            smooth_k = params.get('smooth_k', 3)
            low_min = df['Low'].rolling(window=k).min()
            high_max = df['High'].rolling(window=k).max()
            stoch_k = 100 * (df['Close'] - low_min) / (high_max - low_min)
            df['STOCHk'] = stoch_k.rolling(window=smooth_k).mean()
            df['STOCHd'] = df['STOCHk'].rolling(window=d).mean()

        elif indicator_id == "atr":
            length = params.get('length', 14)
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['ATR'] = tr.rolling(window=length).mean()

        elif indicator_id == "adx":
            length = params.get('length', 14)
            high_diff = df['High'].diff()
            low_diff = -df['Low'].diff()
            pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
            neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

            atr = tr.rolling(window=length).mean()
            pos_di = 100 * (pos_dm.rolling(window=length).mean() / atr)
            neg_di = 100 * (neg_dm.rolling(window=length).mean() / atr)

            dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di)
            df['ADX'] = dx.rolling(window=length).mean()
            df['DMP'] = pos_di
            df['DMN'] = neg_di

        elif indicator_id == "cci":
            length = params.get('length', 20)
            tp = (df['High'] + df['Low'] + df['Close']) / 3
            sma_tp = tp.rolling(window=length).mean()
            mad = tp.rolling(window=length).apply(lambda x: np.abs(x - x.mean()).mean())
            df['CCI'] = (tp - sma_tp) / (0.015 * mad)

        elif indicator_id == "obv":
            obv = [0]
            for i in range(1, len(df)):
                if df['Close'].iloc[i] > df['Close'].iloc[i - 1]:
                    obv.append(obv[-1] + df['Volume'].iloc[i])
                elif df['Close'].iloc[i] < df['Close'].iloc[i - 1]:
                    obv.append(obv[-1] - df['Volume'].iloc[i])
                else:
                    obv.append(obv[-1])
            df['OBV'] = obv

        return df

    @staticmethod
    def calculate_all_indicators(
        price_data: pd.DataFrame,
        indicators: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Calculate multiple indicators on price data.

        Args:
            price_data: DataFrame with OHLCV data
            indicators: List of indicator configs, each with 'id' and 'params'

        Returns:
            DataFrame with all indicator columns
        """
        df = price_data.copy()

        for indicator in indicators:
            indicator_id = indicator.get('id')
            params = indicator.get('params', {})
            df = IndicatorService.calculate_indicator(df, indicator_id, params)

        return df
