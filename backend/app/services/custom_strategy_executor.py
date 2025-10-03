import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from RestrictedPython import compile_restricted, safe_globals
from RestrictedPython.Guards import guarded_iter_unpack_sequence, safer_getattr
from app.services.indicators import IndicatorService


class CustomStrategyExecutor:
    """
    Execute user-defined Python strategies in a sandboxed environment.
    Uses RestrictedPython to prevent malicious code execution.
    """

    @staticmethod
    def get_safe_globals():
        """
        Get safe globals for user code execution.
        Includes pandas and numpy but restricts file I/O and imports.
        """
        safe_dict = {
            '__builtins__': safe_globals,
            '_getiter_': guarded_iter_unpack_sequence,
            '_getattr_': safer_getattr,
            'pd': pd,
            'np': np,
            'indicators': IndicatorService,
            # Math functions
            'abs': abs,
            'max': max,
            'min': min,
            'sum': sum,
            'len': len,
            'range': range,
            # Type conversions
            'int': int,
            'float': float,
            'str': str,
            'bool': bool,
            'list': list,
            'dict': dict,
        }
        return safe_dict

    @staticmethod
    def validate_strategy_code(code: str) -> tuple[bool, Optional[str]]:
        """
        Validate user strategy code before execution.

        Args:
            code: Python code string

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for restricted keywords
        forbidden_keywords = [
            'import os', 'import sys', 'import subprocess',
            'open(', '__import__', 'eval(', 'exec(',
            'compile(', 'globals(', 'locals(',
            'file', 'input(', 'raw_input('
        ]

        for keyword in forbidden_keywords:
            if keyword in code:
                return False, f"Forbidden keyword detected: {keyword}"

        # Try to compile with RestrictedPython
        try:
            byte_code = compile_restricted(code, '<string>', 'exec')
            if byte_code.errors:
                return False, f"Compilation errors: {', '.join(byte_code.errors)}"
            return True, None
        except Exception as e:
            return False, f"Invalid Python syntax: {str(e)}"

    @staticmethod
    def create_strategy_template() -> str:
        """
        Get a template for custom strategy code.

        Returns:
            Python code template string
        """
        return """def generate_signal(df, indicators):
    \"\"\"
    Generate trading signals based on price data and indicators.

    Args:
        df: DataFrame with columns ['date', 'open', 'high', 'low', 'close', 'volume']
        indicators: Dict with calculated indicator DataFrames

    Returns:
        Series with values: 'BUY', 'SELL', or 'HOLD' for each date
    \"\"\"
    signals = pd.Series('HOLD', index=df.index)

    # Example: RSI strategy
    # if 'RSI' in indicators:
    #     rsi = indicators['RSI']
    #     signals[rsi < 30] = 'BUY'   # Oversold
    #     signals[rsi > 70] = 'SELL'  # Overbought

    # Example: MACD crossover
    # if 'MACD_12_26_9' in indicators and 'MACDs_12_26_9' in indicators:
    #     macd = indicators['MACD_12_26_9']
    #     signal = indicators['MACDs_12_26_9']
    #     signals[(macd > signal) & (macd.shift(1) <= signal.shift(1))] = 'BUY'
    #     signals[(macd < signal) & (macd.shift(1) >= signal.shift(1))] = 'SELL'

    # Add your strategy logic here

    return signals
"""

    @staticmethod
    def execute_custom_strategy(
        code: str,
        price_data: pd.DataFrame,
        indicator_data: Dict[str, pd.Series]
    ) -> pd.Series:
        """
        Execute user-defined strategy code.

        Args:
            code: Python code string with generate_signal function
            price_data: DataFrame with OHLCV data
            indicator_data: Dict of calculated indicators

        Returns:
            Series with 'BUY', 'SELL', or 'HOLD' signals

        Raises:
            ValueError: If code is invalid or execution fails
        """
        # Validate code
        is_valid, error_msg = CustomStrategyExecutor.validate_strategy_code(code)
        if not is_valid:
            raise ValueError(f"Invalid strategy code: {error_msg}")

        # Compile code
        byte_code = compile_restricted(code, '<string>', 'exec')
        if byte_code.errors:
            raise ValueError(f"Compilation errors: {', '.join(byte_code.errors)}")

        # Execute code in sandboxed environment
        safe_dict = CustomStrategyExecutor.get_safe_globals()
        exec(byte_code, safe_dict)

        # Check if generate_signal function exists
        if 'generate_signal' not in safe_dict:
            raise ValueError("Strategy code must define a 'generate_signal' function")

        generate_signal = safe_dict['generate_signal']

        # Call the function
        try:
            signals = generate_signal(price_data, indicator_data)

            # Validate output
            if not isinstance(signals, pd.Series):
                raise ValueError("generate_signal must return a pandas Series")

            # Ensure all values are valid signals
            valid_signals = {'BUY', 'SELL', 'HOLD'}
            invalid = set(signals.unique()) - valid_signals
            if invalid:
                raise ValueError(f"Invalid signal values: {invalid}. Must be one of: {valid_signals}")

            return signals

        except Exception as e:
            raise ValueError(f"Error executing strategy: {str(e)}")

    @staticmethod
    def get_strategy_examples() -> Dict[str, str]:
        """
        Get example strategy implementations.

        Returns:
            Dict of example name to code
        """
        return {
            "RSI Oversold/Overbought": """def generate_signal(df, indicators):
    signals = pd.Series('HOLD', index=df.index)

    if 'RSI' in indicators:
        rsi = indicators['RSI']
        signals[rsi < 30] = 'BUY'   # Oversold - buy signal
        signals[rsi > 70] = 'SELL'  # Overbought - sell signal

    return signals
""",
            "MACD Crossover": """def generate_signal(df, indicators):
    signals = pd.Series('HOLD', index=df.index)

    if 'MACD_12_26_9' in indicators and 'MACDs_12_26_9' in indicators:
        macd = indicators['MACD_12_26_9']
        signal_line = indicators['MACDs_12_26_9']

        # Bullish crossover
        bullish = (macd > signal_line) & (macd.shift(1) <= signal_line.shift(1))
        signals[bullish] = 'BUY'

        # Bearish crossover
        bearish = (macd < signal_line) & (macd.shift(1) >= signal_line.shift(1))
        signals[bearish] = 'SELL'

    return signals
""",
            "Bollinger Bands Breakout": """def generate_signal(df, indicators):
    signals = pd.Series('HOLD', index=df.index)

    if 'BBL_20_2.0' in indicators and 'BBU_20_2.0' in indicators:
        lower = indicators['BBL_20_2.0']
        upper = indicators['BBU_20_2.0']
        close = df['close']

        # Buy when price crosses below lower band
        signals[close < lower] = 'BUY'

        # Sell when price crosses above upper band
        signals[close > upper] = 'SELL'

    return signals
""",
            "Multi-Indicator Confirmation": """def generate_signal(df, indicators):
    signals = pd.Series('HOLD', index=df.index)

    # Require both RSI and MACD to agree
    if 'RSI' in indicators and 'MACD_12_26_9' in indicators and 'MACDs_12_26_9' in indicators:
        rsi = indicators['RSI']
        macd = indicators['MACD_12_26_9']
        signal_line = indicators['MACDs_12_26_9']

        # Buy when RSI oversold AND MACD bullish crossover
        rsi_buy = rsi < 40
        macd_buy = (macd > signal_line) & (macd.shift(1) <= signal_line.shift(1))
        signals[rsi_buy & macd_buy] = 'BUY'

        # Sell when RSI overbought AND MACD bearish crossover
        rsi_sell = rsi > 60
        macd_sell = (macd < signal_line) & (macd.shift(1) >= signal_line.shift(1))
        signals[rsi_sell & macd_sell] = 'SELL'

    return signals
"""
        }
