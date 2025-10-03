import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from app.services.indicators import IndicatorService
from app.services.custom_strategy_executor import CustomStrategyExecutor


class BacktesterService:
    """
    Core backtesting engine for simulating trading strategies.
    """

    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.shares = 0
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[Dict[str, Any]] = []

    def run_backtest(
        self,
        price_data: pd.DataFrame,
        config: Dict[str, Any],
        ai_signals: Optional[Dict[str, pd.Series]] = None
    ) -> Dict[str, Any]:
        """
        Execute a backtest simulation.

        Args:
            price_data: DataFrame with columns [date, open, high, low, close, volume]
            config: Strategy configuration with parameters
            ai_signals: Optional dictionary with 'sentiment' and/or 'prediction' series

        Returns:
            Dictionary with performance metrics and trade log
        """
        # Reset state
        self.capital = self.initial_capital
        self.shares = 0
        self.trades = []
        self.equity_curve = []

        # Extract config
        mode = config.get('mode', 'simple')
        position_size = config.get('position_size', 1.0)
        use_prediction = config.get('use_prediction', False)

        # Prepare DataFrame
        df = price_data.copy()

        # Generate signals based on strategy mode
        if mode == 'simple':
            signals = self._generate_simple_signals(df, config, use_prediction, ai_signals)
        elif mode == 'indicators':
            signals = self._generate_indicator_signals(df, config)
        elif mode == 'custom':
            signals = self._generate_custom_signals(df, config)
        else:
            raise ValueError(f"Unknown strategy mode: {mode}")

        # Add signals to dataframe
        df['signal'] = signals

        # Simulate trading day by day
        for idx, row in df.iterrows():
            date = row['date']
            price = row['close']
            signal = row.get('signal', 'HOLD')

            # Record daily equity
            portfolio_value = self.capital + (self.shares * price)
            self.equity_curve.append({
                'date': date,
                'equity': float(portfolio_value)
            })

            # Execute trades based on signal
            if signal == 'BUY' and self.shares == 0:
                # Buy signal - enter position
                shares_to_buy = int((self.capital * position_size) / price)
                if shares_to_buy > 0:
                    cost = shares_to_buy * price
                    self.capital -= cost
                    self.shares += shares_to_buy

                    self.trades.append({
                        'date': date,
                        'action': 'BUY',
                        'price': float(price),
                        'shares': shares_to_buy,
                        'value': float(cost),
                        'balance': float(self.capital)
                    })

            elif signal == 'SELL' and self.shares > 0:
                # Sell signal - exit position
                proceeds = self.shares * price
                self.capital += proceeds

                self.trades.append({
                    'date': date,
                    'action': 'SELL',
                    'price': float(price),
                    'shares': self.shares,
                    'value': float(proceeds),
                    'balance': float(self.capital)
                })

                self.shares = 0

        # Close any open positions at the end
        if self.shares > 0:
            final_price = df.iloc[-1]['close']
            proceeds = self.shares * final_price
            self.capital += proceeds

            self.trades.append({
                'date': df.iloc[-1]['date'],
                'action': 'SELL',
                'price': float(final_price),
                'shares': self.shares,
                'value': float(proceeds),
                'balance': float(self.capital)
            })

            self.shares = 0

        # Calculate performance metrics
        metrics = self._calculate_metrics()

        return {
            **metrics,
            'trade_log': self.trades,
            'equity_curve': self.equity_curve
        }

    def _generate_simple_signals(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any],
        use_prediction: bool,
        ai_signals: Optional[Dict[str, pd.Series]]
    ) -> pd.Series:
        """Generate signals for simple MA crossover strategy"""
        ma_short = config.get('ma_short', 50)
        ma_long = config.get('ma_long', 200)

        df['ma_short'] = df['close'].rolling(window=ma_short).mean()
        df['ma_long'] = df['close'].rolling(window=ma_long).mean()

        # Add AI prediction if available
        if ai_signals and 'prediction' in ai_signals:
            prediction_series = ai_signals['prediction']
            df['prediction'] = df['date'].map(prediction_series.to_dict())
            df['prediction'] = df['prediction'].fillna(0.5)

        # Generate signals
        signals = pd.Series('HOLD', index=df.index)

        for idx, row in df.iterrows():
            if pd.isna(row['ma_short']) or pd.isna(row['ma_long']):
                continue

            ma_signal = 'HOLD'
            if row['ma_short'] > row['ma_long']:
                ma_signal = 'BUY'
            elif row['ma_short'] < row['ma_long']:
                ma_signal = 'SELL'

            # Apply AI prediction if enabled
            if use_prediction and 'prediction' in row and not pd.isna(row['prediction']):
                pred_signal = (row['prediction'] - 0.5) * 2
                if ma_signal == 'BUY' and pred_signal >= -0.2:
                    signals.iloc[idx] = 'BUY'
                elif ma_signal == 'SELL' and pred_signal <= 0.2:
                    signals.iloc[idx] = 'SELL'
            else:
                signals.iloc[idx] = ma_signal

        return signals

    def _generate_indicator_signals(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.Series:
        """Generate signals based on selected indicators and conditions"""
        indicators = config.get('indicators', [])

        # Calculate all indicators
        for indicator in indicators:
            df = IndicatorService.calculate_indicator(
                df,
                indicator.get('id'),
                indicator.get('params', {})
            )

        # Initialize signals
        signals = pd.Series('HOLD', index=df.index)

        # If no indicators selected, return HOLD
        if not indicators:
            return signals

        # Generate signals based on indicator types
        # Collect all indicator IDs
        indicator_ids = [ind.get('id') for ind in indicators]

        # Apply indicator-specific logic
        for idx in range(len(df)):
            buy_signals = 0
            sell_signals = 0
            total_indicators = 0

            # RSI Strategy
            if 'rsi' in indicator_ids and 'RSI' in df.columns:
                total_indicators += 1
                rsi_value = df.iloc[idx]['RSI']
                if pd.notna(rsi_value):
                    if rsi_value < 30:  # Oversold
                        buy_signals += 1
                    elif rsi_value > 70:  # Overbought
                        sell_signals += 1

            # MACD Strategy
            if 'macd' in indicator_ids and 'MACD' in df.columns and 'MACDs' in df.columns:
                total_indicators += 1
                if idx > 0:  # Need previous value for crossover
                    macd_curr = df.iloc[idx]['MACD']
                    signal_curr = df.iloc[idx]['MACDs']
                    macd_prev = df.iloc[idx - 1]['MACD']
                    signal_prev = df.iloc[idx - 1]['MACDs']

                    if pd.notna(macd_curr) and pd.notna(signal_curr):
                        # Bullish crossover
                        if macd_prev < signal_prev and macd_curr > signal_curr:
                            buy_signals += 1
                        # Bearish crossover
                        elif macd_prev > signal_prev and macd_curr < signal_curr:
                            sell_signals += 1

            # Bollinger Bands Strategy
            if 'bbands' in indicator_ids:
                # Find bollinger band columns (they vary by parameters)
                bb_cols = [col for col in df.columns if col.startswith('BBL_') or col.startswith('BBU_')]
                if len(bb_cols) >= 2:
                    total_indicators += 1
                    bbl_col = [col for col in bb_cols if col.startswith('BBL_')][0]
                    bbu_col = [col for col in bb_cols if col.startswith('BBU_')][0]

                    close = df.iloc[idx]['close']
                    lower = df.iloc[idx][bbl_col]
                    upper = df.iloc[idx][bbu_col]

                    if pd.notna(lower) and pd.notna(upper):
                        # Price touches lower band - oversold
                        if close <= lower * 1.01:  # Within 1% of lower band
                            buy_signals += 1
                        # Price touches upper band - overbought
                        elif close >= upper * 0.99:  # Within 1% of upper band
                            sell_signals += 1

            # Stochastic Strategy
            if 'stoch' in indicator_ids and 'STOCHk' in df.columns:
                total_indicators += 1
                stoch_k = df.iloc[idx]['STOCHk']
                if pd.notna(stoch_k):
                    if stoch_k < 20:  # Oversold
                        buy_signals += 1
                    elif stoch_k > 80:  # Overbought
                        sell_signals += 1

            # Moving Average Strategy (SMA/EMA)
            if 'sma' in indicator_ids or 'ema' in indicator_ids:
                # Find MA columns
                ma_cols = [col for col in df.columns if col.startswith('SMA_') or col.startswith('EMA_')]
                if len(ma_cols) >= 1:
                    total_indicators += 1
                    ma_col = ma_cols[0]  # Use first MA
                    close = df.iloc[idx]['close']
                    ma_value = df.iloc[idx][ma_col]

                    if pd.notna(ma_value) and idx > 0:
                        prev_close = df.iloc[idx - 1]['close']
                        prev_ma = df.iloc[idx - 1][ma_col]

                        # Price crosses above MA
                        if prev_close < prev_ma and close > ma_value:
                            buy_signals += 1
                        # Price crosses below MA
                        elif prev_close > prev_ma and close < ma_value:
                            sell_signals += 1

            # CCI Strategy
            if 'cci' in indicator_ids and 'CCI' in df.columns:
                total_indicators += 1
                cci_value = df.iloc[idx]['CCI']
                if pd.notna(cci_value):
                    if cci_value < -100:  # Oversold
                        buy_signals += 1
                    elif cci_value > 100:  # Overbought
                        sell_signals += 1

            # ADX for trend strength (filter)
            has_strong_trend = True
            if 'adx' in indicator_ids and 'ADX' in df.columns:
                adx_value = df.iloc[idx]['ADX']
                if pd.notna(adx_value):
                    # ADX > 25 indicates strong trend
                    has_strong_trend = adx_value > 25

            # Decision logic: majority vote with trend filter
            if total_indicators > 0 and has_strong_trend:
                # Need majority agreement for signal
                if buy_signals > sell_signals and buy_signals >= total_indicators * 0.5:
                    signals.iloc[idx] = 'BUY'
                elif sell_signals > buy_signals and sell_signals >= total_indicators * 0.5:
                    signals.iloc[idx] = 'SELL'

        return signals

    def _generate_custom_signals(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.Series:
        """Generate signals using custom Python code"""
        custom_code = config.get('custom_code')
        indicators = config.get('indicators', [])

        if not custom_code:
            raise ValueError("Custom code is required for custom strategy mode")

        # Calculate indicators
        for indicator in indicators:
            df = IndicatorService.calculate_indicator(
                df,
                indicator.get('id'),
                indicator.get('params', {})
            )

        # Extract indicator series for passing to custom code
        indicator_data = {}
        for col in df.columns:
            if col not in ['date', 'open', 'high', 'low', 'close', 'volume']:
                indicator_data[col] = df[col]

        # Execute custom strategy
        signals = CustomStrategyExecutor.execute_custom_strategy(
            custom_code,
            df,
            indicator_data
        )

        return signals

    def _generate_signal(self, row: pd.Series, use_prediction: bool) -> str:
        """
        DEPRECATED: Legacy method for simple MA crossover.
        Use _generate_simple_signals instead.
        """
        ma_signal = 'HOLD'

        if row['ma_short'] > row['ma_long']:
            ma_signal = 'BUY'
        elif row['ma_short'] < row['ma_long']:
            ma_signal = 'SELL'

        if not use_prediction:
            return ma_signal

        if use_prediction and 'prediction' in row:
            pred_signal = (row['prediction'] - 0.5) * 2
            if ma_signal == 'BUY' and pred_signal >= -0.2:
                return 'BUY'
            elif ma_signal == 'SELL' and pred_signal <= 0.2:
                return 'SELL'
            else:
                return 'HOLD'

        return ma_signal

    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics for the backtest"""

        final_balance = self.capital
        total_return = ((final_balance - self.initial_capital) / self.initial_capital) * 100

        # Calculate Sharpe ratio
        if len(self.equity_curve) > 1:
            equity_series = pd.Series([e['equity'] for e in self.equity_curve])
            returns = equity_series.pct_change().dropna()

            if len(returns) > 0 and returns.std() > 0:
                sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)  # Annualized
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0

        # Calculate maximum drawdown
        if len(self.equity_curve) > 0:
            equity_series = pd.Series([e['equity'] for e in self.equity_curve])
            rolling_max = equity_series.expanding().max()
            drawdowns = (equity_series - rolling_max) / rolling_max
            max_drawdown = abs(drawdowns.min()) * 100
        else:
            max_drawdown = 0.0

        # Analyze trades
        total_trades = len(self.trades)
        buy_trades = [t for t in self.trades if t['action'] == 'BUY']
        sell_trades = [t for t in self.trades if t['action'] == 'SELL']

        winning_trades = 0
        losing_trades = 0

        for i in range(min(len(buy_trades), len(sell_trades))):
            buy_price = buy_trades[i]['price']
            sell_price = sell_trades[i]['price']

            if sell_price > buy_price:
                winning_trades += 1
            else:
                losing_trades += 1

        return {
            'final_balance': float(final_balance),
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades
        }

    def calculate_benchmark(
        self,
        price_data: pd.DataFrame,
        initial_capital: float
    ) -> Dict[str, Any]:
        """
        Calculate buy-and-hold benchmark performance.

        Args:
            price_data: DataFrame with price history
            initial_capital: Starting capital amount

        Returns:
            Dictionary with benchmark equity curve and total return
        """
        if price_data.empty:
            return {
                'equity_curve': [],
                'total_return': 0.0
            }

        # Calculate number of shares bought at start
        initial_price = price_data.iloc[0]['close']
        shares = initial_capital / initial_price

        # Calculate portfolio value each day
        benchmark_equity = []
        for idx, row in price_data.iterrows():
            value = shares * row['close']
            benchmark_equity.append({
                'date': row['date'],
                'equity': float(value)
            })

        # Calculate total return
        final_value = benchmark_equity[-1]['equity']
        total_return = ((final_value - initial_capital) / initial_capital) * 100

        return {
            'equity_curve': benchmark_equity,
            'total_return': float(total_return)
        }
