import numpy as np
import pandas as pd
from typing import Dict, Optional
from app.services.lstm_trainer import LSTMTrainer


class AISignalsService:
    """
    Service for generating AI-based trading signals.
    Includes price prediction capabilities.
    """

    def get_prediction_signal(self, price_data: pd.DataFrame) -> pd.Series:
        """
        Generate price direction predictions using a simplified LSTM-like approach.

        In production, this would:
        1. Train an LSTM model on historical price data
        2. Use features like past returns, volume, technical indicators
        3. Predict next-day price direction

        For now, uses a simplified technical approach that mimics ML predictions:
        - Combines momentum and mean reversion signals
        - Adds noise to simulate model uncertainty
        - Returns probability of upward movement (0 to 1)

        Args:
            price_data: DataFrame with price history

        Returns:
            Series of prediction scores (0-1) indexed by date
        """
        df = price_data.copy()

        # Calculate returns
        df['returns'] = df['close'].pct_change()

        # Calculate momentum (10-day)
        df['momentum'] = df['close'].pct_change(periods=10)

        # Calculate mean reversion signal (distance from 20-day MA)
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['mean_reversion'] = (df['close'] - df['ma_20']) / df['ma_20']

        # Calculate RSI-like indicator
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 1 - (1 / (1 + rs))

        # Combine signals (weighted)
        # Momentum: positive momentum -> buy signal
        momentum_signal = np.tanh(df['momentum'] * 10)  # Scale and bound

        # Mean reversion: oversold -> buy, overbought -> sell
        mean_reversion_signal = -np.tanh(df['mean_reversion'] * 5)

        # RSI signal
        rsi_signal = df['rsi']

        # Weighted combination
        prediction = (
            0.4 * momentum_signal +
            0.3 * mean_reversion_signal +
            0.3 * rsi_signal
        )

        # Add noise to simulate model uncertainty
        noise = np.random.normal(0, 0.1, len(prediction))
        prediction = prediction + noise

        # Convert to probability (0 to 1)
        prediction = (prediction + 1) / 2  # From [-1, 1] to [0, 1]
        prediction = np.clip(prediction, 0, 1)

        # Fill NaN values
        prediction = prediction.fillna(0.5)

        return pd.Series(prediction.values, index=df['date'])

    def get_lstm_prediction_signal(
        self,
        price_data: pd.DataFrame,
        model_path: str,
        scaler_params: Dict,
        lookback_period: int
    ) -> pd.Series:
        """
        Generate predictions using a trained LSTM model.

        Args:
            price_data: Historical price data
            model_path: Path to trained LSTM model
            scaler_params: Scaler parameters from training
            lookback_period: Lookback period used in training

        Returns:
            Series of prediction probabilities (0-1) indexed by date
        """
        trainer = LSTMTrainer()
        predictions = trainer.load_model_and_predict(
            model_path=model_path,
            price_data=price_data,
            scaler_params=scaler_params,
            lookback_period=lookback_period
        )
        return predictions

    def get_combined_signals(
        self,
        price_data: pd.DataFrame,
        use_prediction: bool = False,
        lstm_model: Optional[Dict] = None
    ) -> Dict[str, pd.Series]:
        """
        Get AI signals for a trading strategy.

        Args:
            price_data: Historical price data
            use_prediction: Whether to include price predictions
            lstm_model: Optional trained LSTM model info (path, scaler, config)

        Returns:
            Dictionary with 'prediction' Series if enabled
        """
        signals = {}

        if use_prediction and not price_data.empty:
            if lstm_model and lstm_model.get('model_path'):
                # Use trained LSTM model
                signals['prediction'] = self.get_lstm_prediction_signal(
                    price_data=price_data,
                    model_path=lstm_model['model_path'],
                    scaler_params=lstm_model['scaler_params'],
                    lookback_period=lstm_model['lookback_period']
                )
            else:
                # Fallback to simplified technical approach
                signals['prediction'] = self.get_prediction_signal(price_data)

        return signals
