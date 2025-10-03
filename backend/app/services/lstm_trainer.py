import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import os
import json
from datetime import datetime

try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import MinMaxScaler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not installed. LSTM training will not be available.")


class LSTMTrainer:
    """
    Service for training LSTM models for stock price prediction.
    """

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        if not TENSORFLOW_AVAILABLE:
            raise ImportError(
                "TensorFlow is required for LSTM training. "
                "Install it with: pip install tensorflow"
            )

    def prepare_data(
        self,
        price_data: pd.DataFrame,
        lookback_period: int = 60,
        train_test_split: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
        """
        Prepare price data for LSTM training.

        Args:
            price_data: DataFrame with price history (must have 'close' column)
            lookback_period: Number of previous days to use for prediction
            train_test_split: Fraction of data to use for training

        Returns:
            Tuple of (X_train, y_train, X_test, y_test, scaler)
        """
        # Extract close prices
        prices = price_data['close'].values.reshape(-1, 1)

        # Normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_prices = scaler.fit_transform(prices)

        # Create sequences
        X, y = [], []
        for i in range(lookback_period, len(scaled_prices)):
            X.append(scaled_prices[i - lookback_period:i, 0])
            y.append(scaled_prices[i, 0])

        X, y = np.array(X), np.array(y)

        # Reshape X for LSTM [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Split into train and test
        split_idx = int(len(X) * train_test_split)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        return X_train, y_train, X_test, y_test, scaler

    def build_model(
        self,
        lookback_period: int,
        lstm_units: int = 50,
        dropout_rate: float = 0.2
    ) -> keras.Model:
        """
        Build LSTM model architecture.

        Args:
            lookback_period: Number of time steps in input
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate for regularization

        Returns:
            Compiled Keras model
        """
        model = Sequential([
            LSTM(lstm_units, return_sequences=True, input_shape=(lookback_period, 1)),
            Dropout(dropout_rate),
            LSTM(lstm_units, return_sequences=False),
            Dropout(dropout_rate),
            Dense(25, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model

    def train_model(
        self,
        price_data: pd.DataFrame,
        lstm_config: Dict,
        model_id: int,
        progress_callback: Optional[callable] = None
    ) -> Dict:
        """
        Train LSTM model on price data.

        Args:
            price_data: Historical price data
            lstm_config: LSTM configuration parameters
            model_id: Unique identifier for saving the model
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with training results and metadata
        """
        # Extract config
        lookback_period = lstm_config.get('lookback_period', 60)
        epochs = lstm_config.get('epochs', 50)
        batch_size = lstm_config.get('batch_size', 32)
        lstm_units = lstm_config.get('lstm_units', 50)
        dropout_rate = lstm_config.get('dropout_rate', 0.2)
        train_test_split = lstm_config.get('train_test_split', 0.8)

        # Prepare data
        if progress_callback:
            progress_callback(10, "Preparing training data...")

        X_train, y_train, X_test, y_test, scaler = self.prepare_data(
            price_data,
            lookback_period=lookback_period,
            train_test_split=train_test_split
        )

        # Build model
        if progress_callback:
            progress_callback(20, "Building LSTM architecture...")

        model = self.build_model(
            lookback_period=lookback_period,
            lstm_units=lstm_units,
            dropout_rate=dropout_rate
        )

        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )

        # Custom callback for progress updates
        class ProgressCallback(keras.callbacks.Callback):
            def __init__(self, callback_fn, total_epochs):
                super().__init__()
                self.callback_fn = callback_fn
                self.total_epochs = total_epochs

            def on_epoch_end(self, epoch, logs=None):
                if self.callback_fn:
                    progress = 20 + int((epoch + 1) / self.total_epochs * 60)
                    self.callback_fn(progress, f"Training epoch {epoch + 1}/{self.total_epochs}...")

        callbacks = [early_stopping, reduce_lr]
        if progress_callback:
            callbacks.append(ProgressCallback(progress_callback, epochs))

        # Train model
        if progress_callback:
            progress_callback(20, f"Training LSTM model for {epochs} epochs...")

        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )

        # Evaluate model
        if progress_callback:
            progress_callback(85, "Evaluating model performance...")

        train_loss = float(history.history['loss'][-1])
        val_loss = float(history.history['val_loss'][-1])

        # Save model
        if progress_callback:
            progress_callback(90, "Saving trained model...")

        model_path = os.path.join(self.model_dir, f"lstm_model_{model_id}.keras")
        model.save(model_path)

        # Save scaler parameters
        scaler_params = {
            'min': scaler.data_min_.tolist(),
            'max': scaler.data_max_.tolist(),
            'scale': scaler.scale_.tolist(),
            'data_range': scaler.data_range_.tolist()
        }

        # Convert history to JSON-serializable format
        training_history = {
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'mae': [float(x) for x in history.history.get('mae', [])],
            'val_mae': [float(x) for x in history.history.get('val_mae', [])]
        }

        return {
            'model_path': model_path,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'scaler_params': scaler_params,
            'training_history': training_history,
            'trained_at': datetime.utcnow().isoformat()
        }

    def load_model_and_predict(
        self,
        model_path: str,
        price_data: pd.DataFrame,
        scaler_params: Dict,
        lookback_period: int
    ) -> pd.Series:
        """
        Load a trained model and generate predictions.

        Args:
            model_path: Path to saved model
            price_data: Price data for prediction
            scaler_params: Saved scaler parameters
            lookback_period: Number of days to look back

        Returns:
            Series of predictions indexed by date
        """
        # Load model
        model = keras.models.load_model(model_path)

        # Reconstruct scaler
        scaler = MinMaxScaler()
        scaler.fit(np.array([[scaler_params['min'][0]], [scaler_params['max'][0]]]))

        # Prepare data
        prices = price_data['close'].values.reshape(-1, 1)
        scaled_prices = scaler.transform(prices)

        # Generate predictions
        predictions = []
        dates = []

        for i in range(lookback_period, len(scaled_prices)):
            X = scaled_prices[i - lookback_period:i, 0]
            X = np.reshape(X, (1, lookback_period, 1))

            pred = model.predict(X, verbose=0)
            pred_price = scaler.inverse_transform(pred)[0, 0]

            predictions.append(pred_price)
            dates.append(price_data.iloc[i]['date'])

        # Convert predictions to probability scores (0-1)
        # Compare predicted price to current price
        actual_prices = price_data.iloc[lookback_period:]['close'].values
        prediction_array = np.array(predictions)

        # Calculate percentage change
        pct_change = (prediction_array - actual_prices) / actual_prices

        # Convert to probability (sigmoid-like transformation)
        # Positive change -> higher probability, negative -> lower
        probabilities = 1 / (1 + np.exp(-pct_change * 10))  # Scale factor of 10
        probabilities = np.clip(probabilities, 0.1, 0.9)  # Clip to reasonable range

        return pd.Series(probabilities, index=dates)
