"""
LSTM model training script for stock price prediction.
"""

import argparse
import numpy as np
import pandas as pd
import os
import sys
import json
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

# TensorFlow imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class LSTMTrainer:
    """LSTM model trainer for stock price prediction."""

    def __init__(
        self,
        sequence_length: int = 60,
        lstm_units: int = 50,
        dropout: float = 0.2,
        learning_rate: float = 0.001
    ):
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = None
        self.feature_columns = None

    def load_data(self, ticker: str) -> pd.DataFrame:
        """Load prepared data from disk."""
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        filepath = os.path.join(data_dir, f'{ticker}_prepared.csv')

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Data file not found: {filepath}\n"
                f"Run: python scripts/prepare_data.py --ticker {ticker}"
            )

        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        print(f"Loaded data: {df.shape}")
        return df

    def prepare_features(self, df: pd.DataFrame):
        """Select and scale features."""
        # Select feature columns (exclude target and raw OHLCV)
        exclude_cols = ['target', 'open', 'high', 'low', 'close', 'volume']
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]

        print(f"\nUsing {len(self.feature_columns)} features:")
        for col in self.feature_columns[:10]:
            print(f"  - {col}")
        if len(self.feature_columns) > 10:
            print(f"  ... and {len(self.feature_columns) - 10} more")

        # Scale features to [0, 1]
        self.scaler = MinMaxScaler()
        scaled_features = self.scaler.fit_transform(df[self.feature_columns])

        return scaled_features, df['target'].values

    def create_sequences(self, X: np.ndarray, y: np.ndarray):
        """Create sequences for LSTM input."""
        X_seq = []
        y_seq = []

        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i - self.sequence_length:i])
            y_seq.append(y[i])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        print(f"\nCreated sequences:")
        print(f"  X shape: {X_seq.shape}")
        print(f"  y shape: {y_seq.shape}")

        return X_seq, y_seq

    def build_model(self, n_features: int):
        """Build LSTM model architecture."""
        print("\nBuilding model...")

        model = Sequential([
            # First LSTM layer with return sequences
            LSTM(
                self.lstm_units,
                return_sequences=True,
                input_shape=(self.sequence_length, n_features)
            ),
            Dropout(self.dropout),

            # Second LSTM layer
            LSTM(self.lstm_units, return_sequences=False),
            Dropout(self.dropout),

            # Dense layers
            Dense(25, activation='relu'),
            Dropout(self.dropout / 2),

            # Output layer (binary classification)
            Dense(1, activation='sigmoid')
        ])

        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )

        print("\nModel architecture:")
        model.summary()

        self.model = model
        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32
    ):
        """Train the LSTM model."""
        print("\nTraining model...")

        # Callbacks
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lstm')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'best_model_temp.keras')

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]

        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return history

    def save_model(self, ticker: str, history):
        """Save trained model and metadata."""
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lstm')
        os.makedirs(model_dir, exist_ok=True)

        # Save model
        model_path = os.path.join(model_dir, f'lstm_model_{ticker}.keras')
        self.model.save(model_path)
        print(f"\nModel saved to: {model_path}")

        # Save scaler
        scaler_path = os.path.join(model_dir, f'scaler_{ticker}.pkl')
        joblib.dump(self.scaler, scaler_path)
        print(f"Scaler saved to: {scaler_path}")

        # Save feature columns
        features_path = os.path.join(model_dir, f'features_{ticker}.json')
        with open(features_path, 'w') as f:
            json.dump(self.feature_columns, f, indent=2)
        print(f"Features saved to: {features_path}")

        # Save training history
        history_path = os.path.join(model_dir, f'training_history_{ticker}.json')
        history_dict = {
            'loss': [float(x) for x in history.history['loss']],
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        }
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)

        # Save config
        config = {
            'ticker': ticker,
            'sequence_length': self.sequence_length,
            'lstm_units': self.lstm_units,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'n_features': len(self.feature_columns),
            'training_date': datetime.now().isoformat(),
            'final_val_accuracy': float(history.history['val_accuracy'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1])
        }
        config_path = os.path.join(model_dir, f'model_config_{ticker}.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\nAll files saved successfully!")


def main():
    parser = argparse.ArgumentParser(description='Train LSTM model for stock prediction')
    parser.add_argument('--ticker', type=str, default='SPY',
                       help='Stock ticker symbol (default: SPY)')
    parser.add_argument('--sequence-length', type=int, default=60,
                       help='Sequence length for LSTM (default: 60)')
    parser.add_argument('--lstm-units', type=int, default=50,
                       help='Number of LSTM units (default: 50)')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate (default: 0.2)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--test-split', type=float, default=0.2,
                       help='Test set size (default: 0.2)')

    args = parser.parse_args()

    print("="*70)
    print("LSTM Stock Price Prediction - Training")
    print("="*70)
    print(f"\nTicker: {args.ticker}")
    print(f"Sequence Length: {args.sequence_length}")
    print(f"LSTM Units: {args.lstm_units}")
    print(f"Dropout: {args.dropout}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print()

    try:
        # Initialize trainer
        trainer = LSTMTrainer(
            sequence_length=args.sequence_length,
            lstm_units=args.lstm_units,
            dropout=args.dropout,
            learning_rate=args.learning_rate
        )

        # Load data
        print("Step 1: Loading data...")
        df = trainer.load_data(args.ticker)

        # Prepare features
        print("\nStep 2: Preparing features...")
        X, y = trainer.prepare_features(df)

        # Create sequences
        print("\nStep 3: Creating sequences...")
        X_seq, y_seq = trainer.create_sequences(X, y)

        # Split data
        print("\nStep 4: Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq,
            test_size=args.test_split,
            shuffle=False  # Important for time series!
        )

        # Further split train into train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=0.2,
            shuffle=False
        )

        print(f"  Train: {X_train.shape[0]} samples")
        print(f"  Validation: {X_val.shape[0]} samples")
        print(f"  Test: {X_test.shape[0]} samples")

        # Build model
        print("\nStep 5: Building model...")
        n_features = X_seq.shape[2]
        trainer.build_model(n_features)

        # Train model
        print("\nStep 6: Training model...")
        history = trainer.train(
            X_train, y_train,
            X_val, y_val,
            epochs=args.epochs,
            batch_size=args.batch_size
        )

        # Evaluate on test set
        print("\nStep 7: Evaluating on test set...")
        test_loss, test_acc, test_auc = trainer.model.evaluate(X_test, y_test, verbose=0)
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Test AUC: {test_auc:.4f}")

        # Save model
        print("\nStep 8: Saving model...")
        trainer.save_model(args.ticker, history)

        print("\n" + "="*70)
        print("Training Complete!")
        print("="*70)
        print(f"\nFinal Results:")
        print(f"  Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"\nModel ready for use in backtesting!")
        print(f"\nTo evaluate: python scripts/evaluate_model.py --ticker {args.ticker}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
