"""
Model evaluation script - test LSTM performance on new data.
"""

import argparse
import numpy as np
import pandas as pd
import os
import sys
import json
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_model_artifacts(ticker: str):
    """Load trained model and associated artifacts."""
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lstm')

    # Load model
    model_path = os.path.join(model_dir, f'lstm_model_{ticker}.keras')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded model from {model_path}")

    # Load scaler
    scaler_path = os.path.join(model_dir, f'scaler_{ticker}.pkl')
    scaler = joblib.load(scaler_path)
    print(f"Loaded scaler from {scaler_path}")

    # Load features
    features_path = os.path.join(model_dir, f'features_{ticker}.json')
    with open(features_path, 'r') as f:
        features = json.load(f)
    print(f"Loaded {len(features)} features")

    # Load config
    config_path = os.path.join(model_dir, f'model_config_{ticker}.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    return model, scaler, features, config


def prepare_test_data(ticker: str, features: list, scaler, sequence_length: int):
    """Prepare test data from CSV."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    filepath = os.path.join(data_dir, f'{ticker}_prepared.csv')

    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    print(f"\nLoaded test data: {df.shape}")

    # Get last 20% as test set (same as training split)
    test_size = int(len(df) * 0.2)
    df_test = df.iloc[-test_size:]

    print(f"Test period: {df_test.index.min()} to {df_test.index.max()}")
    print(f"Test samples: {len(df_test)}")

    # Scale features
    X = scaler.transform(df_test[features])
    y = df_test['target'].values

    # Create sequences
    X_seq = []
    y_seq = []
    dates = []

    for i in range(sequence_length, len(X)):
        X_seq.append(X[i - sequence_length:i])
        y_seq.append(y[i])
        dates.append(df_test.index[i])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    return X_seq, y_seq, dates, df_test


def evaluate_predictions(y_true, y_pred, y_pred_proba):
    """Evaluate model predictions."""
    print("\n" + "="*70)
    print("Classification Report")
    print("="*70)
    print(classification_report(y_true, y_pred, target_names=['Down', 'Up']))

    print("\n" + "="*70)
    print("Confusion Matrix")
    print("="*70)
    cm = confusion_matrix(y_true, y_pred)
    print(f"              Predicted")
    print(f"              Down    Up")
    print(f"Actual Down   {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"       Up     {cm[1,0]:4d}  {cm[1,1]:4d}")

    # Calculate metrics
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
    recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # ROC AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    print(f"\n" + "="*70)
    print("Summary Metrics")
    print("="*70)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist()
    }


def plot_results(y_true, y_pred_proba, dates, ticker):
    """Create visualization plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'LSTM Model Evaluation - {ticker}', fontsize=16, fontweight='bold')

    # 1. Prediction probabilities over time
    ax = axes[0, 0]
    ax.plot(dates, y_pred_proba, label='Predicted Probability (Up)', alpha=0.7)
    ax.axhline(y=0.5, color='r', linestyle='--', label='Decision Threshold')
    ax.fill_between(dates, 0, 1, where=(y_true == 1), alpha=0.2, color='green', label='Actual Up Days')
    ax.set_xlabel('Date')
    ax.set_ylabel('Probability')
    ax.set_title('Prediction Probabilities Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Prediction distribution
    ax = axes[0, 1]
    ax.hist(y_pred_proba[y_true == 0], bins=30, alpha=0.5, label='Actual Down', color='red')
    ax.hist(y_pred_proba[y_true == 1], bins=30, alpha=0.5, label='Actual Up', color='green')
    ax.axvline(x=0.5, color='black', linestyle='--', label='Threshold')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Predictions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Confusion matrix heatmap
    ax = axes[1, 0]
    cm = confusion_matrix(y_true, (y_pred_proba > 0.5).astype(int))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title('Confusion Matrix')

    # 4. ROC Curve
    ax = axes[1, 1]
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lstm')
    plot_path = os.path.join(output_dir, f'evaluation_{ticker}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained LSTM model')
    parser.add_argument('--ticker', type=str, default='SPY',
                       help='Stock ticker symbol (default: SPY)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting visualizations')

    args = parser.parse_args()

    print("="*70)
    print(f"LSTM Model Evaluation - {args.ticker}")
    print("="*70)

    try:
        # Load model artifacts
        print("\n1. Loading model artifacts...")
        model, scaler, features, config = load_model_artifacts(args.ticker)

        print(f"\nModel trained on: {config['training_date']}")
        print(f"Training accuracy: {config['final_val_accuracy']:.4f}")

        # Prepare test data
        print("\n2. Preparing test data...")
        X_test, y_test, dates, df_test = prepare_test_data(
            args.ticker, features, scaler, config['sequence_length']
        )

        # Make predictions
        print("\n3. Making predictions...")
        y_pred_proba = model.predict(X_test, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Evaluate
        print("\n4. Evaluating performance...")
        metrics = evaluate_predictions(y_test, y_pred, y_pred_proba)

        # Save evaluation results
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lstm')
        eval_path = os.path.join(output_dir, f'evaluation_results_{args.ticker}.json')
        eval_results = {
            'ticker': args.ticker,
            'evaluation_date': datetime.now().isoformat(),
            'test_period': {
                'start': str(dates[0]),
                'end': str(dates[-1])
            },
            'metrics': metrics,
            'model_config': config
        }
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"\nResults saved to: {eval_path}")

        # Plot results
        if not args.no_plot:
            print("\n5. Creating visualizations...")
            plot_results(y_test, y_pred_proba, dates, args.ticker)

        print("\n" + "="*70)
        print("Evaluation Complete!")
        print("="*70)

        # Interpretation
        print("\n📊 Interpretation:")
        if metrics['accuracy'] > 0.55:
            print("✅ Model shows promising predictive power (>55% accuracy)")
        elif metrics['accuracy'] > 0.52:
            print("⚠️  Model shows slight edge over random (52-55% accuracy)")
        elif metrics['accuracy'] > 0.50:
            print("⚠️  Model barely beats random guessing (50-52% accuracy)")
        else:
            print("❌ Model underperforms random guessing (<50% accuracy)")

        print("\n💡 Tips:")
        print("  - Accuracy >52% can be profitable with proper position sizing")
        print("  - Focus on ROC AUC score - higher is better")
        print("  - Check if model works better in certain market conditions")
        print("  - Consider retraining with more data or different features")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
