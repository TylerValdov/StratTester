# LSTM Price Prediction Model Training

This folder contains the code and instructions for training a real LSTM model for stock price direction prediction.

## 📁 Folder Structure

```
ml_models/
├── scripts/
│   ├── train_lstm.py           # Main training script
│   ├── prepare_data.py         # Data preparation utilities
│   └── evaluate_model.py       # Model evaluation script
├── lstm/
│   └── (trained models saved here as .keras files)
├── data/
│   └── (downloaded training data cached here)
├── requirements.txt            # ML-specific dependencies
└── README.md                   # This file
```

## 🎯 What This Model Does

The LSTM (Long Short-Term Memory) model predicts the probability that a stock's price will go UP the next day.

**Input Features:**
- Past 60 days of price data (open, high, low, close, volume)
- Technical indicators (RSI, MACD, Bollinger Bands)
- Normalized returns

**Output:**
- Probability between 0 and 1
  - 0.0 = Very likely to go DOWN
  - 0.5 = Uncertain
  - 1.0 = Very likely to go UP

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd ml_models
pip install -r requirements.txt
```

### 2. Prepare Training Data

```bash
# Download historical data for training
python scripts/prepare_data.py --ticker SPY --years 5
```

This downloads 5 years of S&P 500 data to train on.

### 3. Train the Model

```bash
# Train LSTM model
python scripts/train_lstm.py --ticker SPY --epochs 50 --batch-size 32
```

This will:
- Load and preprocess the data
- Train an LSTM neural network
- Save the model to `lstm/lstm_model.keras`
- Display training metrics

### 4. Evaluate the Model

```bash
# Test model performance
python scripts/evaluate_model.py --model lstm/lstm_model.keras --ticker AAPL
```

### 5. Integrate with Backend

The trained model will automatically be loaded by the backend service:

```python
# backend/app/services/ai_signals.py will detect and use it
if os.path.exists('ml_models/lstm/lstm_model.keras'):
    model = load_model('ml_models/lstm/lstm_model.keras')
    predictions = model.predict(features)
```

## 📊 Training Process Explained

### Step 1: Data Collection
```python
# Downloads historical price data using Alpaca API or yfinance
data = fetch_historical_data('SPY', start='2019-01-01', end='2024-01-01')
```

### Step 2: Feature Engineering
```python
# Creates features from raw price data
features = {
    'returns': price.pct_change(),
    'volume_change': volume.pct_change(),
    'rsi': calculate_rsi(price, 14),
    'macd': calculate_macd(price),
    'bb_position': (price - bb_lower) / (bb_upper - bb_lower)
}
```

### Step 3: Sequence Creation
```python
# Creates 60-day sequences for LSTM input
# Example: Days 1-60 → Predict Day 61
X = []
y = []
for i in range(60, len(data)):
    X.append(data[i-60:i])  # Past 60 days
    y.append(1 if data[i]['close'] > data[i-1]['close'] else 0)  # Up or down
```

### Step 4: Model Architecture
```python
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(60, n_features)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1, activation='sigmoid')  # Probability output
])
```

### Step 5: Training
```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train,
          epochs=50,
          batch_size=32,
          validation_split=0.2)
```

### Step 6: Evaluation
```python
# Test on unseen data
accuracy = model.evaluate(X_test, y_test)
predictions = model.predict(X_test)
```

## 📈 Expected Performance

**Typical Metrics:**
- **Training Accuracy**: 55-65%
- **Validation Accuracy**: 52-58%
- **Test Accuracy**: 50-56%

**Note:** Even 52-55% accuracy is valuable! Stock prediction is extremely difficult. Anything above 50% (random) with consistent edge can be profitable.

## 🎓 Training Tips

### 1. Start with a Broad Index
Train on SPY (S&P 500) first - it has more stable patterns than individual stocks.

### 2. Use Sufficient Data
- Minimum: 2 years
- Recommended: 5+ years
- More data = better generalization

### 3. Experiment with Hyperparameters
```bash
# Try different configurations
python scripts/train_lstm.py --lstm-units 100 --dropout 0.3 --learning-rate 0.0001
```

### 4. Use Early Stopping
The script includes early stopping to prevent overfitting:
```python
EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
```

### 5. Train Multiple Models
Train different models for different market conditions:
- `lstm_bull_market.keras` - Trained on 2019-2021
- `lstm_bear_market.keras` - Trained on 2022
- `lstm_general.keras` - Trained on all available data

## 🔧 Advanced Configuration

### Custom Features
Edit `scripts/prepare_data.py` to add your own features:
```python
def add_custom_features(df):
    df['feature_1'] = ...
    df['feature_2'] = ...
    return df
```

### Different Architectures
Try different LSTM configurations in `scripts/train_lstm.py`:
```python
# Deeper network
LSTM(100) → LSTM(50) → LSTM(25) → Dense(1)

# Bidirectional LSTM
Bidirectional(LSTM(50))

# With attention mechanism
LSTM(50) + Attention layer
```

### Ensemble Models
Train multiple models and average predictions:
```python
pred_1 = model_1.predict(X)
pred_2 = model_2.predict(X)
pred_3 = model_3.predict(X)
final_pred = (pred_1 + pred_2 + pred_3) / 3
```

## 🚨 Common Issues

### Issue: "Not enough memory"
```bash
# Reduce batch size
python scripts/train_lstm.py --batch-size 16
```

### Issue: "Model overfitting"
- Increase dropout (0.3-0.5)
- Add more training data
- Reduce model complexity
- Use regularization

### Issue: "Training too slow"
```bash
# Use GPU if available (automatic with TensorFlow)
# Or reduce sequence length
python scripts/train_lstm.py --sequence-length 30
```

### Issue: "Poor accuracy"
- Try different features
- Normalize data properly
- Check for data leakage
- Train on more data

## 📚 Resources

**LSTM Fundamentals:**
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [TensorFlow LSTM Guide](https://www.tensorflow.org/guide/keras/rnn)

**Stock Prediction:**
- [Financial Time Series Prediction](https://arxiv.org/abs/1809.10341)
- [Deep Learning for Trading](https://github.com/firmai/financial-machine-learning)

**Technical Analysis:**
- [TA-Lib Documentation](https://ta-lib.org/)
- [Pandas-TA Library](https://github.com/twopirllc/pandas-ta)

## 🎯 Next Steps After Training

1. **Evaluate on Multiple Stocks**: Test your model on various tickers
2. **Backtest Integration**: The model will be automatically used in backtests
3. **Monitor Performance**: Track prediction accuracy over time
4. **Retrain Regularly**: Update the model monthly with new data
5. **A/B Test**: Compare strategies with and without LSTM predictions

## 📝 Model Files

After training, you'll have:
```
ml_models/lstm/
├── lstm_model.keras           # Main trained model
├── scaler.pkl                 # Feature scaler (for normalization)
├── training_history.json      # Loss and accuracy over epochs
└── model_config.json          # Hyperparameters used
```

## 🔐 Production Deployment

For production use:
1. Version your models (lstm_v1.keras, lstm_v2.keras)
2. Track training data date ranges
3. Monitor prediction drift
4. Implement model rollback capability
5. A/B test before full deployment

---

**Ready to train your first model? Start with:**
```bash
cd ml_models
pip install -r requirements.txt
python scripts/train_lstm.py --ticker SPY --epochs 50
```

Good luck! 🚀
