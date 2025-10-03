import { useState, useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { strategyApi, backtestApi } from '../services/api';
import type { StrategyConfig, LSTMConfig, IndicatorConfig } from '../types';
import IndicatorSelector from '../components/IndicatorSelector';
import CodeEditor from '../components/CodeEditor';

export default function StrategyBuilder() {
  const navigate = useNavigate();
  const { strategyId } = useParams<{ strategyId: string }>();
  const [loading, setLoading] = useState(false);
  const [loadingStrategy, setLoadingStrategy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const isEditMode = !!strategyId;

  const [formData, setFormData] = useState({
    name: '',
    ticker: '',
    start_date: '2023-01-01',
    end_date: '2024-01-01',
  });

  const [strategyMode, setStrategyMode] = useState<'simple' | 'indicators' | 'custom'>('simple');

  const [config, setConfig] = useState<StrategyConfig>({
    mode: 'simple',
    ma_short: 50,
    ma_long: 200,
    indicators: [],
    custom_code: '',
    use_prediction: false,
    initial_capital: 100000,
    position_size: 1.0,
  });

  const [lstmConfig, setLstmConfig] = useState<LSTMConfig>({
    lookback_period: 60,
    epochs: 50,
    batch_size: 32,
    lstm_units: 50,
    dropout_rate: 0.2,
    train_test_split: 0.8,
  });

  // Load existing strategy if in edit mode
  useEffect(() => {
    const loadStrategy = async () => {
      if (!strategyId) return;

      try {
        setLoadingStrategy(true);
        const strategy = await strategyApi.get(parseInt(strategyId));

        setFormData({
          name: strategy.name,
          ticker: strategy.ticker,
          start_date: strategy.start_date,
          end_date: strategy.end_date,
        });

        const mode = strategy.config.mode || 'simple';
        setStrategyMode(mode);

        setConfig({
          mode,
          ma_short: strategy.config.ma_short || 50,
          ma_long: strategy.config.ma_long || 200,
          indicators: strategy.config.indicators || [],
          custom_code: strategy.config.custom_code || '',
          use_prediction: strategy.config.use_prediction || false,
          initial_capital: strategy.config.initial_capital || 100000,
          position_size: strategy.config.position_size || 1.0,
        });

        if (strategy.config.lstm_config) {
          setLstmConfig(strategy.config.lstm_config);
        }
      } catch (err: any) {
        setError(err.response?.data?.detail || 'Failed to load strategy');
      } finally {
        setLoadingStrategy(false);
      }
    };

    loadStrategy();
  }, [strategyId]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      // Create strategy config with mode-specific parameters
      const strategyConfig: StrategyConfig = {
        mode: strategyMode,
        ma_short: strategyMode === 'simple' ? config.ma_short : 50,
        ma_long: strategyMode === 'simple' ? config.ma_long : 200,
        indicators: strategyMode === 'indicators' || strategyMode === 'custom' ? config.indicators : undefined,
        custom_code: strategyMode === 'custom' ? config.custom_code : undefined,
        use_prediction: config.use_prediction,
        lstm_config: config.use_prediction ? lstmConfig : undefined,
        initial_capital: config.initial_capital,
        position_size: config.position_size,
      };

      let strategy;
      if (isEditMode) {
        strategy = await strategyApi.update(parseInt(strategyId!), {
          ...formData,
          config: strategyConfig,
        });
      } else {
        strategy = await strategyApi.create({
          ...formData,
          config: strategyConfig,
        });
      }

      // Run backtest
      await backtestApi.run(strategy.id);

      // Navigate to dashboard
      navigate('/');
    } catch (err: any) {
      setError(err.response?.data?.detail || `Failed to ${isEditMode ? 'update' : 'create'} strategy`);
    } finally {
      setLoading(false);
    }
  };

  if (loadingStrategy) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="text-gray-500">Loading strategy...</div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-8">
        {isEditMode ? 'Edit Trading Strategy' : 'Create Trading Strategy'}
      </h1>

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-6">
          {error}
        </div>
      )}

      <form onSubmit={handleSubmit} className="bg-white shadow-md rounded-lg p-6">
        {/* Basic Info */}
        <div className="mb-6">
          <h2 className="text-xl font-semibold mb-4">Basic Information</h2>
          <div className="grid grid-cols-1 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Strategy Name
              </label>
              <input
                type="text"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Stock Ticker
              </label>
              <input
                type="text"
                value={formData.ticker}
                onChange={(e) => setFormData({ ...formData, ticker: e.target.value.toUpperCase() })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="AAPL"
                required
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Start Date
                </label>
                <input
                  type="date"
                  value={formData.start_date}
                  onChange={(e) => setFormData({ ...formData, start_date: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  End Date
                </label>
                <input
                  type="date"
                  value={formData.end_date}
                  onChange={(e) => setFormData({ ...formData, end_date: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  required
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Initial Capital ($)
                </label>
                <input
                  type="number"
                  value={config.initial_capital}
                  onChange={(e) => setConfig({ ...config, initial_capital: parseFloat(e.target.value) })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  min="1000"
                  step="1000"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Position Size (% of Capital)
                </label>
                <input
                  type="number"
                  value={config.position_size * 100}
                  onChange={(e) => setConfig({ ...config, position_size: parseFloat(e.target.value) / 100 })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  min="10"
                  max="100"
                  step="5"
                  required
                />
              </div>
            </div>
          </div>
        </div>

        {/* Strategy Mode Tabs */}
        <div className="mb-6 border-t pt-6">
          <h2 className="text-xl font-semibold mb-4">Strategy Type</h2>

          <div className="flex space-x-1 bg-gray-100 p-1 rounded-lg mb-6">
            <button
              type="button"
              onClick={() => setStrategyMode('simple')}
              className={`flex-1 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                strategyMode === 'simple'
                  ? 'bg-white text-blue-600 shadow-sm'
                  : 'text-gray-700 hover:text-gray-900'
              }`}
            >
              Simple MA Crossover
            </button>
            <button
              type="button"
              onClick={() => setStrategyMode('indicators')}
              className={`flex-1 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                strategyMode === 'indicators'
                  ? 'bg-white text-blue-600 shadow-sm'
                  : 'text-gray-700 hover:text-gray-900'
              }`}
            >
              Visual Indicator Builder
            </button>
            <button
              type="button"
              onClick={() => setStrategyMode('custom')}
              className={`flex-1 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                strategyMode === 'custom'
                  ? 'bg-white text-blue-600 shadow-sm'
                  : 'text-gray-700 hover:text-gray-900'
              }`}
            >
              Custom Python Code
            </button>
          </div>

          {/* Simple Mode */}
          {strategyMode === 'simple' && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h3 className="text-sm font-medium text-gray-900 mb-3">Moving Average Crossover Strategy</h3>
              <p className="text-sm text-gray-700 mb-4">
                Classic strategy that generates buy signals when the short-term MA crosses above the long-term MA,
                and sell signals when it crosses below.
              </p>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Short MA Period
                  </label>
                  <input
                    type="number"
                    value={config.ma_short}
                    onChange={(e) => setConfig({ ...config, ma_short: parseInt(e.target.value) })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    min="1"
                    required
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Long MA Period
                  </label>
                  <input
                    type="number"
                    value={config.ma_long}
                    onChange={(e) => setConfig({ ...config, ma_long: parseInt(e.target.value) })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    min="1"
                    required
                  />
                </div>
              </div>
            </div>
          )}

          {/* Indicator Mode */}
          {strategyMode === 'indicators' && (
            <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
              <h3 className="text-sm font-medium text-gray-900 mb-3">Visual Indicator Builder</h3>
              <p className="text-sm text-gray-700 mb-4">
                Select technical indicators and configure their parameters. The system will automatically
                generate buy/sell signals based on each indicator's characteristics.
              </p>
              <IndicatorSelector
                selectedIndicators={config.indicators || []}
                onChange={(indicators) => setConfig({ ...config, indicators })}
              />
              <div className="mt-4 bg-blue-50 border border-blue-200 rounded p-3">
                <h4 className="text-xs font-semibold text-blue-900 mb-2">How It Works:</h4>
                <ul className="text-xs text-blue-800 space-y-1">
                  <li>• <strong>RSI:</strong> Buy when &lt;30 (oversold), Sell when &gt;70 (overbought)</li>
                  <li>• <strong>MACD:</strong> Buy on bullish crossover, Sell on bearish crossover</li>
                  <li>• <strong>Bollinger Bands:</strong> Buy at lower band, Sell at upper band</li>
                  <li>• <strong>Stochastic:</strong> Buy when &lt;20, Sell when &gt;80</li>
                  <li>• <strong>MA (SMA/EMA):</strong> Buy on upward cross, Sell on downward cross</li>
                  <li>• <strong>CCI:</strong> Buy when &lt;-100, Sell when &gt;100</li>
                  <li>• <strong>ADX:</strong> Used as trend filter (signals only in strong trends)</li>
                </ul>
                <p className="text-xs text-blue-800 mt-2">
                  Signals are generated using a <strong>majority vote</strong> system - multiple indicators
                  must agree before a buy/sell signal is triggered.
                </p>
              </div>
            </div>
          )}

          {/* Custom Code Mode */}
          {strategyMode === 'custom' && (
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <h3 className="text-sm font-medium text-gray-900 mb-3">Custom Python Strategy</h3>
              <p className="text-sm text-gray-700 mb-4">
                Write custom Python code to implement your own trading logic. You have access to price data,
                indicators, pandas, and numpy.
              </p>

              {/* Indicator selection for custom code */}
              <div className="mb-4">
                <h4 className="text-sm font-medium text-gray-700 mb-2">Select Indicators to Calculate</h4>
                <IndicatorSelector
                  selectedIndicators={config.indicators || []}
                  onChange={(indicators) => setConfig({ ...config, indicators })}
                />
              </div>

              {/* Code editor */}
              <CodeEditor
                code={config.custom_code || ''}
                onChange={(code) => setConfig({ ...config, custom_code: code })}
              />
            </div>
          )}
        </div>

        {/* AI Signals */}
        <div className="mb-6 border-t pt-6">
          <h2 className="text-xl font-semibold mb-4">AI Signals (Optional)</h2>

          <div className="space-y-4">
            <label className="flex items-center space-x-3">
              <input
                type="checkbox"
                checked={config.use_prediction}
                onChange={(e) => setConfig({ ...config, use_prediction: e.target.checked })}
                className="w-5 h-5 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
              />
              <span className="text-sm font-medium text-gray-700">
                Use LSTM Price Prediction (AI-Powered)
              </span>
            </label>

            {config.use_prediction && (
              <div className="ml-8 p-4 bg-blue-50 rounded-lg border border-blue-200 space-y-4">
                <p className="text-sm text-blue-800 mb-3">
                  Configure LSTM model parameters. The model will be trained on historical data before backtesting.
                </p>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Lookback Period (days)
                    </label>
                    <input
                      type="number"
                      value={lstmConfig.lookback_period}
                      onChange={(e) => setLstmConfig({ ...lstmConfig, lookback_period: parseInt(e.target.value) })}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                      min="10"
                      max="365"
                      required
                    />
                    <p className="text-xs text-gray-500 mt-1">Number of past days used for prediction (10-365)</p>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Training Epochs
                    </label>
                    <input
                      type="number"
                      value={lstmConfig.epochs}
                      onChange={(e) => setLstmConfig({ ...lstmConfig, epochs: parseInt(e.target.value) })}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                      min="10"
                      max="200"
                      required
                    />
                    <p className="text-xs text-gray-500 mt-1">Number of training iterations (10-200)</p>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Batch Size
                    </label>
                    <input
                      type="number"
                      value={lstmConfig.batch_size}
                      onChange={(e) => setLstmConfig({ ...lstmConfig, batch_size: parseInt(e.target.value) })}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                      min="8"
                      max="128"
                      step="8"
                      required
                    />
                    <p className="text-xs text-gray-500 mt-1">Training batch size (8-128)</p>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      LSTM Units
                    </label>
                    <input
                      type="number"
                      value={lstmConfig.lstm_units}
                      onChange={(e) => setLstmConfig({ ...lstmConfig, lstm_units: parseInt(e.target.value) })}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                      min="20"
                      max="200"
                      step="10"
                      required
                    />
                    <p className="text-xs text-gray-500 mt-1">Number of LSTM neurons (20-200)</p>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Dropout Rate
                    </label>
                    <input
                      type="number"
                      value={lstmConfig.dropout_rate}
                      onChange={(e) => setLstmConfig({ ...lstmConfig, dropout_rate: parseFloat(e.target.value) })}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                      min="0"
                      max="0.5"
                      step="0.05"
                      required
                    />
                    <p className="text-xs text-gray-500 mt-1">Regularization dropout (0-0.5)</p>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Train/Test Split
                    </label>
                    <input
                      type="number"
                      value={lstmConfig.train_test_split * 100}
                      onChange={(e) => setLstmConfig({ ...lstmConfig, train_test_split: parseFloat(e.target.value) / 100 })}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                      min="50"
                      max="95"
                      step="5"
                      required
                    />
                    <p className="text-xs text-gray-500 mt-1">% of data for training (50-95%)</p>
                  </div>
                </div>

                <div className="bg-yellow-50 border border-yellow-200 rounded p-3 mt-3">
                  <p className="text-xs text-yellow-800">
                    ⚠️ Training an LSTM model may take several minutes depending on the data size and parameters.
                    You'll be able to monitor the training progress on the dashboard.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Submit Button */}
        <div className="flex justify-end space-x-4 border-t pt-6">
          <button
            type="button"
            onClick={() => navigate('/')}
            className="px-6 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={loading}
            className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              config.use_prediction ?
                (isEditMode ? 'Updating & Training...' : 'Creating & Training...') :
                (isEditMode ? 'Updating...' : 'Creating...')
            ) : (
              config.use_prediction ?
                (isEditMode ? 'Update, Train & Backtest' : 'Create, Train & Backtest') :
                (isEditMode ? 'Update & Run Backtest' : 'Create & Run Backtest')
            )}
          </button>
        </div>
      </form>
    </div>
  );
}
