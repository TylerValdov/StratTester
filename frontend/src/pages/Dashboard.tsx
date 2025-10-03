import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { strategyApi, backtestApi } from '../services/api';
import type { Strategy, BacktestResult } from '../types';

interface StrategyWithBacktests extends Strategy {
  backtests: BacktestResult[];
}

export default function Dashboard() {
  const [strategies, setStrategies] = useState<StrategyWithBacktests[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [deletingStrategyId, setDeletingStrategyId] = useState<number | null>(null);

  const loadData = async () => {
    try {
      setLoading(true);
      const [strategiesData, backtestsData] = await Promise.all([
        strategyApi.list(),
        backtestApi.list(),
      ]);

      // Combine strategies with their backtests
      const combined = strategiesData.map((strategy) => ({
        ...strategy,
        backtests: backtestsData.filter((bt) => bt.strategy_id === strategy.id),
      }));

      // Sort strategies by creation date (most recent first)
      combined.sort((a, b) =>
        new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
      );

      setStrategies(combined);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();

    // Poll for updates every 5 seconds
    const interval = setInterval(loadData, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleDeleteStrategy = async (strategyId: number, strategyName: string) => {
    if (!window.confirm(`Are you sure you want to delete the entire strategy "${strategyName}" and all its backtests?`)) {
      return;
    }

    try {
      setDeletingStrategyId(strategyId);
      await strategyApi.delete(strategyId);
      // Reload data after deletion
      await loadData();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to delete strategy');
    } finally {
      setDeletingStrategyId(null);
    }
  };

  const getLatestBacktest = (backtests: BacktestResult[]) => {
    return backtests.sort((a, b) =>
      new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
    )[0];
  };

  const getStatusBadge = (status: string) => {
    const colors = {
      PENDING: 'bg-yellow-100 text-yellow-800',
      PROGRESS: 'bg-blue-100 text-blue-800',
      TRAINING: 'bg-purple-100 text-purple-800',
      SUCCESS: 'bg-green-100 text-green-800',
      FAILURE: 'bg-red-100 text-red-800',
    };

    const labels = {
      PENDING: 'Pending',
      PROGRESS: 'Running',
      TRAINING: 'Training LSTM',
      SUCCESS: 'Complete',
      FAILURE: 'Failed',
    };

    return (
      <span className={`px-2 py-1 text-xs font-semibold rounded ${colors[status as keyof typeof colors] || 'bg-gray-100 text-gray-800'}`}>
        {labels[status as keyof typeof labels] || status}
      </span>
    );
  };

  const getStrategyDescription = (strategy: Strategy) => {
    const mode = strategy.config.mode || 'simple';

    switch (mode) {
      case 'simple':
        return {
          label: 'MA Crossover',
          value: `${strategy.config.ma_short}/${strategy.config.ma_long}`,
          color: 'text-blue-600',
        };

      case 'indicators':
        const indicatorCount = strategy.config.indicators?.length || 0;
        const indicatorNames = strategy.config.indicators
          ?.map(ind => ind.id.toUpperCase())
          .slice(0, 3)
          .join(', ') || 'None';
        return {
          label: 'Visual Indicators',
          value: indicatorCount > 0
            ? `${indicatorCount} indicator${indicatorCount > 1 ? 's' : ''} (${indicatorNames}${indicatorCount > 3 ? '...' : ''})`
            : 'No indicators',
          color: 'text-purple-600',
        };

      case 'custom':
        const customIndicatorCount = strategy.config.indicators?.length || 0;
        return {
          label: 'Custom Python',
          value: customIndicatorCount > 0
            ? `With ${customIndicatorCount} indicator${customIndicatorCount > 1 ? 's' : ''}`
            : 'Custom logic',
          color: 'text-green-600',
        };

      default:
        return {
          label: 'Strategy',
          value: 'Unknown',
          color: 'text-gray-600',
        };
    }
  };

  if (loading && strategies.length === 0) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="text-gray-500">Loading...</div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-3xl font-bold">Trading Strategies Dashboard</h1>
        <Link
          to="/strategies/new"
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
        >
          + New Strategy
        </Link>
      </div>

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-6">
          {error}
        </div>
      )}

      {strategies.length === 0 ? (
        <div className="bg-white rounded-lg shadow-md p-12 text-center">
          <p className="text-gray-500 mb-4">No strategies created yet</p>
          <Link
            to="/strategies/new"
            className="text-blue-600 hover:text-blue-800 font-medium"
          >
            Create your first strategy →
          </Link>
        </div>
      ) : (
        <div className="grid gap-6">
          {strategies.map((strategy) => {
            const latestBacktest = getLatestBacktest(strategy.backtests);

            return (
              <div
                key={strategy.id}
                className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow"
              >
                <div className="flex justify-between items-start mb-4">
                  <div>
                    <h3 className="text-xl font-semibold mb-2">{strategy.name}</h3>
                    <p className="text-gray-600 text-sm">
                      {strategy.ticker} | {strategy.start_date} to {strategy.end_date}
                    </p>
                  </div>
                  <div className="flex items-center space-x-2">
                    {latestBacktest && getStatusBadge(latestBacktest.status)}
                    <Link
                      to={`/strategies/${strategy.id}/edit`}
                      className="px-3 py-1 text-xs bg-blue-100 text-blue-700 rounded hover:bg-blue-200"
                      title="Edit strategy settings"
                    >
                      ✏️ Edit
                    </Link>
                    <button
                      onClick={() => handleDeleteStrategy(strategy.id, strategy.name)}
                      disabled={deletingStrategyId === strategy.id}
                      className="px-3 py-1 text-xs bg-gray-100 text-gray-700 rounded hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed"
                      title="Delete entire strategy"
                    >
                      {deletingStrategyId === strategy.id ? 'Deleting...' : '🗑️ Delete'}
                    </button>
                  </div>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                  <div>
                    <p className="text-xs text-gray-500 mb-1">Strategy Type</p>
                    <p className={`font-semibold ${getStrategyDescription(strategy).color}`}>
                      {getStrategyDescription(strategy).label}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-500 mb-1">Configuration</p>
                    <p className="font-semibold text-sm">
                      {getStrategyDescription(strategy).value}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-500 mb-1">AI Enhancement</p>
                    <p className="font-semibold">
                      {strategy.config.use_prediction ? (
                        <span className="text-purple-600">✓ LSTM</span>
                      ) : (
                        <span className="text-gray-400">None</span>
                      )}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-500 mb-1">Initial Capital</p>
                    <p className="font-semibold">
                      ${strategy.config.initial_capital.toLocaleString()}
                    </p>
                  </div>
                </div>

                {latestBacktest && latestBacktest.status === 'SUCCESS' && (
                  <div className="border-t pt-4 mt-4">
                    <h4 className="font-semibold mb-3">Backtest Results</h4>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                      <div>
                        <p className="text-xs text-gray-500 mb-1">Total Return</p>
                        <p className={`font-bold ${(latestBacktest.total_return || 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                          {latestBacktest.total_return?.toFixed(2)}%
                        </p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-500 mb-1">Sharpe Ratio</p>
                        <p className="font-semibold">
                          {latestBacktest.sharpe_ratio?.toFixed(2)}
                        </p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-500 mb-1">Max Drawdown</p>
                        <p className="font-semibold text-red-600">
                          {latestBacktest.max_drawdown?.toFixed(2)}%
                        </p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-500 mb-1">Win Rate</p>
                        <p className="font-semibold">
                          {latestBacktest.total_trades && latestBacktest.total_trades > 0
                            ? ((latestBacktest.winning_trades || 0) / latestBacktest.total_trades * 100).toFixed(1)
                            : '0'}%
                        </p>
                      </div>
                    </div>

                    <div className="flex justify-end">
                      <Link
                        to={`/results/${latestBacktest.id}`}
                        className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
                      >
                        View Detailed Results →
                      </Link>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
