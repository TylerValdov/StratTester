import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { backtestApi } from '../services/api';
import type { BacktestResult } from '../types';
import TradingChart from '../components/TradingChart';
import EquityChart from '../components/EquityChart';
import BenchmarkChart from '../components/BenchmarkChart';

export default function ResultsPage() {
  const { backtestId } = useParams<{ backtestId: string }>();
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadResults = async () => {
      if (!backtestId) return;

      try {
        setLoading(true);
        const data = await backtestApi.getResult(parseInt(backtestId));
        setResult(data);
      } catch (err: any) {
        setError(err.response?.data?.detail || 'Failed to load results');
      } finally {
        setLoading(false);
      }
    };

    loadResults();
  }, [backtestId]);

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="text-gray-500">Loading results...</div>
      </div>
    );
  }

  if (error || !result) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          {error || 'Results not found'}
        </div>
      </div>
    );
  }

  if (result.status !== 'SUCCESS') {
    return (
      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="bg-yellow-100 border border-yellow-400 text-yellow-700 px-4 py-3 rounded">
          Backtest is still {result.status.toLowerCase()}...
          {result.status === 'FAILURE' && result.error_message && (
            <p className="mt-2 text-sm">Error: {result.error_message}</p>
          )}
        </div>
      </div>
    );
  }

  // Prepare price data from trades for chart
  const priceData = result.trade_log?.map((trade) => ({
    time: trade.date,
    value: trade.price,
  })) || [];

  // Prepare equity curve data
  const equityData = result.equity_curve?.map((point) => ({
    time: point.date,
    value: point.equity,
  })) || [];

  // Prepare benchmark data
  const buyHoldData = result.buy_hold_benchmark?.map((point) => ({
    time: point.date,
    value: point.equity,
  })) || [];

  const spyData = result.spy_benchmark?.map((point) => ({
    time: point.date,
    value: point.equity,
  })) || [];

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="mb-8">
        <Link to="/" className="text-blue-600 hover:text-blue-800 mb-4 inline-block">
          ← Back to Dashboard
        </Link>
        <h1 className="text-3xl font-bold">Backtest Results</h1>
        <p className="text-gray-600 mt-2">
          Backtest ID: {result.id} | Completed: {new Date(result.completed_at || '').toLocaleString()}
        </p>
      </div>

      {/* Performance Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-sm font-medium text-gray-500 mb-2">Strategy Return</h3>
          <p className={`text-3xl font-bold ${(result.total_return || 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
            {result.total_return?.toFixed(2)}%
          </p>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-sm font-medium text-gray-500 mb-2">Buy & Hold Return</h3>
          <p className={`text-3xl font-bold ${(result.buy_hold_return || 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
            {result.buy_hold_return?.toFixed(2)}%
          </p>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-sm font-medium text-gray-500 mb-2">SPY Return</h3>
          <p className={`text-3xl font-bold ${(result.spy_return || 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
            {result.spy_return !== null && result.spy_return !== undefined ? `${result.spy_return.toFixed(2)}%` : 'N/A'}
          </p>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-sm font-medium text-gray-500 mb-2">Sharpe Ratio</h3>
          <p className="text-3xl font-bold text-blue-600">
            {result.sharpe_ratio?.toFixed(2)}
          </p>
        </div>
      </div>

      {/* Additional Metrics Row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-sm font-medium text-gray-500 mb-2">Max Drawdown</h3>
          <p className="text-3xl font-bold text-red-600">
            {result.max_drawdown?.toFixed(2)}%
          </p>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-sm font-medium text-gray-500 mb-2">Final Balance</h3>
          <p className="text-3xl font-bold text-gray-800">
            ${result.final_balance?.toLocaleString()}
          </p>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-sm font-medium text-gray-500 mb-2">Alpha vs Buy & Hold</h3>
          <p className={`text-3xl font-bold ${((result.total_return || 0) - (result.buy_hold_return || 0)) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
            {((result.total_return || 0) - (result.buy_hold_return || 0)).toFixed(2)}%
          </p>
        </div>
      </div>

      {/* Additional Metrics */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-8">
        <h2 className="text-xl font-semibold mb-4">Trade Statistics</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          <div>
            <p className="text-sm text-gray-500 mb-1">Total Trades</p>
            <p className="text-2xl font-semibold">{result.total_trades}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500 mb-1">Winning Trades</p>
            <p className="text-2xl font-semibold text-green-600">{result.winning_trades}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500 mb-1">Losing Trades</p>
            <p className="text-2xl font-semibold text-red-600">{result.losing_trades}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500 mb-1">Win Rate</p>
            <p className="text-2xl font-semibold">
              {result.total_trades && result.total_trades > 0
                ? ((result.winning_trades || 0) / result.total_trades * 100).toFixed(1)
                : '0'}%
            </p>
          </div>
        </div>
      </div>

      {/* Benchmark Comparison Chart */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-8">
        <h2 className="text-xl font-semibold mb-4">Performance Comparison</h2>
        <div className="mb-4 flex items-center gap-6 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-4 h-0.5 bg-blue-600"></div>
            <span className="text-gray-700">Strategy</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-0.5 bg-orange-500"></div>
            <span className="text-gray-700">Buy & Hold</span>
          </div>
          {spyData.length > 0 && (
            <div className="flex items-center gap-2">
              <div className="w-4 h-0.5 bg-green-500"></div>
              <span className="text-gray-700">SPY</span>
            </div>
          )}
        </div>
        {equityData.length > 0 ? (
          <BenchmarkChart
            strategyData={equityData}
            buyHoldData={buyHoldData}
            spyData={spyData}
            height={400}
          />
        ) : (
          <p className="text-gray-500">No equity data available</p>
        )}
      </div>

      {/* Price Chart with Trade Markers */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-8">
        <h2 className="text-xl font-semibold mb-4">Price Action & Trade Signals</h2>
        {priceData.length > 0 && result.trade_log ? (
          <TradingChart priceData={priceData} trades={result.trade_log} height={400} />
        ) : (
          <p className="text-gray-500">No price data available</p>
        )}
      </div>

      {/* Trade Log Table */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold mb-4">Trade Log</h2>
        {result.trade_log && result.trade_log.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Date
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Action
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Price
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Shares
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Value
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Balance
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {result.trade_log.map((trade, idx) => (
                  <tr key={idx} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {trade.date}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`px-2 py-1 text-xs font-semibold rounded ${
                        trade.action === 'BUY'
                          ? 'bg-green-100 text-green-800'
                          : 'bg-red-100 text-red-800'
                      }`}>
                        {trade.action}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      ${trade.price.toFixed(2)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {trade.shares}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      ${trade.value.toLocaleString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      ${trade.balance.toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-gray-500">No trades executed</p>
        )}
      </div>
    </div>
  );
}
