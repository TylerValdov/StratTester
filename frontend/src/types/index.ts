export interface LSTMConfig {
  lookback_period: number;
  epochs: number;
  batch_size: number;
  lstm_units: number;
  dropout_rate: number;
  train_test_split: number;
}

export interface IndicatorParam {
  name: string;
  type: string;
  default: number;
  min?: number;
  max?: number;
}

export interface IndicatorDefinition {
  id: string;
  name: string;
  category: string;
  params: IndicatorParam[];
}

export interface IndicatorConfig {
  id: string;
  params: Record<string, any>;
}

export interface StrategyConfig {
  mode?: 'simple' | 'indicators' | 'custom';
  // Simple mode (MA crossover)
  ma_short: number;
  ma_long: number;
  // Indicator mode
  indicators?: IndicatorConfig[];
  entry_conditions?: string[];
  exit_conditions?: string[];
  // Custom mode
  custom_code?: string;
  // Common
  use_prediction: boolean;
  lstm_config?: LSTMConfig;
  initial_capital: number;
  position_size: number;
}

export interface Strategy {
  id: number;
  name: string;
  ticker: string;
  start_date: string;
  end_date: string;
  config: StrategyConfig;
  created_at: string;
  updated_at?: string;
}

export interface Trade {
  date: string;
  action: 'BUY' | 'SELL';
  price: number;
  shares: number;
  value: number;
  balance: number;
}

export interface EquityPoint {
  date: string;
  equity: number;
}

export interface BacktestResult {
  id: number;
  strategy_id: number;
  task_id?: string;
  status: string;
  final_balance?: number;
  total_return?: number;
  sharpe_ratio?: number;
  max_drawdown?: number;
  total_trades?: number;
  winning_trades?: number;
  losing_trades?: number;
  trade_log?: Trade[];
  equity_curve?: EquityPoint[];
  spy_benchmark?: EquityPoint[];
  buy_hold_benchmark?: EquityPoint[];
  spy_return?: number;
  buy_hold_return?: number;
  error_message?: string;
  created_at: string;
  completed_at?: string;
}

export interface BacktestTaskResponse {
  task_id: string;
  status: string;
  message: string;
}

export interface BacktestStatusResponse {
  task_id: string;
  status: string;
  current?: number;
  total?: number;
  result?: any;
  error?: string;
}
