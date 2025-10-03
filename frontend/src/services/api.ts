import axios from 'axios';
import type {
  Strategy,
  BacktestResult,
  BacktestTaskResponse,
  BacktestStatusResponse,
  IndicatorDefinition,
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001/api/v1';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor to include auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add response interceptor to handle auth errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Token expired or invalid
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Strategy endpoints
export const strategyApi = {
  create: async (data: Omit<Strategy, 'id' | 'created_at' | 'updated_at'>): Promise<Strategy> => {
    const response = await api.post('/strategies/', data);
    return response.data;
  },

  list: async (): Promise<Strategy[]> => {
    const response = await api.get('/strategies/');
    return response.data;
  },

  get: async (id: number): Promise<Strategy> => {
    const response = await api.get(`/strategies/${id}`);
    return response.data;
  },

  update: async (id: number, data: Partial<Strategy>): Promise<Strategy> => {
    const response = await api.put(`/strategies/${id}`, data);
    return response.data;
  },

  delete: async (id: number): Promise<void> => {
    await api.delete(`/strategies/${id}`);
  },

  deleteStrategy: async (id: number): Promise<void> => {
    await api.delete(`/strategies/${id}`);
  },
};

// Backtest endpoints
export const backtestApi = {
  run: async (strategyId: number): Promise<BacktestTaskResponse> => {
    const response = await api.post(`/backtests/strategies/${strategyId}/run`);
    return response.data;
  },

  getStatus: async (taskId: string): Promise<BacktestStatusResponse> => {
    const response = await api.get(`/backtests/status/${taskId}`);
    return response.data;
  },

  getResult: async (backtestId: number): Promise<BacktestResult> => {
    const response = await api.get(`/backtests/${backtestId}`);
    return response.data;
  },

  list: async (): Promise<BacktestResult[]> => {
    const response = await api.get('/backtests/');
    return response.data;
  },

  getStrategyResults: async (strategyId: number): Promise<BacktestResult[]> => {
    const response = await api.get(`/backtests/strategy/${strategyId}/results`);
    return response.data;
  },

  delete: async (backtestId: number): Promise<void> => {
    await api.delete(`/backtests/${backtestId}`);
  },
};

// Indicator endpoints
export const indicatorApi = {
  list: async (): Promise<IndicatorDefinition[]> => {
    const response = await api.get('/indicators/list');
    return response.data;
  },

  getTemplates: async (): Promise<Record<string, string>> => {
    const response = await api.get('/indicators/templates');
    return response.data;
  },

  getBlankTemplate: async (): Promise<{ template: string }> => {
    const response = await api.get('/indicators/template/blank');
    return response.data;
  },

  validateCode: async (code: string): Promise<{ valid: boolean; error?: string }> => {
    const response = await api.post('/indicators/validate', { code });
    return response.data;
  },
};

export default api;
