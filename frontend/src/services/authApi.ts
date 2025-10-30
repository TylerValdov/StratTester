import axios from 'axios';
import type { LoginCredentials, RegisterData, AuthToken, User } from '../types/auth';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001/api/v1';

const authApi = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const authService = {
  register: async (data: RegisterData): Promise<User> => {
    const response = await authApi.post('/auth/register', data);
    return response.data;
  },

  login: async (credentials: LoginCredentials): Promise<AuthToken> => {
    const response = await authApi.post('/auth/login', credentials);
    return response.data;
  },

  getCurrentUser: async (token: string): Promise<User> => {
    const response = await authApi.get('/auth/me', {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
    return response.data;
  },
};

export default authApi;
