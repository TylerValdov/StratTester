import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { authService } from '../services/authApi';
import type { AuthContextType, User, LoginCredentials, RegisterData } from '../types/auth';

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Load token and user from localStorage on mount
  useEffect(() => {
    const loadUserFromStorage = async () => {
      const storedToken = localStorage.getItem('token');
      if (storedToken) {
        try {
          const userData = await authService.getCurrentUser(storedToken);
          setUser(userData);
          setToken(storedToken);
        } catch (error) {
          // Token invalid or expired
          localStorage.removeItem('token');
        }
      }
      setIsLoading(false);
    };

    loadUserFromStorage();
  }, []);

  const login = async (credentials: LoginCredentials) => {
    const authToken = await authService.login(credentials);
    const userData = await authService.getCurrentUser(authToken.access_token);

    setToken(authToken.access_token);
    setUser(userData);
    localStorage.setItem('token', authToken.access_token);
  };

  const register = async (data: RegisterData) => {
    const userData = await authService.register(data);
    // Auto-login after registration
    await login({ email: data.email, password: data.password });
  };

  const logout = () => {
    // Logout is handled client-side by clearing token and user data
    setUser(null);
    setToken(null);
    localStorage.removeItem('token');
  };

  const value = {
    user,
    token,
    login,
    register,
    logout,
    isAuthenticated: !!user,
    isLoading,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
