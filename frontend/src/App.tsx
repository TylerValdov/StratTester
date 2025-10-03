import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { AuthProvider } from './contexts/AuthContext';
import ProtectedRoute from './components/ProtectedRoute';
import Navbar from './components/Navbar';
import Login from './pages/Login';
import Register from './pages/Register';
import Dashboard from './pages/Dashboard';
import StrategyBuilder from './pages/StrategyBuilder';
import ResultsPage from './pages/ResultsPage';

function App() {
  return (
    <Router>
      <AuthProvider>
        <div className="min-h-screen bg-gray-50">
          <Navbar />
          <Routes>
            {/* Public routes */}
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />

            {/* Protected routes */}
            <Route
              path="/"
              element={
                <ProtectedRoute>
                  <Dashboard />
                </ProtectedRoute>
              }
            />
            <Route
              path="/strategies/new"
              element={
                <ProtectedRoute>
                  <StrategyBuilder />
                </ProtectedRoute>
              }
            />
            <Route
              path="/strategies/:strategyId/edit"
              element={
                <ProtectedRoute>
                  <StrategyBuilder />
                </ProtectedRoute>
              }
            />
            <Route
              path="/results/:backtestId"
              element={
                <ProtectedRoute>
                  <ResultsPage />
                </ProtectedRoute>
              }
            />
          </Routes>
        </div>
      </AuthProvider>
    </Router>
  );
}

export default App;
