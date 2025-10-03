import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

export default function Navbar() {
  const location = useLocation();
  const navigate = useNavigate();
  const { isAuthenticated, user, logout } = useAuth();

  const isActive = (path: string) => location.pathname === path;

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <nav className="bg-gray-800 shadow-lg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex">
            <div className="flex-shrink-0 flex items-center">
              <h1 className="text-xl font-bold text-white">
                AI Trading Platform
              </h1>
            </div>
            {isAuthenticated && (
              <div className="ml-6 flex space-x-4">
                <Link
                  to="/"
                  className={`inline-flex items-center px-3 py-2 text-sm font-medium ${
                    isActive('/')
                      ? 'text-white border-b-2 border-blue-500'
                      : 'text-gray-300 hover:text-white hover:border-gray-300'
                  }`}
                >
                  Dashboard
                </Link>
                <Link
                  to="/strategies/new"
                  className={`inline-flex items-center px-3 py-2 text-sm font-medium ${
                    isActive('/strategies/new')
                      ? 'text-white border-b-2 border-blue-500'
                      : 'text-gray-300 hover:text-white hover:border-gray-300'
                  }`}
                >
                  Create Strategy
                </Link>
              </div>
            )}
          </div>

          <div className="flex items-center">
            {isAuthenticated ? (
              <div className="flex items-center space-x-4">
                <span className="text-gray-300 text-sm">
                  {user?.full_name || user?.email}
                </span>
                <button
                  onClick={handleLogout}
                  className="px-3 py-2 text-sm font-medium text-gray-300 hover:text-white"
                >
                  Logout
                </button>
              </div>
            ) : (
              <div className="flex space-x-4">
                <Link
                  to="/login"
                  className="px-3 py-2 text-sm font-medium text-gray-300 hover:text-white"
                >
                  Login
                </Link>
                <Link
                  to="/register"
                  className="px-3 py-2 text-sm font-medium bg-blue-600 text-white rounded hover:bg-blue-700"
                >
                  Sign Up
                </Link>
              </div>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
}
