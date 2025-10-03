# AI-Powered Algorithmic Trading & Backtesting Platform

A full-stack web application for creating, backtesting, and analyzing algorithmic trading strategies with machine learning enhancements. Features custom Python strategy execution, real-time market data integration, and LSTM-based price prediction.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![React](https://img.shields.io/badge/React-18-blue)
![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)

---

## 🎯 Project Overview

Professional-grade trading platform that enables users to design, backtest, and analyze algorithmic trading strategies with three distinct approaches:

- **Simple MA Crossover**: Classic moving average strategy for beginners
- **Visual Indicator Builder**: Combine 10+ technical indicators with intelligent signal generation
- **Custom Python Strategies**: Write custom trading logic with full indicator support in a sandboxed environment

### Key Capabilities

- **Real-time Market Data**: Integration with Alpaca Markets API for historical stock data
- **Machine Learning**: LSTM neural networks for price prediction and trend forecasting
- **Advanced Analytics**: Comprehensive performance metrics including Sharpe ratio, drawdown analysis, and win rates
- **Async Processing**: Background task execution for non-blocking backtests
- **Benchmark Comparison**: Compare strategy performance against SPY and buy-and-hold approaches
- **Production-Ready**: Fully containerized with Docker for consistent deployment

---

## 🏗️ Technology Stack

### Backend
- **FastAPI** - High-performance async Python web framework
- **PostgreSQL** - Relational database with async support (asyncpg)
- **SQLAlchemy** - Modern ORM with async capabilities
- **Celery** - Distributed task queue for background processing
- **Redis** - Message broker and caching layer
- **TensorFlow/Keras** - LSTM neural network implementation
- **pandas & NumPy** - Financial data analysis and computation
- **Alpaca API** - Real-time and historical market data
- **RestrictedPython** - Sandboxed execution of user-defined strategies

### Frontend
- **React 18** - Modern UI library with hooks
- **TypeScript** - Type-safe JavaScript
- **Vite** - Next-generation build tool and dev server
- **TailwindCSS** - Utility-first CSS framework
- **Recharts** - Composable charting library for data visualization
- **React Router** - Client-side routing
- **Axios** - Promise-based HTTP client

### Infrastructure
- **Docker & Docker Compose** - Containerization and orchestration
- **uvicorn** - ASGI server for FastAPI
- **JWT Authentication** - Secure user authentication

### Machine Learning & Analytics
- **LSTM (Long Short-Term Memory)** - Sequential neural network for price prediction
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Stochastic Oscillator, CCI, ADX, ATR, OBV, SMA/EMA
- **Custom Strategy Execution** - Safe Python code execution with RestrictedPython
- **Multi-Indicator Signal Generation** - Majority voting system for combining indicator signals

---

## ✨ Features

### Strategy Creation
- **Three Strategy Modes**: Simple, Visual Indicators, Custom Python Code
- **10+ Technical Indicators**: Momentum, trend, volatility, and volume indicators
- **LSTM Price Prediction**: Optional ML-based price forecasting
- **Custom Code Editor**: Write Python strategies with template support and validation
- **Parameter Optimization**: Configurable position sizing and capital allocation

### Backtesting Engine
- **Event-Driven Simulation**: Day-by-day trade execution
- **Realistic Position Sizing**: Fractional share support
- **Multiple Signal Sources**: Technical indicators, AI predictions, custom logic
- **Stock Split Handling**: Automatic price and volume adjustments
- **Benchmark Comparison**: SPY index and buy-and-hold strategies

### Analytics & Visualization
- **Performance Metrics**:
  - Total Return & Final Portfolio Value
  - Sharpe Ratio (risk-adjusted returns)
  - Maximum Drawdown
  - Win Rate & Trade Statistics
- **Interactive Charts**:
  - Equity curve with benchmark overlay
  - Price action with buy/sell markers
  - Indicator overlays
- **Detailed Trade Logs**: Complete transaction history with P&L

### User Experience
- **Real-time Updates**: Polling for backtest status
- **Responsive Design**: Mobile-friendly interface
- **Strategy Management**: Edit, delete, and re-run backtests
- **Multi-Strategy Dashboard**: Compare strategies at a glance
- **Error Handling**: Comprehensive validation and error messages

---

## 🚀 Getting Started

### Prerequisites
- Docker Desktop installed
- (Optional) Alpaca API account for live market data: https://alpaca.markets/

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ai-trading-backtester.git
cd ai-trading-backtester
```

2. **Configure environment**
```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your Alpaca API credentials (optional)
# If not provided, system uses simulated data
```

3. **Launch with Docker**
```bash
# Build and start all services
docker-compose up --build -d

# Check service status
docker-compose ps
```

4. **Access the application**
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### Services Started
- **PostgreSQL** (port 5432) - Database
- **Redis** (port 6379) - Task broker
- **FastAPI Backend** (port 8000) - REST API
- **Celery Worker** - Background task processor
- **React Frontend** (port 5173) - Web UI

---

## 📊 How It Works

### Strategy Execution Flow

1. **User creates strategy** via web interface
2. **API saves strategy** to PostgreSQL database
3. **Celery task queued** for background processing
4. **Worker fetches data** from Alpaca API
5. **LSTM model runs** (if enabled) for price predictions
6. **Backtesting engine** simulates trades day-by-day
7. **Results calculated** and saved to database
8. **Frontend displays** analytics and visualizations

### Signal Generation

#### Simple Mode
- Moving average crossover (golden cross/death cross)
- Optional LSTM prediction filter

#### Indicator Mode
- Each indicator generates buy/sell signals using technical analysis rules
- Majority voting system combines signals
- ADX filter ensures strong trends (>25)

#### Custom Mode
- User-provided Python code executed in sandboxed environment
- Access to all selected indicators
- Full pandas/NumPy support
- Security restrictions prevent harmful operations

---

## 🧠 Technical Indicators

The platform implements these indicators from scratch using pandas and NumPy:

| Indicator | Category | Use Case |
|-----------|----------|----------|
| **RSI** (Relative Strength Index) | Momentum | Overbought/oversold conditions |
| **MACD** (Moving Average Convergence Divergence) | Trend | Trend direction and momentum |
| **Bollinger Bands** | Volatility | Price extremes and mean reversion |
| **Stochastic Oscillator** | Momentum | Overbought/oversold with momentum |
| **SMA/EMA** (Moving Averages) | Trend | Trend direction and support/resistance |
| **CCI** (Commodity Channel Index) | Momentum | Cyclical trends and extremes |
| **ADX** (Average Directional Index) | Trend | Trend strength (not direction) |
| **ATR** (Average True Range) | Volatility | Position sizing and stop losses |
| **OBV** (On Balance Volume) | Volume | Volume-price confirmation |
| **Williams %R** | Momentum | Overbought/oversold timing |

---

## 🤖 Machine Learning Architecture

### LSTM Price Prediction

- **Architecture**: Sequential neural network with LSTM layers
- **Input Features**: Multi-day price sequences (configurable lookback)
- **Output**: Next-day price probability
- **Training**: Automated training on historical data per ticker
- **Integration**: Optional signal filter in backtesting engine
- **Framework**: TensorFlow/Keras

### Model Pipeline
1. Fetch historical price data
2. Create time-series sequences
3. Normalize data for training
4. Train LSTM with validation split
5. Generate predictions for backtest period
6. Apply predictions as signal filters

---

## 🔧 Configuration

### Environment Variables

**Backend** (`.env`):
```env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@db:5432/trading_db
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0
SECRET_KEY=your_secret_key_here
ALPACA_API_KEY=your_alpaca_key
ALPACA_API_SECRET=your_alpaca_secret
```

**Frontend** (`frontend/.env`):
```env
VITE_API_URL=http://localhost:8000/api/v1
```

---

## 📡 API Endpoints

### Authentication
- `POST /api/v1/auth/register` - Create new user account
- `POST /api/v1/auth/login` - Login and receive JWT token

### Strategies
- `POST /api/v1/strategies/` - Create new strategy
- `GET /api/v1/strategies/` - List all user strategies
- `GET /api/v1/strategies/{id}` - Get strategy details
- `PUT /api/v1/strategies/{id}` - Update strategy configuration
- `DELETE /api/v1/strategies/{id}` - Delete strategy and all backtests

### Backtests
- `POST /api/v1/backtests/strategies/{id}/run` - Execute backtest
- `GET /api/v1/backtests/status/{task_id}` - Check execution status
- `GET /api/v1/backtests/{id}` - Get detailed results
- `GET /api/v1/backtests/` - List all backtests

### Indicators
- `GET /api/v1/indicators/list` - Available indicators with parameters
- `GET /api/v1/indicators/templates` - Strategy code templates
- `POST /api/v1/indicators/validate` - Validate custom Python code

Full API documentation available at: http://localhost:8000/docs

---

## 🎨 Architecture Highlights

### Backend Architecture
- **Async-First Design**: FastAPI with asyncpg for concurrent request handling
- **Task Queue Pattern**: Celery workers for CPU-intensive backtests
- **Repository Pattern**: CRUD operations abstracted from business logic
- **Service Layer**: Clean separation of data access and business logic
- **Schema Validation**: Pydantic models for request/response validation

### Frontend Architecture
- **Component-Based**: Reusable React components with TypeScript
- **Context API**: Global authentication state management
- **Protected Routes**: Authentication-based route guards
- **API Service Layer**: Centralized API communication
- **Responsive Design**: Mobile-first with TailwindCSS

### Security
- **JWT Authentication**: Token-based auth with expiration
- **Password Hashing**: Bcrypt for secure password storage
- **RestrictedPython**: Sandboxed execution of user code
- **CORS Protection**: Configured allowed origins
- **Environment Secrets**: Sensitive data in environment variables

---

## 🛠️ Development

### Local Development (Without Docker)

**Backend**:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload

# In separate terminal for Celery worker
celery -A app.core.celery_app worker -l info
```

**Frontend**:
```bash
cd frontend
npm install
npm run dev
```

### Testing
```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm run test
```

---

## 📈 Performance Metrics Explained

- **Total Return**: Percentage gain/loss from initial capital
- **Sharpe Ratio**: Risk-adjusted return (annualized). Higher is better. >1 is good, >2 is excellent
- **Maximum Drawdown**: Largest peak-to-trough decline. Lower is better
- **Win Rate**: Percentage of profitable trades
- **Total Trades**: Number of complete buy-sell cycles

---

## 🚀 Production Deployment Considerations

1. **Database**: Use managed PostgreSQL (AWS RDS, DigitalOcean, etc.)
2. **Redis**: Enable persistence for task result storage
3. **Secrets Management**: Use environment-specific secrets (AWS Secrets Manager, etc.)
4. **Reverse Proxy**: nginx for load balancing and SSL termination
5. **HTTPS**: Enable SSL certificates (Let's Encrypt)
6. **Monitoring**: Application logging and performance monitoring
7. **Scaling**: Horizontal scaling of Celery workers based on load
8. **CI/CD**: Automated testing and deployment pipelines

---

## 🤝 Contributing

Potential enhancements:
- Additional technical indicators (Ichimoku, Fibonacci, etc.)
- Multi-asset portfolio backtesting
- Walk-forward optimization
- Monte Carlo simulation
- Real-time paper trading
- Advanced risk management (stop-loss, take-profit)
- Portfolio optimization algorithms
- Integration with additional data providers

---

## 📝 License

This project is available for educational and portfolio demonstration purposes.

---

## 👨‍💻 Author

Built as a demonstration of full-stack development capabilities including:
- Modern web frameworks (FastAPI, React)
- Machine learning integration (TensorFlow/Keras)
- Financial data analysis (pandas, NumPy)
- Distributed systems (Celery, Redis)
- Containerization (Docker)
- TypeScript and type-safe development
- Asynchronous programming patterns
- RESTful API design

---

**Built with modern technologies and best practices for production-ready applications**
