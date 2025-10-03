# AI-Powered Algorithmic Trading & Backtesting Platform

A production-grade web application for creating, backtesting, and analyzing algorithmic trading strategies with AI-powered signals including sentiment analysis and price prediction.

## 🏗️ Architecture Overview

### Backend Stack
- **FastAPI**: High-performance async web framework
- **PostgreSQL**: Relational database with async support (asyncpg)
- **Celery**: Distributed task queue for background backtesting
- **Redis**: Message broker and result backend for Celery
- **SQLAlchemy**: ORM with async support
- **Alpaca API**: Live market data source

### Frontend Stack
- **React 18** with **TypeScript**
- **Vite**: Next-generation build tool
- **TailwindCSS**: Utility-first CSS framework
- **TradingView Lightweight Charts**: Professional charting library
- **React Router**: Client-side routing
- **Axios**: HTTP client

### AI/ML Components
- **Sentiment Analysis**: Simulated sentiment scoring (production would use FinBERT)
- **Price Prediction**: Technical analysis-based prediction signals (LSTM-inspired)
- **Technical Indicators**: Moving averages, RSI, momentum, mean reversion

## 📁 Project Structure

```
backtesting/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── v1/
│   │   │       ├── endpoints/
│   │   │       │   ├── strategies.py      # Strategy CRUD endpoints
│   │   │       │   └── backtests.py       # Backtest execution & status
│   │   │       └── api.py                 # API router aggregation
│   │   ├── core/
│   │   │   ├── config.py                  # Pydantic settings
│   │   │   └── celery_app.py              # Celery configuration
│   │   ├── crud/
│   │   │   ├── crud_strategy.py           # Strategy database operations
│   │   │   └── crud_backtest.py           # Backtest database operations
│   │   ├── db/
│   │   │   └── session.py                 # Database session & engine
│   │   ├── models/
│   │   │   ├── strategy.py                # Strategy SQLAlchemy model
│   │   │   └── backtest_result.py         # BacktestResult model
│   │   ├── schemas/
│   │   │   ├── strategy.py                # Pydantic schemas for strategies
│   │   │   └── backtest_result.py         # Pydantic schemas for results
│   │   ├── services/
│   │   │   ├── data_service.py            # Alpaca API data fetching
│   │   │   ├── ai_signals.py              # Sentiment & prediction signals
│   │   │   └── backtester.py              # Core backtesting engine
│   │   └── tasks/
│   │       └── run_backtest_task.py       # Celery background task
│   ├── main.py                            # FastAPI application entrypoint
│   ├── requirements.txt                   # Python dependencies
│   └── Dockerfile                         # Backend container
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Navbar.tsx                 # Navigation bar
│   │   │   ├── TradingChart.tsx           # Price chart with trade markers
│   │   │   └── EquityChart.tsx            # Portfolio equity curve
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx              # Main dashboard view
│   │   │   ├── StrategyBuilder.tsx        # Strategy creation form
│   │   │   └── ResultsPage.tsx            # Detailed backtest results
│   │   ├── services/
│   │   │   └── api.ts                     # API client
│   │   ├── types/
│   │   │   └── index.ts                   # TypeScript type definitions
│   │   ├── App.tsx                        # Root component
│   │   ├── main.tsx                       # Application entry point
│   │   └── index.css                      # Global styles
│   ├── package.json                       # Frontend dependencies
│   ├── vite.config.ts                     # Vite configuration
│   ├── tailwind.config.js                 # Tailwind configuration
│   └── Dockerfile                         # Frontend container
├── docker-compose.yml                     # Multi-container orchestration
├── .env.example                           # Environment variables template
└── README.md                              # This file
```

## 🚀 Getting Started

### Prerequisites
- Docker & Docker Compose installed
- (Optional) Alpaca API account for live market data

### 1. Clone & Setup

```bash
# Navigate to project directory
cd backtesting

# Copy environment template
cp .env.example .env

# (Optional) Edit .env with your Alpaca API credentials
# If no credentials provided, the system will use simulated data
nano .env
```

### 2. Launch the Application

```bash
# Build and start all services
docker-compose up --build

# Or run in detached mode
docker-compose up --build -d
```

This command will:
1. Start PostgreSQL database on port 5432
2. Start Redis message broker on port 6379
3. Build and start FastAPI backend on port 8000
4. Build and start Celery worker for background tasks
5. Build and start React frontend on port 5173

### 3. Access the Application

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### 4. Stop the Application

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (clears database)
docker-compose down -v
```

## 📊 How It Works

### 1. Create a Strategy
Navigate to "Create Strategy" and define:
- **Basic Info**: Name, ticker symbol, date range
- **Parameters**: Moving average periods, initial capital, position size
- **AI Signals**: Enable sentiment analysis and/or price prediction

### 2. Backtest Execution
When you create a strategy:
1. Strategy is saved to PostgreSQL
2. Celery task is queued immediately
3. Worker fetches historical data via Alpaca API
4. AI signals are generated (if enabled)
5. Backtesting engine simulates trades day-by-day
6. Results are saved to database
7. Frontend polls for completion

### 3. View Results
The results page displays:
- **Key Metrics**: Total return, Sharpe ratio, max drawdown
- **Trade Statistics**: Total trades, win rate, winning/losing trades
- **Equity Curve Chart**: Portfolio value over time
- **Price Action Chart**: Price history with BUY/SELL markers
- **Trade Log Table**: Detailed transaction history

## 🧠 Trading Strategy Logic

### Base Strategy: Moving Average Crossover
- **Buy Signal**: Short MA crosses above Long MA
- **Sell Signal**: Short MA crosses below Long MA

### AI Enhancement (Optional)
- **Sentiment Analysis**: Requires positive sentiment (>-0.2) to confirm buy
- **Price Prediction**: Uses prediction probability to validate signals
- **Combined**: Both signals must agree or be neutral

## 🔧 Configuration

### Backend Environment Variables
```env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@db:5432/trading_db
CELERY_BROKER_URL=redis://redis:6379/0
ALPACA_API_KEY=your_api_key
ALPACA_API_SECRET=your_api_secret
```

### Frontend Environment Variables
```env
VITE_API_URL=http://localhost:8000/api/v1
```

## 🛠️ Development

### Backend Development
```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Run FastAPI server
uvicorn main:app --reload

# Run Celery worker
celery -A app.core.celery_app worker -l info
```

### Frontend Development
```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build
```

## 📡 API Endpoints

### Strategies
- `POST /api/v1/strategies/` - Create new strategy
- `GET /api/v1/strategies/` - List all strategies
- `GET /api/v1/strategies/{id}` - Get strategy by ID
- `PUT /api/v1/strategies/{id}` - Update strategy
- `DELETE /api/v1/strategies/{id}` - Delete strategy

### Backtests
- `POST /api/v1/backtests/strategies/{id}/run` - Start backtest
- `GET /api/v1/backtests/status/{task_id}` - Check task status
- `GET /api/v1/backtests/{id}` - Get backtest results
- `GET /api/v1/backtests/` - List all backtests
- `GET /api/v1/backtests/strategy/{id}/results` - Get strategy's backtests

## 🎨 Key Features

✅ **Async Architecture**: FastAPI + asyncpg for high performance
✅ **Background Processing**: Celery for non-blocking backtests
✅ **Real-time Updates**: Dashboard polls for backtest progress
✅ **Professional Charts**: TradingView Lightweight Charts
✅ **Type Safety**: Full TypeScript frontend
✅ **Responsive Design**: TailwindCSS with mobile support
✅ **Containerized**: Easy deployment with Docker Compose
✅ **Modular Design**: Clear separation of concerns
✅ **Live Data**: Alpaca API integration (with fallback simulation)
✅ **AI Integration**: Sentiment + prediction signals

## 🔐 Alpaca API Setup

1. Sign up at https://alpaca.markets/
2. Generate API keys (paper trading recommended)
3. Add keys to `.env` file
4. Restart services: `docker-compose restart`

Without API keys, the system generates realistic simulated data.

## 📈 Performance Metrics

The platform calculates:
- **Total Return**: Percentage gain/loss
- **Sharpe Ratio**: Risk-adjusted return (annualized)
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Total Trades**: Number of round-trip trades

## 🐛 Troubleshooting

### Database Connection Issues
```bash
# Check database status
docker-compose ps db

# View database logs
docker-compose logs db
```

### Celery Worker Not Processing
```bash
# Check worker status
docker-compose ps worker

# View worker logs
docker-compose logs worker
```

### Frontend Can't Connect to Backend
1. Check backend is running: `docker-compose ps backend`
2. Verify CORS settings in `backend/app/core/config.py`
3. Check frontend env: `frontend/.env`

## 🚢 Production Deployment

For production:
1. Use production-grade database (managed PostgreSQL)
2. Enable Redis persistence
3. Configure environment-specific secrets
4. Set up reverse proxy (nginx)
5. Enable HTTPS
6. Configure logging and monitoring
7. Scale Celery workers as needed

## 📝 License

This project is created as a demonstration platform for educational purposes.

## 🤝 Contributing

This is a complete, self-contained trading platform. Feel free to extend with:
- Additional technical indicators
- Real sentiment analysis (FinBERT integration)
- More sophisticated ML models
- Live trading capabilities
- Multiple asset support
- Advanced risk management

---

**Built with ❤️ using FastAPI, React, and modern DevOps practices**
