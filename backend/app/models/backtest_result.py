from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.session import Base


class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False)
    task_id = Column(String, nullable=True, unique=True)  # Celery task ID
    status = Column(String, default="PENDING")  # PENDING, PROGRESS, SUCCESS, FAILURE

    # Performance Metrics
    final_balance = Column(Float, nullable=True)
    total_return = Column(Float, nullable=True)  # Percentage
    sharpe_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    total_trades = Column(Integer, nullable=True)
    winning_trades = Column(Integer, nullable=True)
    losing_trades = Column(Integer, nullable=True)

    # Detailed Data
    trade_log = Column(JSON, nullable=True)  # List of all trades
    equity_curve = Column(JSON, nullable=True)  # Portfolio value over time

    # Benchmark Data
    spy_benchmark = Column(JSON, nullable=True)  # SPY returns over time
    buy_hold_benchmark = Column(JSON, nullable=True)  # Buy-and-hold returns for the ticker
    spy_return = Column(Float, nullable=True)  # Total SPY return percentage
    buy_hold_return = Column(Float, nullable=True)  # Total buy-and-hold return percentage

    error_message = Column(String, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationship
    strategy = relationship("Strategy", back_populates="backtest_results")
