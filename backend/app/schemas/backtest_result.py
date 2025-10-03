from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime


class BacktestTaskResponse(BaseModel):
    task_id: str
    status: str
    message: str


class BacktestStatusResponse(BaseModel):
    task_id: str
    status: str
    current: Optional[int] = None
    total: Optional[int] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class TradeEntry(BaseModel):
    date: str
    action: str  # BUY or SELL
    price: float
    shares: int
    value: float
    balance: float


class BacktestResultResponse(BaseModel):
    id: int
    strategy_id: int
    task_id: Optional[str]
    status: str

    # Performance Metrics
    final_balance: Optional[float]
    total_return: Optional[float]
    sharpe_ratio: Optional[float]
    max_drawdown: Optional[float]
    total_trades: Optional[int]
    winning_trades: Optional[int]
    losing_trades: Optional[int]

    # Detailed Data
    trade_log: Optional[List[Dict[str, Any]]]
    equity_curve: Optional[List[Dict[str, Any]]]

    # Benchmark Data
    spy_benchmark: Optional[List[Dict[str, Any]]]
    buy_hold_benchmark: Optional[List[Dict[str, Any]]]
    spy_return: Optional[float]
    buy_hold_return: Optional[float]

    error_message: Optional[str]

    created_at: datetime
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True
