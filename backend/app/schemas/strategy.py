from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Literal
from datetime import datetime


class LSTMConfig(BaseModel):
    """LSTM model configuration parameters"""
    lookback_period: int = Field(default=60, ge=10, le=365, description="Number of days to look back for prediction")
    epochs: int = Field(default=50, ge=10, le=200, description="Number of training epochs")
    batch_size: int = Field(default=32, ge=8, le=128, description="Training batch size")
    lstm_units: int = Field(default=50, ge=20, le=200, description="Number of LSTM units in hidden layer")
    dropout_rate: float = Field(default=0.2, ge=0.0, le=0.5, description="Dropout rate for regularization")
    train_test_split: float = Field(default=0.8, ge=0.5, le=0.95, description="Fraction of data for training")


class IndicatorConfig(BaseModel):
    """Configuration for a technical indicator"""
    id: str = Field(..., description="Indicator ID (e.g., 'rsi', 'macd')")
    params: Dict[str, Any] = Field(default_factory=dict, description="Indicator parameters")


class StrategyConfig(BaseModel):
    # Strategy mode: 'simple' (MA crossover), 'indicators' (visual builder), 'custom' (Python code)
    mode: Literal['simple', 'indicators', 'custom'] = Field(default='simple')

    # Simple MA crossover parameters (mode='simple')
    ma_short: int = Field(default=50, ge=1)
    ma_long: int = Field(default=200, ge=1)

    # Indicator-based strategy (mode='indicators')
    indicators: Optional[List[IndicatorConfig]] = Field(default=None, description="List of indicators to calculate")
    entry_conditions: Optional[List[str]] = Field(default=None, description="Conditions for entry (AND combined)")
    exit_conditions: Optional[List[str]] = Field(default=None, description="Conditions for exit (AND combined)")

    # Custom Python code (mode='custom')
    custom_code: Optional[str] = Field(default=None, description="Custom Python strategy code")

    # Common parameters
    use_prediction: bool = Field(default=False)
    lstm_config: Optional[LSTMConfig] = Field(default=None, description="LSTM configuration if use_prediction is True")
    initial_capital: float = Field(default=100000.0, gt=0)
    position_size: float = Field(default=1.0, ge=0.1, le=1.0)  # Fraction of capital per trade


class StrategyCreate(BaseModel):
    name: str = Field(..., min_length=1)
    ticker: str = Field(..., min_length=1)
    start_date: str = Field(..., description="ISO format date (YYYY-MM-DD)")
    end_date: str = Field(..., description="ISO format date (YYYY-MM-DD)")
    config: StrategyConfig


class StrategyUpdate(BaseModel):
    name: Optional[str] = None
    ticker: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class StrategyResponse(BaseModel):
    id: int
    name: str
    ticker: str
    start_date: str
    end_date: str
    config: Dict[str, Any]
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True
