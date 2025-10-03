from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.session import Base


class Strategy(Base):
    __tablename__ = "strategies"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    ticker = Column(String, nullable=False)
    start_date = Column(String, nullable=False)  # ISO format date string
    end_date = Column(String, nullable=False)  # ISO format date string
    config = Column(JSON, nullable=False)  # Strategy parameters
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="strategies")
    backtest_results = relationship("BacktestResult", back_populates="strategy", cascade="all, delete-orphan")
    lstm_model = relationship("LSTMModel", back_populates="strategy", uselist=False, cascade="all, delete-orphan")
