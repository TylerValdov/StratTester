from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey, Float, LargeBinary
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.session import Base


class LSTMModel(Base):
    __tablename__ = "lstm_models"

    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False)
    ticker = Column(String, nullable=False)

    # Training parameters
    lstm_config = Column(JSON, nullable=False)

    # Training metadata
    training_start_date = Column(String, nullable=False)
    training_end_date = Column(String, nullable=False)

    # Training results
    status = Column(String, default="PENDING")  # PENDING, TRAINING, SUCCESS, FAILURE
    train_loss = Column(Float, nullable=True)
    val_loss = Column(Float, nullable=True)
    training_history = Column(JSON, nullable=True)  # Full training history

    # Model storage
    model_path = Column(String, nullable=True)  # Path to saved model file
    scaler_params = Column(JSON, nullable=True)  # MinMaxScaler parameters

    error_message = Column(String, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    trained_at = Column(DateTime(timezone=True), nullable=True)

    # Relationship
    strategy = relationship("Strategy", back_populates="lstm_model")
