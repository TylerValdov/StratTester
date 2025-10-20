"""
Admin endpoints for platform management and optimization.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.db.session import get_db
from app.services.model_cleanup import ModelCleanupService
from app.core.deps import get_current_user
from app.models.user import User
from pydantic import BaseModel


router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


class CleanupStatsResponse(BaseModel):
    """Response model for cleanup statistics."""
    backtests_checked: Optional[int] = None
    models_cleaned: Optional[int] = None
    files_deleted: Optional[int] = None
    errors: Optional[int] = None
    files_found: Optional[int] = None
    total_models: Optional[int] = None
    models_with_files: Optional[int] = None
    total_file_size_mb: Optional[float] = None
    models_by_status: Optional[dict] = None
    error: Optional[str] = None


@router.post("/cleanup/old-models", response_model=CleanupStatsResponse)
@limiter.limit("10/hour")
async def cleanup_old_models(
    request: Request,
    days_old: int = 7,
    keep_training_history: bool = False,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Clean up model files from completed backtests older than specified days.
    Rate limit: 10 requests per hour per IP.

    This endpoint removes:
    - Model files (.keras) from disk
    - Model paths from database
    - Scaler parameters (not needed after backtesting)
    - Optionally, training history (can be large)

    This keeps:
    - All backtest results (metrics, trades, equity curves)
    - Model configuration
    - Training metrics (train_loss, val_loss)

    Args:
        days_old: Clean up models from backtests older than this many days (default: 7)
        keep_training_history: If True, keeps the full training history (default: False)

    Returns:
        Statistics about the cleanup operation
    """
    cleanup_service = ModelCleanupService()
    stats = await cleanup_service.cleanup_old_models(
        db=db,
        days_old=days_old,
        keep_training_history=keep_training_history
    )
    return CleanupStatsResponse(**stats)


@router.post("/cleanup/orphaned-files", response_model=CleanupStatsResponse)
@limiter.limit("10/hour")
async def cleanup_orphaned_files(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """
    Remove model files from disk that don't have database records.
    Rate limit: 10 requests per hour per IP.

    This is useful for cleaning up files that were left behind due to
    errors or interrupted operations.

    Returns:
        Statistics about the cleanup operation
    """
    cleanup_service = ModelCleanupService()
    stats = cleanup_service.cleanup_orphaned_model_files()
    return CleanupStatsResponse(**stats)


@router.get("/storage/stats", response_model=CleanupStatsResponse)
@limiter.limit("30/minute")
async def get_storage_stats(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get statistics about model storage usage.
    Rate limit: 30 requests per minute per IP.

    Returns:
        Current storage statistics including:
        - Total number of models
        - Number of models with files on disk
        - Total file size in MB
        - Breakdown by model status
    """
    cleanup_service = ModelCleanupService()
    stats = await cleanup_service.get_model_storage_stats(db)
    return CleanupStatsResponse(**stats)


@router.delete("/models/{model_id}")
@limiter.limit("20/hour")
async def delete_model(
    request: Request,
    model_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Manually delete a specific model's file and clean up its data.
    Rate limit: 20 requests per hour per IP.

    This removes:
    - Model file from disk
    - Model path and scaler params from database

    This keeps:
    - Model record in database
    - All backtest results

    Args:
        model_id: ID of the LSTM model to clean up

    Returns:
        Success message
    """
    cleanup_service = ModelCleanupService()
    success = await cleanup_service.cleanup_lstm_model(
        db=db,
        model_id=model_id,
        keep_training_history=True  # Keep history for manual deletions
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with ID {model_id} not found or cleanup failed"
        )

    return {"message": f"Successfully cleaned up model {model_id}"}
