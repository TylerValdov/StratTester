"""
Service for cleaning up ML model files and metadata after backtesting.

This service removes model files that are no longer needed after backtesting
completes, keeping only the essential data that users need to see.
"""
import os
import logging
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from datetime import datetime, timedelta

from app.models.lstm_model import LSTMModel
from app.models.backtest_result import BacktestResult

logger = logging.getLogger(__name__)


class ModelCleanupService:
    """Service to cleanup ML models after backtesting."""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir

    def delete_model_file(self, model_path: str) -> bool:
        """
        Delete a model file from the filesystem.

        Args:
            model_path: Path to the model file

        Returns:
            True if file was deleted, False otherwise
        """
        try:
            if os.path.exists(model_path):
                os.remove(model_path)
                logger.info(f"Deleted model file: {model_path}")
                return True
            else:
                logger.warning(f"Model file not found: {model_path}")
                return False
        except Exception as e:
            logger.error(f"Error deleting model file {model_path}: {str(e)}")
            return False

    async def cleanup_lstm_model(
        self,
        db: AsyncSession,
        model_id: int,
        keep_training_history: bool = False
    ) -> bool:
        """
        Clean up an LSTM model after backtesting.

        This removes:
        - The model file from disk
        - Scaler parameters (not needed after backtesting)
        - Optionally, training history (can be large)

        This keeps:
        - Model configuration (for reference)
        - Training metrics (train_loss, val_loss)
        - Status and error messages

        Args:
            db: Database session
            model_id: ID of the LSTM model to clean up
            keep_training_history: If True, keeps the full training history

        Returns:
            True if cleanup was successful, False otherwise
        """
        try:
            # Get the model record
            result = await db.execute(
                select(LSTMModel).where(LSTMModel.id == model_id)
            )
            model = result.scalar_one_or_none()

            if not model:
                logger.warning(f"LSTM model {model_id} not found")
                return False

            # Delete the model file from disk
            if model.model_path:
                self.delete_model_file(model.model_path)

            # Clear unnecessary database fields
            model.model_path = None  # Remove file path reference
            model.scaler_params = None  # Remove scaler params (no longer needed)

            if not keep_training_history:
                model.training_history = None  # Remove large training history

            await db.commit()
            logger.info(f"Cleaned up LSTM model {model_id}")
            return True

        except Exception as e:
            logger.error(f"Error cleaning up LSTM model {model_id}: {str(e)}")
            await db.rollback()
            return False

    async def cleanup_model_after_backtest(
        self,
        db: AsyncSession,
        strategy_id: int,
        keep_training_history: bool = False
    ) -> bool:
        """
        Clean up model data after a backtest completes successfully.

        Args:
            db: Database session
            strategy_id: ID of the strategy that was backtested
            keep_training_history: If True, keeps the full training history

        Returns:
            True if cleanup was successful, False otherwise
        """
        try:
            # Find the LSTM model for this strategy
            result = await db.execute(
                select(LSTMModel).where(LSTMModel.strategy_id == strategy_id)
            )
            model = result.scalar_one_or_none()

            if not model:
                logger.info(f"No LSTM model found for strategy {strategy_id}")
                return True  # No model to clean up, not an error

            # Only cleanup if model training was successful
            if model.status == "SUCCESS":
                return await self.cleanup_lstm_model(
                    db,
                    model.id,
                    keep_training_history
                )
            else:
                logger.info(f"Skipping cleanup for model {model.id} with status {model.status}")
                return True

        except Exception as e:
            logger.error(f"Error in cleanup_model_after_backtest: {str(e)}")
            return False

    async def cleanup_old_models(
        self,
        db: AsyncSession,
        days_old: int = 7,
        keep_training_history: bool = False
    ) -> dict:
        """
        Clean up models from completed backtests older than specified days.

        Args:
            db: Database session
            days_old: Clean up models from backtests older than this many days
            keep_training_history: If True, keeps the full training history

        Returns:
            Dictionary with cleanup statistics
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)

            # Find all completed backtests older than cutoff
            result = await db.execute(
                select(BacktestResult).where(
                    BacktestResult.status == "SUCCESS",
                    BacktestResult.completed_at < cutoff_date
                )
            )
            old_backtests = result.scalars().all()

            stats = {
                "backtests_checked": len(old_backtests),
                "models_cleaned": 0,
                "files_deleted": 0,
                "errors": 0
            }

            # Get unique strategy IDs
            strategy_ids = set(bt.strategy_id for bt in old_backtests)

            for strategy_id in strategy_ids:
                # Find associated LSTM model
                result = await db.execute(
                    select(LSTMModel).where(LSTMModel.strategy_id == strategy_id)
                )
                model = result.scalar_one_or_none()

                if model and model.status == "SUCCESS":
                    # Delete file if it exists
                    if model.model_path and os.path.exists(model.model_path):
                        if self.delete_model_file(model.model_path):
                            stats["files_deleted"] += 1

                    # Clean up database record
                    model.model_path = None
                    model.scaler_params = None
                    if not keep_training_history:
                        model.training_history = None

                    stats["models_cleaned"] += 1

            await db.commit()
            logger.info(f"Cleanup completed: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Error in cleanup_old_models: {str(e)}")
            await db.rollback()
            return {"error": str(e)}

    async def get_model_storage_stats(self, db: AsyncSession) -> dict:
        """
        Get statistics about model storage usage.

        Returns:
            Dictionary with storage statistics
        """
        try:
            # Get all LSTM models
            result = await db.execute(select(LSTMModel))
            models = result.scalars().all()

            stats = {
                "total_models": len(models),
                "models_with_files": 0,
                "total_file_size_bytes": 0,
                "models_by_status": {}
            }

            for model in models:
                # Count by status
                status = model.status
                stats["models_by_status"][status] = stats["models_by_status"].get(status, 0) + 1

                # Check file size
                if model.model_path and os.path.exists(model.model_path):
                    stats["models_with_files"] += 1
                    stats["total_file_size_bytes"] += os.path.getsize(model.model_path)

            # Convert to MB for readability
            stats["total_file_size_mb"] = round(stats["total_file_size_bytes"] / (1024 * 1024), 2)

            return stats

        except Exception as e:
            logger.error(f"Error getting storage stats: {str(e)}")
            return {"error": str(e)}

    def cleanup_orphaned_model_files(self) -> dict:
        """
        Remove model files from disk that don't have database records.

        Returns:
            Dictionary with cleanup statistics
        """
        stats = {
            "files_found": 0,
            "files_deleted": 0,
            "errors": 0
        }

        try:
            if not os.path.exists(self.model_dir):
                return stats

            for filename in os.listdir(self.model_dir):
                if filename.startswith("lstm_model_") and filename.endswith(".keras"):
                    stats["files_found"] += 1
                    filepath = os.path.join(self.model_dir, filename)

                    try:
                        os.remove(filepath)
                        stats["files_deleted"] += 1
                        logger.info(f"Deleted orphaned model file: {filepath}")
                    except Exception as e:
                        stats["errors"] += 1
                        logger.error(f"Error deleting {filepath}: {str(e)}")

            return stats

        except Exception as e:
            logger.error(f"Error in cleanup_orphaned_model_files: {str(e)}")
            return {"error": str(e)}
