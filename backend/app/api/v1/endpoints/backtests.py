from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from celery.result import AsyncResult
from slowapi import Limiter
from slowapi.util import get_remote_address
from app.db.session import get_db
from app.schemas.backtest_result import (
    BacktestTaskResponse,
    BacktestStatusResponse,
    BacktestResultResponse
)
from app.crud import crud_strategy, crud_backtest
from app.core.celery_app import celery_app
from app.tasks.run_backtest_task import run_backtest_task
from app.core.deps import get_current_user
from app.models.user import User

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.post("/strategies/{strategy_id}/run", response_model=BacktestTaskResponse, status_code=status.HTTP_202_ACCEPTED)
@limiter.limit("20/hour")
async def run_backtest(
    request: Request,
    strategy_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Start a backtest for a strategy. Returns immediately with a task ID.
    The backtest runs in the background via Celery.
    Rate limit: 20 backtests per hour per IP.
    """
    # Check if strategy exists and user owns it
    strategy = await crud_strategy.get_strategy(db, strategy_id)
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy with ID {strategy_id} not found"
        )
    if strategy.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to run backtest for this strategy"
        )

    # Start Celery task
    task = run_backtest_task.delay(strategy_id)

    # Create backtest result entry
    backtest = await crud_backtest.create_backtest_result(db, strategy_id, task.id)

    return BacktestTaskResponse(
        task_id=task.id,
        status="PENDING",
        message=f"Backtest started for strategy {strategy_id}"
    )


@router.get("/status/{task_id}", response_model=BacktestStatusResponse)
async def get_backtest_status(task_id: str):
    """
    Get the status of a running backtest by task ID.
    Poll this endpoint to check progress.
    """
    task_result = AsyncResult(task_id, app=celery_app)

    response = BacktestStatusResponse(
        task_id=task_id,
        status=task_result.state
    )

    if task_result.state == 'PENDING':
        response.current = 0
        response.total = 100
    elif task_result.state == 'PROGRESS':
        info = task_result.info
        response.current = info.get('current', 0)
        response.total = info.get('total', 100)
    elif task_result.state == 'SUCCESS':
        response.result = task_result.result
    elif task_result.state == 'FAILURE':
        response.error = str(task_result.info)

    return response


@router.get("/{backtest_id}", response_model=BacktestResultResponse)
async def get_backtest_result(
    backtest_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get the full results of a completed backtest by backtest ID.
    """
    backtest = await crud_backtest.get_backtest_result(db, backtest_id)
    if not backtest:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Backtest with ID {backtest_id} not found"
        )

    return backtest


@router.get("/", response_model=List[BacktestResultResponse])
async def list_all_backtests(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """List all backtest results"""
    backtests = await crud_backtest.get_all_backtests(db, skip=skip, limit=limit)
    return backtests


@router.get("/strategy/{strategy_id}/results", response_model=List[BacktestResultResponse])
async def get_strategy_backtests(
    strategy_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get all backtest results for a specific strategy"""
    backtests = await crud_backtest.get_backtests_by_strategy(db, strategy_id)
    return backtests


@router.delete("/{backtest_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_backtest(
    backtest_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a backtest result"""
    # Check if backtest exists
    backtest = await crud_backtest.get_backtest_result(db, backtest_id)
    if not backtest:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Backtest with ID {backtest_id} not found"
        )

    # Check if user owns the strategy associated with this backtest
    strategy = await crud_strategy.get_strategy(db, backtest.strategy_id)
    if not strategy or strategy.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this backtest"
        )

    # Delete the backtest
    success = await crud_backtest.delete_backtest_result(db, backtest_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete backtest"
        )

    return None
