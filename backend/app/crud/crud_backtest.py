from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional
from datetime import datetime
from app.models.backtest_result import BacktestResult


async def create_backtest_result(db: AsyncSession, strategy_id: int, task_id: str) -> BacktestResult:
    db_backtest = BacktestResult(
        strategy_id=strategy_id,
        task_id=task_id,
        status="PENDING"
    )
    db.add(db_backtest)
    await db.commit()
    await db.refresh(db_backtest)
    return db_backtest


async def get_backtest_result(db: AsyncSession, backtest_id: int) -> Optional[BacktestResult]:
    result = await db.execute(select(BacktestResult).where(BacktestResult.id == backtest_id))
    return result.scalar_one_or_none()


async def get_backtest_by_task_id(db: AsyncSession, task_id: str) -> Optional[BacktestResult]:
    result = await db.execute(select(BacktestResult).where(BacktestResult.task_id == task_id))
    return result.scalar_one_or_none()


async def get_backtests_by_strategy(db: AsyncSession, strategy_id: int) -> List[BacktestResult]:
    result = await db.execute(select(BacktestResult).where(BacktestResult.strategy_id == strategy_id))
    return result.scalars().all()


async def get_all_backtests(db: AsyncSession, skip: int = 0, limit: int = 100) -> List[BacktestResult]:
    result = await db.execute(
        select(BacktestResult)
        .order_by(BacktestResult.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    return result.scalars().all()


async def update_backtest_result(
    db: AsyncSession,
    backtest_id: int,
    status: str,
    results: Optional[dict] = None,
    error_message: Optional[str] = None
) -> Optional[BacktestResult]:
    db_backtest = await get_backtest_result(db, backtest_id)
    if not db_backtest:
        return None

    db_backtest.status = status

    if status == "SUCCESS" and results:
        db_backtest.final_balance = results.get("final_balance")
        db_backtest.total_return = results.get("total_return")
        db_backtest.sharpe_ratio = results.get("sharpe_ratio")
        db_backtest.max_drawdown = results.get("max_drawdown")
        db_backtest.total_trades = results.get("total_trades")
        db_backtest.winning_trades = results.get("winning_trades")
        db_backtest.losing_trades = results.get("losing_trades")
        db_backtest.trade_log = results.get("trade_log")
        db_backtest.equity_curve = results.get("equity_curve")
        db_backtest.completed_at = datetime.utcnow()

    if status == "FAILURE" and error_message:
        db_backtest.error_message = error_message
        db_backtest.completed_at = datetime.utcnow()

    await db.commit()
    await db.refresh(db_backtest)
    return db_backtest


async def delete_backtest_result(db: AsyncSession, backtest_id: int) -> bool:
    """Delete a backtest result by ID"""
    db_backtest = await get_backtest_result(db, backtest_id)
    if not db_backtest:
        return False

    await db.delete(db_backtest)
    await db.commit()
    return True
