from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional
from app.models.strategy import Strategy
from app.schemas.strategy import StrategyCreate, StrategyUpdate


async def create_strategy(db: AsyncSession, strategy: StrategyCreate, user_id: int) -> Strategy:
    db_strategy = Strategy(
        user_id=user_id,
        name=strategy.name,
        ticker=strategy.ticker,
        start_date=strategy.start_date,
        end_date=strategy.end_date,
        config=strategy.config.model_dump()
    )
    db.add(db_strategy)
    await db.commit()
    await db.refresh(db_strategy)
    return db_strategy


async def get_strategy(db: AsyncSession, strategy_id: int) -> Optional[Strategy]:
    result = await db.execute(select(Strategy).where(Strategy.id == strategy_id))
    return result.scalar_one_or_none()


async def get_strategies(db: AsyncSession, user_id: int, skip: int = 0, limit: int = 100) -> List[Strategy]:
    result = await db.execute(
        select(Strategy)
        .where(Strategy.user_id == user_id)
        .offset(skip)
        .limit(limit)
    )
    return result.scalars().all()


async def update_strategy(db: AsyncSession, strategy_id: int, strategy_update: StrategyUpdate) -> Optional[Strategy]:
    from sqlalchemy import delete
    from app.models.backtest_result import BacktestResult
    from app.models.lstm_model import LSTMModel

    db_strategy = await get_strategy(db, strategy_id)
    if not db_strategy:
        return None

    # Delete all old backtests and LSTM models when updating strategy
    await db.execute(delete(BacktestResult).where(BacktestResult.strategy_id == strategy_id))
    await db.execute(delete(LSTMModel).where(LSTMModel.strategy_id == strategy_id))

    update_data = strategy_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_strategy, field, value)

    await db.commit()
    await db.refresh(db_strategy)
    return db_strategy


async def delete_strategy(db: AsyncSession, strategy_id: int) -> bool:
    db_strategy = await get_strategy(db, strategy_id)
    if not db_strategy:
        return False

    await db.delete(db_strategy)
    await db.commit()
    return True
