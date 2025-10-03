from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from app.db.session import get_db
from app.schemas.strategy import StrategyCreate, StrategyResponse, StrategyUpdate
from app.crud import crud_strategy
from app.core.deps import get_current_user
from app.models.user import User

router = APIRouter()


@router.post("/", response_model=StrategyResponse, status_code=status.HTTP_201_CREATED)
async def create_strategy(
    strategy: StrategyCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new trading strategy"""
    try:
        db_strategy = await crud_strategy.create_strategy(db, strategy, current_user.id)
        return db_strategy
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create strategy: {str(e)}"
        )


@router.get("/", response_model=List[StrategyResponse])
async def list_strategies(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List all strategies for current user"""
    strategies = await crud_strategy.get_strategies(db, current_user.id, skip=skip, limit=limit)
    return strategies


@router.get("/{strategy_id}", response_model=StrategyResponse)
async def get_strategy(
    strategy_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get a specific strategy by ID"""
    strategy = await crud_strategy.get_strategy(db, strategy_id)
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy with ID {strategy_id} not found"
        )
    # Check ownership
    if strategy.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this strategy"
        )
    return strategy


@router.put("/{strategy_id}", response_model=StrategyResponse)
async def update_strategy(
    strategy_id: int,
    strategy_update: StrategyUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update an existing strategy"""
    # Check ownership first
    strategy = await crud_strategy.get_strategy(db, strategy_id)
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy with ID {strategy_id} not found"
        )
    if strategy.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this strategy"
        )

    strategy = await crud_strategy.update_strategy(db, strategy_id, strategy_update)
    return strategy


@router.delete("/{strategy_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_strategy(
    strategy_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a strategy"""
    # Check ownership first
    strategy = await crud_strategy.get_strategy(db, strategy_id)
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy with ID {strategy_id} not found"
        )
    if strategy.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this strategy"
        )

    await crud_strategy.delete_strategy(db, strategy_id)
    return None
