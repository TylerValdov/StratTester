from fastapi import APIRouter
from app.api.v1.endpoints import auth, strategies, backtests, indicators, admin

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(strategies.router, prefix="/strategies", tags=["strategies"])
api_router.include_router(backtests.router, prefix="/backtests", tags=["backtests"])
api_router.include_router(indicators.router, prefix="/indicators", tags=["indicators"])
api_router.include_router(admin.router, prefix="/admin", tags=["admin"])
