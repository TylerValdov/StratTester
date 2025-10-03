from fastapi import APIRouter
from typing import List, Dict, Any
from app.services.indicators import IndicatorService
from app.services.custom_strategy_executor import CustomStrategyExecutor

router = APIRouter()


@router.get("/list", response_model=List[Dict[str, Any]])
async def list_indicators():
    """Get list of available technical indicators"""
    return IndicatorService.get_available_indicators()


@router.get("/templates", response_model=Dict[str, str])
async def get_strategy_templates():
    """Get example custom strategy code"""
    return CustomStrategyExecutor.get_strategy_examples()


@router.get("/template/blank", response_model=Dict[str, str])
async def get_blank_template():
    """Get blank strategy template"""
    return {
        "template": CustomStrategyExecutor.create_strategy_template()
    }


@router.post("/validate")
async def validate_custom_code(code: Dict[str, str]):
    """
    Validate custom Python strategy code.

    Args:
        code: Dict with 'code' key containing Python code string

    Returns:
        Dict with validation result
    """
    strategy_code = code.get('code', '')
    is_valid, error_msg = CustomStrategyExecutor.validate_strategy_code(strategy_code)

    return {
        "valid": is_valid,
        "error": error_msg
    }
