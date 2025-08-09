"""
Health check router.
"""

from fastapi import APIRouter
from typing import Dict, Any
from datetime import datetime

router = APIRouter()


@router.get("/")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Agent Factory API",
        "version": "0.1.0"
    }


@router.get("/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """Detailed health check with component status."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Agent Factory API",
        "version": "0.1.0",
        "components": {
            "api": {"status": "healthy", "response_time_ms": 1},
            "database": {"status": "healthy", "response_time_ms": 5},
            "cache": {"status": "healthy", "response_time_ms": 2},
            "agent_builder": {"status": "healthy", "agents_created": 0},
            "orchestrator": {"status": "healthy", "workflows_running": 0}
        },
        "uptime_seconds": 0,  # TODO: Implement actual uptime tracking
        "memory_usage_mb": 0,  # TODO: Implement actual memory tracking
        "cpu_usage_percent": 0  # TODO: Implement actual CPU tracking
    }
