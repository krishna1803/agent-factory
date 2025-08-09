"""
Connections API router.
"""

from typing import List, Optional, Dict, Any
from uuid import UUID
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agent_factory.core.models import ConnectionSpec, ConnectionType
from agent_factory.agent_builder import ConnectionManager

router = APIRouter()

# Global connection manager instance (in production, use dependency injection)
connection_manager = ConnectionManager()


class ConnectionResponse(BaseModel):
    """Response model for connection operations."""
    success: bool
    message: str
    connection: Optional[ConnectionSpec] = None


class ConnectionListResponse(BaseModel):
    """Response model for connection list operations."""
    success: bool
    message: str
    connections: List[ConnectionSpec]
    total: int


class ConnectionTestResponse(BaseModel):
    """Response model for connection test operations."""
    success: bool
    message: str
    test_result: Optional[Dict[str, Any]] = None


@router.post("/", response_model=ConnectionResponse)
async def create_connection(connection_spec: ConnectionSpec):
    """Register a new connection."""
    try:
        await connection_manager.register_connection(connection_spec)
        return ConnectionResponse(
            success=True,
            message=f"Connection '{connection_spec.name}' registered successfully",
            connection=connection_spec
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/", response_model=ConnectionListResponse)
async def list_connections(connection_type: Optional[ConnectionType] = None, active_only: bool = True):
    """List all connections."""
    try:
        connections = connection_manager.list_connections(connection_type=connection_type, active_only=active_only)
        return ConnectionListResponse(
            success=True,
            message="Connections retrieved successfully",
            connections=connections,
            total=len(connections)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{connection_id}", response_model=ConnectionResponse)
async def get_connection(connection_id: UUID):
    """Get a connection by ID."""
    connection = connection_manager.get_connection(connection_id)
    if not connection:
        raise HTTPException(status_code=404, detail="Connection not found")
    
    return ConnectionResponse(
        success=True,
        message="Connection retrieved successfully",
        connection=connection
    )


@router.get("/by-name/{connection_name}", response_model=ConnectionResponse)
async def get_connection_by_name(connection_name: str):
    """Get a connection by name."""
    connection = connection_manager.get_connection_by_name(connection_name)
    if not connection:
        raise HTTPException(status_code=404, detail="Connection not found")
    
    return ConnectionResponse(
        success=True,
        message="Connection retrieved successfully",
        connection=connection
    )


@router.put("/{connection_id}", response_model=ConnectionResponse)
async def update_connection(connection_id: UUID, updated_spec: ConnectionSpec):
    """Update an existing connection."""
    try:
        await connection_manager.update_connection(connection_id, updated_spec)
        updated_connection = connection_manager.get_connection(connection_id)
        return ConnectionResponse(
            success=True,
            message=f"Connection '{updated_spec.name}' updated successfully",
            connection=updated_connection
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{connection_id}", response_model=ConnectionResponse)
async def delete_connection(connection_id: UUID):
    """Unregister a connection."""
    try:
        success = await connection_manager.unregister_connection(connection_id)
        if not success:
            raise HTTPException(status_code=404, detail="Connection not found")
        
        return ConnectionResponse(
            success=True,
            message="Connection unregistered successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{connection_id}/test", response_model=ConnectionTestResponse)
async def test_connection(connection_id: UUID):
    """Test a connection."""
    try:
        test_result = await connection_manager.test_connection(connection_id)
        return ConnectionTestResponse(
            success=True,
            message="Connection test completed successfully",
            test_result=test_result
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{connection_id}/health")
async def get_connection_health(connection_id: UUID):
    """Get connection health status."""
    try:
        health_status = connection_manager.get_connection_health(connection_id)
        if health_status is None:
            raise HTTPException(status_code=404, detail="Connection not found")
        
        return {
            "success": True,
            "message": "Connection health retrieved successfully",
            "connection_id": str(connection_id),
            "healthy": health_status,
            "status": "healthy" if health_status else "unhealthy"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
