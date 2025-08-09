"""
Tools API router.
"""

from typing import List, Optional, Dict, Any
from uuid import UUID
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agent_factory.core.models import ToolSpec, ToolType
from agent_factory.agent_builder import ToolManager

router = APIRouter()

# Global tool manager instance (in production, use dependency injection)
tool_manager = ToolManager()


class ToolResponse(BaseModel):
    """Response model for tool operations."""
    success: bool
    message: str
    tool: Optional[ToolSpec] = None


class ToolListResponse(BaseModel):
    """Response model for tool list operations."""
    success: bool
    message: str
    tools: List[ToolSpec]
    total: int


class ToolTestResponse(BaseModel):
    """Response model for tool test operations."""
    success: bool
    message: str
    test_result: Optional[Dict[str, Any]] = None


@router.post("/", response_model=ToolResponse)
async def create_tool(tool_spec: ToolSpec):
    """Register a new tool."""
    try:
        await tool_manager.register_tool(tool_spec)
        return ToolResponse(
            success=True,
            message=f"Tool '{tool_spec.name}' registered successfully",
            tool=tool_spec
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/", response_model=ToolListResponse)
async def list_tools(tool_type: Optional[ToolType] = None, active_only: bool = True):
    """List all tools."""
    try:
        tools = tool_manager.list_tools(tool_type=tool_type, active_only=active_only)
        return ToolListResponse(
            success=True,
            message="Tools retrieved successfully",
            tools=tools,
            total=len(tools)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{tool_id}", response_model=ToolResponse)
async def get_tool(tool_id: UUID):
    """Get a tool by ID."""
    tool = tool_manager.get_tool(tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    
    return ToolResponse(
        success=True,
        message="Tool retrieved successfully",
        tool=tool
    )


@router.get("/by-name/{tool_name}", response_model=ToolResponse)
async def get_tool_by_name(tool_name: str):
    """Get a tool by name."""
    tool = tool_manager.get_tool_by_name(tool_name)
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    
    return ToolResponse(
        success=True,
        message="Tool retrieved successfully",
        tool=tool
    )


@router.put("/{tool_id}", response_model=ToolResponse)
async def update_tool(tool_id: UUID, updated_spec: ToolSpec):
    """Update an existing tool."""
    try:
        await tool_manager.update_tool(tool_id, updated_spec)
        updated_tool = tool_manager.get_tool(tool_id)
        return ToolResponse(
            success=True,
            message=f"Tool '{updated_spec.name}' updated successfully",
            tool=updated_tool
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{tool_id}", response_model=ToolResponse)
async def delete_tool(tool_id: UUID):
    """Unregister a tool."""
    try:
        success = await tool_manager.unregister_tool(tool_id)
        if not success:
            raise HTTPException(status_code=404, detail="Tool not found")
        
        return ToolResponse(
            success=True,
            message="Tool unregistered successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{tool_id}/test", response_model=ToolTestResponse)
async def test_tool(tool_id: UUID, test_input: Dict[str, Any]):
    """Test a tool with given input."""
    try:
        test_result = await tool_manager.test_tool(tool_id, test_input)
        return ToolTestResponse(
            success=True,
            message="Tool test completed successfully",
            test_result=test_result
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
