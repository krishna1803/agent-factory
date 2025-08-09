"""
Workflows API router.
"""

from typing import List, Optional, Dict, Any
from uuid import UUID
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agent_factory.core.models import WorkflowDefinition, WorkflowState
from agent_factory.orchestration import WorkflowOrchestrator

router = APIRouter()

# Global orchestrator instance (in production, use dependency injection)
orchestrator = WorkflowOrchestrator()


class WorkflowResponse(BaseModel):
    """Response model for workflow operations."""
    success: bool
    message: str
    workflow: Optional[WorkflowDefinition] = None


class WorkflowListResponse(BaseModel):
    """Response model for workflow list operations."""
    success: bool
    message: str
    workflows: List[WorkflowDefinition]
    total: int


class WorkflowExecutionResponse(BaseModel):
    """Response model for workflow execution operations."""
    success: bool
    message: str
    execution_state: Optional[WorkflowState] = None


@router.post("/", response_model=WorkflowResponse)
async def create_workflow(workflow_def: WorkflowDefinition):
    """Create a new workflow."""
    try:
        created_workflow = await orchestrator.create_workflow(workflow_def)
        return WorkflowResponse(
            success=True,
            message=f"Workflow '{created_workflow.name}' created successfully",
            workflow=created_workflow
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/", response_model=WorkflowListResponse)
async def list_workflows(active_only: bool = True):
    """List all workflows."""
    try:
        workflows = orchestrator.list_workflows(active_only=active_only)
        return WorkflowListResponse(
            success=True,
            message="Workflows retrieved successfully",
            workflows=workflows,
            total=len(workflows)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{workflow_id}/execute", response_model=WorkflowExecutionResponse)
async def execute_workflow(workflow_id: UUID, initial_data: Optional[Dict[str, Any]] = None):
    """Execute a workflow."""
    try:
        execution_state = await orchestrator.execute_workflow(workflow_id, initial_data)
        return WorkflowExecutionResponse(
            success=True,
            message="Workflow executed successfully",
            execution_state=execution_state
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{workflow_id}/status/{execution_id}", response_model=WorkflowExecutionResponse)
async def get_workflow_status(workflow_id: UUID, execution_id: UUID):
    """Get workflow execution status."""
    try:
        execution_state = await orchestrator.get_workflow_status(execution_id)
        if not execution_state:
            raise HTTPException(status_code=404, detail="Workflow execution not found")
        
        return WorkflowExecutionResponse(
            success=True,
            message="Workflow status retrieved successfully",
            execution_state=execution_state
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{workflow_id}/stop/{execution_id}")
async def stop_workflow(workflow_id: UUID, execution_id: UUID):
    """Stop a running workflow."""
    try:
        success = await orchestrator.stop_workflow(execution_id)
        if not success:
            raise HTTPException(status_code=404, detail="Workflow execution not found")
        
        return {
            "success": True,
            "message": "Workflow stopped successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/executions/active")
async def list_active_executions():
    """List all active workflow executions."""
    try:
        executions = orchestrator.list_active_executions()
        return {
            "success": True,
            "message": "Active executions retrieved successfully",
            "executions": executions,
            "total": len(executions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
