"""
Agents API router.
"""

from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from agent_factory.core.models import AgentSpec
from agent_factory.agent_builder import AgentBuilder

router = APIRouter()

# Global agent builder instance (in production, use dependency injection)
agent_builder = AgentBuilder()


class AgentResponse(BaseModel):
    """Response model for agent operations."""
    success: bool
    message: str
    agent: Optional[AgentSpec] = None


class AgentListResponse(BaseModel):
    """Response model for agent list operations."""
    success: bool
    message: str
    agents: List[AgentSpec]
    total: int


@router.post("/", response_model=AgentResponse)
async def create_agent(agent_spec: AgentSpec):
    """Create a new agent."""
    try:
        created_agent = await agent_builder.create_agent(agent_spec)
        return AgentResponse(
            success=True,
            message=f"Agent '{created_agent.name}' created successfully",
            agent=created_agent
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/", response_model=AgentListResponse)
async def list_agents(active_only: bool = True):
    """List all agents."""
    try:
        agents = agent_builder.list_agents(active_only=active_only)
        return AgentListResponse(
            success=True,
            message="Agents retrieved successfully",
            agents=agents,
            total=len(agents)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: UUID):
    """Get an agent by ID."""
    agent = agent_builder.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return AgentResponse(
        success=True,
        message="Agent retrieved successfully",
        agent=agent
    )


@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent(agent_id: UUID, updated_spec: AgentSpec):
    """Update an existing agent."""
    try:
        updated_agent = await agent_builder.update_agent(agent_id, updated_spec)
        return AgentResponse(
            success=True,
            message=f"Agent '{updated_agent.name}' updated successfully",
            agent=updated_agent
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{agent_id}", response_model=AgentResponse)
async def delete_agent(agent_id: UUID):
    """Delete an agent."""
    try:
        success = await agent_builder.delete_agent(agent_id)
        if not success:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return AgentResponse(
            success=True,
            message="Agent deleted successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
