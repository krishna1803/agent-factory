"""
Interactive API router for deployed workflow interactions.

This router provides endpoints for interacting with deployed agentic workflows
through textual prompts, supporting RAG workflows with reference returns.
"""

from typing import Dict, List, Optional, Any
from uuid import UUID
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from agent_factory.core.models import WorkflowDefinition
from agent_factory.interactive.prompt_engine import (
    interactive_engine, 
    WorkflowInteractionResult, 
    ConversationSession
)

router = APIRouter()


class DeployWorkflowRequest(BaseModel):
    """Request model for deploying a workflow."""
    workflow_definition: WorkflowDefinition
    config: Optional[Dict[str, Any]] = None


class CreateSessionRequest(BaseModel):
    """Request model for creating a conversation session."""
    workflow_id: UUID
    user_id: Optional[str] = None
    session_config: Optional[Dict[str, Any]] = None


class PromptRequest(BaseModel):
    """Request model for processing a prompt."""
    session_id: str
    prompt: str
    include_sources: bool = True


class DeployWorkflowResponse(BaseModel):
    """Response model for workflow deployment."""
    success: bool
    message: str
    workflow_id: UUID
    workflow_info: Optional[Dict[str, Any]] = None


class CreateSessionResponse(BaseModel):
    """Response model for session creation."""
    success: bool
    session_id: str
    workflow_id: UUID
    message: str


class PromptResponse(BaseModel):
    """Response model for prompt processing."""
    success: bool
    response: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    execution_time: Optional[float] = None
    agent_path: List[str]
    timestamp: str


class SessionHistoryResponse(BaseModel):
    """Response model for session history."""
    success: bool
    session_id: str
    history: List[Dict[str, Any]]
    total_interactions: int


class WorkflowInfoResponse(BaseModel):
    """Response model for workflow information."""
    success: bool
    workflow_info: Dict[str, Any]


@router.post("/deploy", response_model=DeployWorkflowResponse)
async def deploy_workflow(request: DeployWorkflowRequest):
    """Deploy a workflow for interactive use."""
    try:
        await interactive_engine.deploy_workflow(
            request.workflow_definition,
            request.config
        )
        
        workflow_info = await interactive_engine.get_workflow_info(
            request.workflow_definition.id
        )
        
        return DeployWorkflowResponse(
            success=True,
            message=f"Workflow '{request.workflow_definition.name}' deployed successfully",
            workflow_id=request.workflow_definition.id,
            workflow_info=workflow_info
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to deploy workflow: {str(e)}"
        )


@router.post("/sessions", response_model=CreateSessionResponse)
async def create_session(request: CreateSessionRequest):
    """Create a new conversation session."""
    try:
        session_id = await interactive_engine.create_session(
            request.workflow_id,
            request.user_id,
            request.session_config
        )
        
        return CreateSessionResponse(
            success=True,
            session_id=session_id,
            workflow_id=request.workflow_id,
            message="Session created successfully"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to create session: {str(e)}"
        )


@router.post("/prompt", response_model=PromptResponse)
async def process_prompt(request: PromptRequest):
    """Process a user prompt through the deployed workflow."""
    try:
        result = await interactive_engine.process_prompt(
            request.session_id,
            request.prompt,
            request.include_sources
        )
        
        return PromptResponse(
            success=True,
            response=result.response,
            sources=result.sources,
            metadata=result.metadata,
            execution_time=result.execution_time,
            agent_path=result.agent_path,
            timestamp=result.timestamp.isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process prompt: {str(e)}"
        )


@router.get("/sessions/{session_id}/history", response_model=SessionHistoryResponse)
async def get_session_history(session_id: str):
    """Get conversation history for a session."""
    try:
        history = await interactive_engine.get_session_history(session_id)
        
        return SessionHistoryResponse(
            success=True,
            session_id=session_id,
            history=history,
            total_interactions=len(history)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {str(e)}"
        )


@router.get("/workflows/{workflow_id}/info", response_model=WorkflowInfoResponse)
async def get_workflow_info(workflow_id: UUID):
    """Get information about a deployed workflow."""
    try:
        workflow_info = await interactive_engine.get_workflow_info(workflow_id)
        
        return WorkflowInfoResponse(
            success=True,
            workflow_info=workflow_info
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Workflow not found: {str(e)}"
        )


@router.delete("/sessions/{session_id}")
async def close_session(session_id: str):
    """Close a conversation session."""
    try:
        await interactive_engine.close_session(session_id)
        return {"success": True, "message": f"Session {session_id} closed"}
        
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Failed to close session: {str(e)}"
        )


@router.post("/sessions/cleanup")
async def cleanup_inactive_sessions(max_age_hours: int = 24):
    """Cleanup inactive sessions."""
    try:
        cleaned_count = await interactive_engine.cleanup_inactive_sessions(max_age_hours)
        return {
            "success": True,
            "message": f"Cleaned up {cleaned_count} inactive sessions"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cleanup sessions: {str(e)}"
        )


# RAG-specific endpoints

class IndexDocumentsRequest(BaseModel):
    """Request model for indexing documents."""
    workflow_id: UUID
    file_paths: List[str]
    pipeline_name: Optional[str] = None


class IndexDocumentsResponse(BaseModel):
    """Response model for document indexing."""
    success: bool
    message: str
    documents_indexed: int
    workflow_id: UUID


@router.post("/rag/index", response_model=IndexDocumentsResponse)
async def index_documents(request: IndexDocumentsRequest):
    """Index documents for a RAG-enabled workflow."""
    try:
        # Determine pipeline name
        pipeline_name = request.pipeline_name
        if not pipeline_name:
            # Use default pattern
            workflow_info = await interactive_engine.get_workflow_info(request.workflow_id)
            # Find first RAG agent
            for agent in workflow_info["agents"]:
                if "rag" in agent["role"].lower():
                    pipeline_name = f"{request.workflow_id}_{agent['name']}"
                    break
        
        if not pipeline_name:
            raise ValueError("No RAG pipeline found for workflow")
        
        from agent_factory.rag.pipeline import rag_manager
        
        indexed_count = await rag_manager.index_documents(
            pipeline_name,
            request.file_paths
        )
        
        return IndexDocumentsResponse(
            success=True,
            message=f"Successfully indexed {indexed_count} document chunks",
            documents_indexed=indexed_count,
            workflow_id=request.workflow_id
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to index documents: {str(e)}"
        )


class RAGQueryRequest(BaseModel):
    """Request model for direct RAG query."""
    pipeline_name: str
    query: str
    max_docs: int = 5


class RAGQueryResponse(BaseModel):
    """Response model for RAG query."""
    success: bool
    query: str
    response: str
    sources: List[Dict[str, Any]]
    context_used: int


@router.post("/rag/query", response_model=RAGQueryResponse)
async def query_rag_pipeline(request: RAGQueryRequest):
    """Directly query a RAG pipeline."""
    try:
        from agent_factory.rag.pipeline import rag_manager
        
        # Find the pipeline
        pipeline = rag_manager.pipelines.get(request.pipeline_name)
        if not pipeline:
            raise ValueError(f"RAG pipeline '{request.pipeline_name}' not found")
        
        result = await pipeline.query(
            request.query,
            max_docs=request.max_docs,
            include_sources=True
        )
        
        return RAGQueryResponse(
            success=True,
            query=result["query"],
            response=result["response"],
            sources=result["sources"],
            context_used=result["context_used"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to query RAG pipeline: {str(e)}"
        )


# Database-specific endpoints

class DatabaseQueryRequest(BaseModel):
    """Request model for database query."""
    connection_name: str
    query: str
    params: Optional[Dict[str, Any]] = None


class DatabaseQueryResponse(BaseModel):
    """Response model for database query."""
    success: bool
    results: List[Dict[str, Any]]
    row_count: int
    execution_time: Optional[float] = None


@router.post("/database/query", response_model=DatabaseQueryResponse)
async def execute_database_query(request: DatabaseQueryRequest):
    """Execute a database query."""
    try:
        import time
        from agent_factory.connections.database import db_manager
        
        start_time = time.time()
        
        results = await db_manager.execute_query(
            request.connection_name,
            request.query,
            request.params
        )
        
        execution_time = time.time() - start_time
        
        return DatabaseQueryResponse(
            success=True,
            results=results,
            row_count=len(results),
            execution_time=execution_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to execute database query: {str(e)}"
        )


@router.get("/database/{connection_name}/health")
async def check_database_health(connection_name: str):
    """Check database connection health."""
    try:
        from agent_factory.connections.database import db_manager
        
        connection = await db_manager.get_connection(connection_name)
        is_healthy = await connection.health_check()
        
        return {
            "success": True,
            "connection_name": connection_name,
            "is_healthy": is_healthy
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to check database health: {str(e)}"
        )
