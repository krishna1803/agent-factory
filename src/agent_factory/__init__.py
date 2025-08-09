"""
Agent Factory - A comprehensive framework for creating, orchestrating, and managing AI agents.

This package provides:
- Agent Builder: Create agents with defined specifications
- Orchestration Runtime: LangGraph/LangChain-based workflow orchestration  
- UI Component: StreamLit-based visualization
- API Component: FastAPI-based REST API
"""

__version__ = "0.1.0"
__author__ = "Agent Factory Team"
__email__ = "team@agentfactory.com"

from agent_factory.core.models import AgentSpec, ToolSpec, ConnectionSpec
from agent_factory.agent_builder.builder import AgentBuilder
from agent_factory.orchestration.workflow import WorkflowOrchestrator

__all__ = [
    "AgentSpec",
    "ToolSpec", 
    "ConnectionSpec",
    "AgentBuilder",
    "WorkflowOrchestrator",
]
