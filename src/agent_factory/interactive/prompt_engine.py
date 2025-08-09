"""
Interactive prompt facility for deployed agentic workflows.

This module provides a comprehensive interface for interacting with
deployed workflows through textual prompts, supporting RAG workflows
with reference returns and real-time conversation management.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from agent_factory.core.models import WorkflowDefinition, AgentSpec
from agent_factory.orchestration.langgraph_integration import LangGraphOrchestrator, AgentState
from agent_factory.rag.pipeline import RAGManager, RAGWorkflowAgent
from agent_factory.providers.llm import LLMProviderManager
from agent_factory.connections.database import DatabaseManager

logger = logging.getLogger(__name__)


class ConversationSession:
    """Manages a conversation session with a workflow."""
    
    def __init__(
        self, 
        session_id: str, 
        workflow_id: UUID, 
        user_id: Optional[str] = None
    ):
        """Initialize conversation session."""
        self.session_id = session_id
        self.workflow_id = workflow_id
        self.user_id = user_id
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.conversation_history: List[Dict[str, Any]] = []
        self.context: Dict[str, Any] = {}
        self.is_active = True
    
    def add_interaction(
        self, 
        user_input: str, 
        agent_response: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add an interaction to the conversation history."""
        interaction = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_input": user_input,
            "agent_response": agent_response,
            "metadata": metadata or {}
        }
        
        self.conversation_history.append(interaction)
        self.last_activity = datetime.utcnow()
    
    def get_conversation_context(self, max_turns: int = 10) -> str:
        """Get conversation context as a formatted string."""
        recent_history = self.conversation_history[-max_turns:]
        context_parts = []
        
        for interaction in recent_history:
            context_parts.append(f"User: {interaction['user_input']}")
            context_parts.append(f"Assistant: {interaction['agent_response']}")
        
        return "\n".join(context_parts)
    
    def update_context(self, key: str, value: Any) -> None:
        """Update session context."""
        self.context[key] = value
        self.last_activity = datetime.utcnow()


class WorkflowInteractionResult:
    """Result of a workflow interaction."""
    
    def __init__(
        self,
        response: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        execution_time: Optional[float] = None,
        agent_path: Optional[List[str]] = None
    ):
        """Initialize interaction result."""
        self.response = response
        self.sources = sources or []
        self.metadata = metadata or {}
        self.execution_time = execution_time
        self.agent_path = agent_path or []
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "response": self.response,
            "sources": self.sources,
            "metadata": self.metadata,
            "execution_time": self.execution_time,
            "agent_path": self.agent_path,
            "timestamp": self.timestamp.isoformat()
        }


class InteractiveWorkflowEngine:
    """Engine for interactive workflow execution."""
    
    def __init__(self):
        """Initialize the interactive workflow engine."""
        self.orchestrator = LangGraphOrchestrator()
        self.rag_manager = RAGManager()
        self.llm_manager = LLMProviderManager()
        self.db_manager = DatabaseManager()
        
        # Session management
        self.active_sessions: Dict[str, ConversationSession] = {}
        self.workflow_configs: Dict[UUID, Dict[str, Any]] = {}
    
    async def deploy_workflow(
        self, 
        workflow_def: WorkflowDefinition,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Deploy a workflow for interactive use."""
        try:
            # Create LangGraph workflow
            compiled_graph = await self.orchestrator.create_langgraph_workflow(workflow_def)
            
            # Store workflow configuration
            self.workflow_configs[workflow_def.id] = {
                "definition": workflow_def,
                "compiled_graph": compiled_graph,
                "config": config or {},
                "deployed_at": datetime.utcnow(),
                "is_rag_enabled": self._is_rag_enabled(workflow_def),
                "is_db_enabled": self._is_db_enabled(workflow_def)
            }
            
            # Initialize RAG if needed
            if self._is_rag_enabled(workflow_def):
                await self._setup_rag_for_workflow(workflow_def)
            
            # Setup database connections if needed
            if self._is_db_enabled(workflow_def):
                await self._setup_db_for_workflow(workflow_def)
            
            logger.info(f"Deployed workflow: {workflow_def.name} ({workflow_def.id})")
            
        except Exception as e:
            logger.error(f"Failed to deploy workflow {workflow_def.name}: {str(e)}")
            raise
    
    async def create_session(
        self, 
        workflow_id: UUID, 
        user_id: Optional[str] = None,
        session_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new conversation session."""
        if workflow_id not in self.workflow_configs:
            raise ValueError(f"Workflow {workflow_id} not deployed")
        
        session_id = str(uuid4())
        session = ConversationSession(session_id, workflow_id, user_id)
        
        # Apply session configuration
        if session_config:
            session.context.update(session_config)
        
        self.active_sessions[session_id] = session
        
        logger.info(f"Created session {session_id} for workflow {workflow_id}")
        return session_id
    
    async def process_prompt(
        self, 
        session_id: str, 
        prompt: str,
        include_sources: bool = True
    ) -> WorkflowInteractionResult:
        """Process a user prompt through the workflow."""
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        workflow_config = self.workflow_configs.get(session.workflow_id)
        if not workflow_config:
            raise ValueError(f"Workflow {session.workflow_id} not found")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Prepare initial state
            initial_state = self._prepare_initial_state(session, prompt)
            
            # Execute workflow
            if workflow_config["is_rag_enabled"]:
                result = await self._execute_rag_workflow(
                    session, prompt, workflow_config, include_sources
                )
            else:
                result = await self._execute_standard_workflow(
                    session, prompt, workflow_config
                )
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Create interaction result
            interaction_result = WorkflowInteractionResult(
                response=result["response"],
                sources=result.get("sources", []) if include_sources else [],
                metadata=result.get("metadata", {}),
                execution_time=execution_time,
                agent_path=result.get("agent_path", [])
            )
            
            # Update session
            session.add_interaction(
                prompt, 
                result["response"], 
                {
                    "execution_time": execution_time,
                    "sources_count": len(result.get("sources", [])),
                    "agent_path": result.get("agent_path", [])
                }
            )
            
            return interaction_result
            
        except Exception as e:
            logger.error(f"Error processing prompt in session {session_id}: {str(e)}")
            
            # Create error result
            error_result = WorkflowInteractionResult(
                response=f"I'm sorry, I encountered an error processing your request: {str(e)}",
                metadata={"error": True, "error_message": str(e)},
                execution_time=asyncio.get_event_loop().time() - start_time
            )
            
            # Update session with error
            session.add_interaction(prompt, error_result.response, {"error": True})
            
            return error_result
    
    async def _execute_rag_workflow(
        self, 
        session: ConversationSession, 
        prompt: str,
        workflow_config: Dict[str, Any],
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """Execute a RAG-enabled workflow."""
        workflow_def = workflow_config["definition"]
        
        # Find RAG agent
        rag_agent = None
        for agent_spec in workflow_def.agent_specs:
            if "rag" in agent_spec.role.lower() or "retrieval" in agent_spec.role.lower():
                agent_name = f"{workflow_def.id}_{agent_spec.name}_rag"
                rag_agent = self.rag_manager.agents.get(agent_name)
                break
        
        if not rag_agent:
            raise ValueError("RAG agent not found for RAG-enabled workflow")
        
        # Process with RAG
        rag_result = await rag_agent.process_interactive_query(prompt, session.session_id)
        
        return {
            "response": rag_result["response"],
            "sources": rag_result.get("sources", []) if include_sources else [],
            "metadata": {
                "rag_enabled": True,
                "context_used": rag_result.get("context_used", 0),
                "conversation_id": rag_result.get("conversation_id")
            },
            "agent_path": ["rag_agent"]
        }
    
    async def _execute_standard_workflow(
        self, 
        session: ConversationSession, 
        prompt: str,
        workflow_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a standard (non-RAG) workflow."""
        # Prepare state with conversation context
        initial_state = self._prepare_initial_state(session, prompt)
        
        # Execute through LangGraph
        final_state = await self.orchestrator.execute_langgraph_workflow(
            session.workflow_id, initial_state
        )
        
        return {
            "response": final_state.get("final_response", "No response generated"),
            "metadata": final_state.get("metadata", {}),
            "agent_path": [entry.get("agent", "unknown") for entry in final_state.get("execution_history", [])]
        }
    
    def _prepare_initial_state(self, session: ConversationSession, prompt: str) -> AgentState:
        """Prepare initial state for workflow execution."""
        return {
            "messages": [{"type": "human", "content": prompt}],
            "current_agent": "",
            "workflow_data": session.context.copy(),
            "execution_history": [],
            "metadata": {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "conversation_context": session.get_conversation_context()
            },
            "rag_context": None,
            "references": [],
            "user_query": prompt,
            "final_response": None
        }
    
    def _is_rag_enabled(self, workflow_def: WorkflowDefinition) -> bool:
        """Check if workflow has RAG capabilities."""
        for agent in workflow_def.agent_specs:
            if "rag" in agent.role.lower() or "retrieval" in agent.role.lower():
                return True
        return False
    
    def _is_db_enabled(self, workflow_def: WorkflowDefinition) -> bool:
        """Check if workflow has database capabilities."""
        for agent in workflow_def.agent_specs:
            if any(conn.connection_type.value == "database" for conn in agent.connection_specs):
                return True
        return False
    
    async def _setup_rag_for_workflow(self, workflow_def: WorkflowDefinition) -> None:
        """Setup RAG components for a workflow."""
        try:
            # Find RAG-enabled agents
            for agent_spec in workflow_def.agent_specs:
                if "rag" in agent_spec.role.lower() or "retrieval" in agent_spec.role.lower():
                    # Create RAG pipeline
                    pipeline_name = f"{workflow_def.id}_{agent_spec.name}"
                    pipeline = await self.rag_manager.create_rag_pipeline(
                        pipeline_name, agent_spec
                    )
                    
                    # Create RAG agent
                    agent_name = f"{pipeline_name}_rag"
                    await self.rag_manager.create_rag_agent(
                        agent_name, pipeline_name, agent_spec
                    )
                    
                    logger.info(f"Setup RAG for agent: {agent_spec.name}")
        except Exception as e:
            logger.error(f"Failed to setup RAG for workflow: {str(e)}")
            raise
    
    async def _setup_db_for_workflow(self, workflow_def: WorkflowDefinition) -> None:
        """Setup database connections for a workflow."""
        try:
            for agent_spec in workflow_def.agent_specs:
                for conn_spec in agent_spec.connection_specs:
                    if conn_spec.connection_type.value == "database":
                        conn_name = f"{workflow_def.id}_{agent_spec.name}_{conn_spec.name}"
                        self.db_manager.register_connection(conn_name, conn_spec)
                        logger.info(f"Setup database connection: {conn_name}")
        except Exception as e:
            logger.error(f"Failed to setup database connections: {str(e)}")
            raise
    
    async def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session."""
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        return session.conversation_history
    
    async def get_workflow_info(self, workflow_id: UUID) -> Dict[str, Any]:
        """Get information about a deployed workflow."""
        workflow_config = self.workflow_configs.get(workflow_id)
        if not workflow_config:
            raise ValueError(f"Workflow {workflow_id} not deployed")
        
        workflow_def = workflow_config["definition"]
        
        return {
            "id": str(workflow_id),
            "name": workflow_def.name,
            "description": workflow_def.description,
            "agents": [{"name": agent.name, "role": agent.role} for agent in workflow_def.agent_specs],
            "rag_enabled": workflow_config["is_rag_enabled"],
            "db_enabled": workflow_config["is_db_enabled"],
            "deployed_at": workflow_config["deployed_at"].isoformat(),
            "active_sessions": len([s for s in self.active_sessions.values() if s.workflow_id == workflow_id])
        }
    
    async def close_session(self, session_id: str) -> None:
        """Close a conversation session."""
        session = self.active_sessions.get(session_id)
        if session:
            session.is_active = False
            del self.active_sessions[session_id]
            logger.info(f"Closed session {session_id}")
    
    async def cleanup_inactive_sessions(self, max_age_hours: int = 24) -> int:
        """Cleanup inactive sessions."""
        from datetime import timedelta
        
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        inactive_sessions = [
            session_id for session_id, session in self.active_sessions.items()
            if session.last_activity < cutoff_time
        ]
        
        for session_id in inactive_sessions:
            await self.close_session(session_id)
        
        logger.info(f"Cleaned up {len(inactive_sessions)} inactive sessions")
        return len(inactive_sessions)


class PromptTemplateManager:
    """Manager for prompt templates and conversation patterns."""
    
    def __init__(self):
        """Initialize prompt template manager."""
        self.templates: Dict[str, str] = {}
        self.conversation_patterns: Dict[str, Dict[str, Any]] = {}
    
    def register_template(self, name: str, template: str) -> None:
        """Register a prompt template."""
        self.templates[name] = template
    
    def format_template(self, name: str, **kwargs) -> str:
        """Format a template with provided variables."""
        template = self.templates.get(name)
        if not template:
            raise ValueError(f"Template '{name}' not found")
        
        return template.format(**kwargs)
    
    def register_conversation_pattern(self, name: str, pattern: Dict[str, Any]) -> None:
        """Register a conversation pattern."""
        self.conversation_patterns[name] = pattern
    
    def get_conversation_pattern(self, name: str) -> Dict[str, Any]:
        """Get a conversation pattern."""
        return self.conversation_patterns.get(name, {})


# Global instances
interactive_engine = InteractiveWorkflowEngine()
prompt_manager = PromptTemplateManager()

# Register default templates
prompt_manager.register_template(
    "rag_context",
    """Based on the following context, please answer the user's question:

Context:
{context}

Question: {question}

Please provide a clear and accurate answer based on the context provided. If the context doesn't contain enough information to answer the question, please say so."""
)

prompt_manager.register_template(
    "conversation_context",
    """Previous conversation:
{conversation_history}

Current question: {current_question}

Please respond considering the previous conversation context."""
)
