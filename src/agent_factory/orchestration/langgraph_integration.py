"""
LangGraph integration for workflow orchestration.

This module provides integration with the LangGraph framework for
more advanced workflow orchestration capabilities with support for
OpenAI, Ollama, and OCI GenAI Service.
"""

import logging
import asyncio
from typing import Any, Dict, List, Optional, TypedDict, Annotated, Callable
from uuid import UUID
from datetime import datetime

try:
    from langgraph import StateGraph, CompiledGraph, START, END
    from langgraph.graph import MessagesState
    from langgraph.prebuilt import ToolNode, tools_condition
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    # Fallback for development without LangGraph installed
    StateGraph = Any
    CompiledGraph = Any
    START = "START"
    END = "END"
    MessagesState = Dict
    ToolNode = Any
    tools_condition = Any
    MemorySaver = Any
    LANGGRAPH_AVAILABLE = False

try:
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from langchain_core.tools import BaseTool
    from langchain_openai import ChatOpenAI
    from langchain_community.llms import Ollama
    from langchain_community.chat_models import ChatOllama
    LANGCHAIN_AVAILABLE = True
except ImportError:
    BaseMessage = Any
    HumanMessage = Any
    AIMessage = Any
    SystemMessage = Any
    BaseTool = Any
    ChatOpenAI = Any
    Ollama = Any
    ChatOllama = Any
    LANGCHAIN_AVAILABLE = False

from agent_factory.core.models import WorkflowDefinition, WorkflowState, AgentSpec, ModelProvider
from agent_factory.providers.llm import LLMProviderFactory, BaseLLMProvider
from agent_factory.rag.pipeline import RAGManager, RAGWorkflowAgent
from agent_factory.connections.database import DatabaseManager

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """Enhanced state definition for LangGraph workflows with RAG support."""
    messages: List[Any]  # BaseMessage when available
    current_agent: str
    workflow_data: Dict[str, Any]
    execution_history: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    rag_context: Optional[Dict[str, Any]]
    references: List[Dict[str, Any]]  # For RAG references
    user_query: str
    final_response: Optional[str]


# Type alias for compatibility
GraphState = AgentState


class LLMProvider:
    """Factory for creating LLM instances based on provider."""
    
    @staticmethod
    def create_llm(agent_spec: AgentSpec):
        """Create LLM instance based on agent specification."""
        model_profile = agent_spec.model_profile
        
        if model_profile.provider == ModelProvider.OPENAI:
            return ChatOpenAI(
                model=model_profile.model_name,
                temperature=model_profile.temperature,
                max_tokens=model_profile.max_tokens,
                api_key=model_profile.api_key,
                base_url=model_profile.api_base
            )
        elif model_profile.provider == ModelProvider.LOCAL:  # Ollama
            return ChatOllama(
                model=model_profile.model_name,
                temperature=model_profile.temperature,
                base_url=model_profile.api_base or "http://localhost:11434"
            )
        else:
            raise ValueError(f"Unsupported provider: {model_profile.provider}")


class OCIGenAIProvider:
    """Oracle Cloud Infrastructure Generative AI Service provider."""
    
    def __init__(self, config_file_path: Optional[str] = None, compartment_id: Optional[str] = None):
        """
        Initialize OCI GenAI provider.
        
        Args:
            config_file_path: Path to OCI config file
            compartment_id: OCI compartment ID
        """
        self.config_file_path = config_file_path
        self.compartment_id = compartment_id
        
    def create_llm(self, model_name: str = "cohere.command", **kwargs):
        """Create OCI GenAI LLM instance."""
        try:
            # This would require oci-sdk and proper implementation
            # For now, return a placeholder
            from langchain_community.llms import OCIGenerativeAI
            
            return OCIGenerativeAI(
                model_id=model_name,
                config_file_location=self.config_file_path,
                compartment_id=self.compartment_id,
                **kwargs
            )
        except ImportError:
            logger.warning("OCI SDK not available, using mock implementation")
            return None


class RAGAgent:
    """RAG-enabled agent for document question answering with references."""
    
    def __init__(self, agent_spec: AgentSpec, vector_store=None, retriever=None):
        """
        Initialize RAG agent.
        
        Args:
            agent_spec: Agent specification
            vector_store: Vector store for document retrieval
            retriever: Custom retriever implementation
        """
        self.agent_spec = agent_spec
        self.llm = LLMProvider.create_llm(agent_spec)
        self.vector_store = vector_store
        self.retriever = retriever
        
    async def process(self, state: AgentState) -> AgentState:
        """Process user query with RAG capabilities."""
        user_query = state["user_query"]
        
        # Retrieve relevant documents
        if self.retriever:
            docs = await self._retrieve_documents(user_query)
            state["rag_context"] = {
                "documents": docs,
                "query": user_query
            }
            
            # Extract references
            references = []
            for doc in docs:
                references.append({
                    "content": doc.get("content", ""),
                    "source": doc.get("source", ""),
                    "score": doc.get("score", 0.0),
                    "metadata": doc.get("metadata", {})
                })
            state["references"] = references
        
        # Create context-aware prompt
        context = self._build_context(state.get("rag_context", {}))
        
        # Generate response
        messages = [
            SystemMessage(content=self.agent_spec.instructions),
            SystemMessage(content=f"Context: {context}"),
            HumanMessage(content=user_query)
        ]
        
        response = await self.llm.ainvoke(messages)
        
        # Update state
        state["messages"].append(response)
        state["final_response"] = response.content
        state["execution_history"].append({
            "agent": self.agent_spec.name,
            "timestamp": datetime.utcnow().isoformat(),
            "query": user_query,
            "response": response.content,
            "references_count": len(state.get("references", []))
        })
        
        return state
    
    async def _retrieve_documents(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for the query."""
        if self.retriever:
            # Use custom retriever
            return await self.retriever.retrieve(query)
        elif self.vector_store:
            # Use vector store similarity search
            docs = self.vector_store.similarity_search_with_score(query, k=5)
            return [
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", ""),
                    "score": score,
                    "metadata": doc.metadata
                }
                for doc, score in docs
            ]
        else:
            return []
    
    def _build_context(self, rag_context: Dict[str, Any]) -> str:
        """Build context string from retrieved documents."""
        if not rag_context or not rag_context.get("documents"):
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(rag_context["documents"][:5]):  # Limit to top 5
            context_parts.append(f"Document {i+1}: {doc.get('content', '')}")
        
        return "\n\n".join(context_parts)


class DatabaseAgent:
    """Agent specialized for database operations with Oracle 23ai and Postgres support."""
    
    def __init__(self, agent_spec: AgentSpec, db_connection_spec=None):
        """
        Initialize database agent.
        
        Args:
            agent_spec: Agent specification
            db_connection_spec: Database connection specification
        """
        self.agent_spec = agent_spec
        self.llm = LLMProvider.create_llm(agent_spec)
        self.db_connection = db_connection_spec
        
    async def process(self, state: AgentState) -> AgentState:
        """Process database-related queries."""
        user_query = state["user_query"]
        
        # Generate SQL query if needed
        if "sql" in user_query.lower() or "database" in user_query.lower():
            sql_query = await self._generate_sql_query(user_query)
            
            # Execute query (if safe)
            if sql_query and self._is_safe_query(sql_query):
                result = await self._execute_query(sql_query)
                
                state["execution_history"].append({
                    "agent": self.agent_spec.name,
                    "timestamp": datetime.utcnow().isoformat(),
                    "query": user_query,
                    "sql_generated": sql_query,
                    "result": result
                })
        
        # Generate natural language response
        response = await self._generate_response(user_query, state)
        state["final_response"] = response
        
        return state
    
    async def _generate_sql_query(self, natural_query: str) -> str:
        """Generate SQL query from natural language."""
        prompt = f"""
        Convert this natural language query to SQL:
        Query: {natural_query}
        
        Database schema context: {self._get_schema_context()}
        
        Return only the SQL query, no explanation.
        """
        
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip()
    
    def _is_safe_query(self, sql_query: str) -> bool:
        """Check if SQL query is safe to execute."""
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE']
        return not any(keyword in sql_query.upper() for keyword in dangerous_keywords)
    
    async def _execute_query(self, sql_query: str) -> Dict[str, Any]:
        """Execute SQL query safely."""
        # TODO: Implement actual database execution
        return {"status": "simulated", "query": sql_query}
    
    def _get_schema_context(self) -> str:
        """Get database schema context for SQL generation."""
        # TODO: Implement schema introspection
        return "Schema context would be provided here"
    
    async def _generate_response(self, query: str, state: AgentState) -> str:
        """Generate natural language response."""
        messages = [
            SystemMessage(content=self.agent_spec.instructions),
            HumanMessage(content=query)
        ]
        
        response = await self.llm.ainvoke(messages)
        return response.content


class LangGraphOrchestrator:
    """
    Advanced workflow orchestrator using LangGraph framework.
    
    This provides more sophisticated workflow capabilities including:
    - Complex branching and conditional logic
    - Parallel execution paths
    - Advanced state management
    - Built-in tool integration
    """
    
    def __init__(self):
        """Initialize the LangGraph orchestrator."""
        self._graphs: Dict[UUID, Any] = {}  # Will store CompiledGraph instances
        self._workflows: Dict[UUID, WorkflowDefinition] = {}
        
    async def create_langgraph_workflow(self, workflow_def: WorkflowDefinition) -> Any:
        """
        Create a LangGraph workflow from definition.
        
        Args:
            workflow_def: The workflow definition
            
        Returns:
            The compiled LangGraph workflow
        """
        # TODO: Implement actual LangGraph workflow creation
        # This is a placeholder implementation
        
        logger.info(f"Creating LangGraph workflow: {workflow_def.name}")
        
        # For now, return a mock compiled graph
        compiled_graph = self._create_mock_graph(workflow_def)
        
        self._graphs[workflow_def.id] = compiled_graph
        self._workflows[workflow_def.id] = workflow_def
        
        return compiled_graph
    
    async def execute_langgraph_workflow(
        self, 
        workflow_id: UUID, 
        initial_state: GraphState
    ) -> GraphState:
        """
        Execute a LangGraph workflow.
        
        Args:
            workflow_id: The workflow ID
            initial_state: Initial graph state
            
        Returns:
            Final graph state
        """
        graph = self._graphs.get(workflow_id)
        if not graph:
            raise ValueError(f"Workflow {workflow_id} not found")
            
        workflow_def = self._workflows[workflow_id]
        logger.info(f"Executing LangGraph workflow: {workflow_def.name}")
        
        # TODO: Execute actual LangGraph workflow
        # For now, simulate execution
        final_state = await self._simulate_graph_execution(initial_state, workflow_def)
        
        return final_state
    
    def _create_mock_graph(self, workflow_def: WorkflowDefinition) -> Dict[str, Any]:
        """
        Create a mock graph for demonstration purposes.
        
        Args:
            workflow_def: The workflow definition
            
        Returns:
            Mock compiled graph
        """
        return {
            "id": str(workflow_def.id),
            "name": workflow_def.name,
            "nodes": [agent.name for agent in workflow_def.agent_specs],
            "entry_point": workflow_def.entry_point,
            "created_at": workflow_def.created_at.isoformat()
        }
    
    async def _simulate_graph_execution(
        self, 
        state: GraphState, 
        workflow_def: WorkflowDefinition
    ) -> GraphState:
        """
        Simulate graph execution for demonstration.
        
        Args:
            state: Current graph state
            workflow_def: Workflow definition
            
        Returns:
            Updated graph state
        """
        # Simulate execution through each agent
        for agent in workflow_def.agent_specs:
            # Simulate agent execution
            result = {
                "agent": agent.name,
                "role": agent.role,
                "timestamp": "2024-01-01T00:00:00Z",
                "output": f"Processed by {agent.name}",
                "status": "completed"
            }
            
            state["execution_history"].append(result)
            state["current_agent"] = agent.name
            state["messages"].append({
                "type": "agent_result",
                "content": result["output"],
                "agent": agent.name
            })
        
        # Update metadata
        state["metadata"]["status"] = "completed"
        state["metadata"]["total_agents"] = len(workflow_def.agent_specs)
        
        return state


class ConditionalRouter:
    """Router for conditional workflow paths."""
    
    def __init__(self, conditions: Dict[str, callable]):
        """
        Initialize the router with conditions.
        
        Args:
            conditions: Mapping of condition names to callable functions
        """
        self.conditions = conditions
    
    def route(self, state: GraphState, condition_name: str) -> str:
        """
        Route based on condition evaluation.
        
        Args:
            state: Current graph state
            condition_name: Name of condition to evaluate
            
        Returns:
            Next node name
        """
        condition_func = self.conditions.get(condition_name)
        if condition_func:
            try:
                return condition_func(state)
            except Exception as e:
                logger.error(f"Condition evaluation failed: {str(e)}")
                return "error"
        
        return "default"


class ParallelExecutor:
    """Executor for parallel workflow branches."""
    
    def __init__(self):
        """Initialize the parallel executor."""
        self.active_branches: Dict[str, Any] = {}
    
    async def execute_parallel_branches(
        self, 
        branches: List[str], 
        state: GraphState
    ) -> GraphState:
        """
        Execute multiple branches in parallel.
        
        Args:
            branches: List of branch names to execute
            state: Current graph state
            
        Returns:
            Merged graph state from all branches
        """
        # TODO: Implement actual parallel execution
        # For now, simulate sequential execution
        
        for branch in branches:
            logger.info(f"Executing branch: {branch}")
            # Simulate branch execution
            branch_result = {
                "branch": branch,
                "status": "completed",
                "output": f"Result from {branch}"
            }
            
            state["execution_history"].append(branch_result)
        
        return state


class ToolIntegration:
    """Integration layer for tools in LangGraph workflows."""
    
    def __init__(self):
        """Initialize tool integration."""
        self.tool_registry: Dict[str, Any] = {}
    
    def register_tool_node(self, tool_name: str, tool_spec: Any) -> None:
        """
        Register a tool for use in workflows.
        
        Args:
            tool_name: Name of the tool
            tool_spec: Tool specification
        """
        self.tool_registry[tool_name] = tool_spec
        logger.info(f"Registered tool node: {tool_name}")
    
    async def execute_tool(self, tool_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool with given input.
        
        Args:
            tool_name: Name of the tool to execute
            input_data: Input data for the tool
            
        Returns:
            Tool execution result
        """
        tool_spec = self.tool_registry.get(tool_name)
        if not tool_spec:
            raise ValueError(f"Tool {tool_name} not found")
        
        # TODO: Implement actual tool execution
        # For now, return mock result
        return {
            "tool": tool_name,
            "input": input_data,
            "output": f"Mock output from {tool_name}",
            "status": "success"
        }
