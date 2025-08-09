"""
Workflow orchestration using LangGraph framework.

This module provides the WorkflowOrchestrator class which manages
the execution of agent workflows using LangGraph.
"""

import logging
from typing import Any, Dict, List, Optional, Callable
from uuid import UUID, uuid4
from datetime import datetime

from agent_factory.core.models import WorkflowDefinition, WorkflowState, AgentSpec

logger = logging.getLogger(__name__)


class WorkflowNode:
    """Represents a node in the workflow graph."""
    
    def __init__(self, node_id: str, agent_spec: AgentSpec, node_type: str = "agent"):
        """
        Initialize a workflow node.
        
        Args:
            node_id: Unique identifier for the node
            agent_spec: The agent specification for this node
            node_type: Type of node (agent, condition, start, end)
        """
        self.node_id = node_id
        self.agent_spec = agent_spec
        self.node_type = node_type
        self.execution_count = 0
        self.last_execution = None
        
    async def execute(self, state: WorkflowState) -> Dict[str, Any]:
        """
        Execute the node with the given state.
        
        Args:
            state: Current workflow state
            
        Returns:
            Execution result
        """
        self.execution_count += 1
        self.last_execution = datetime.utcnow()
        
        try:
            # TODO: Implement actual agent execution
            # For now, return a mock result
            result = {
                "node_id": self.node_id,
                "agent_name": self.agent_spec.name,
                "status": "completed",
                "output": f"Output from {self.agent_spec.name}",
                "execution_time": datetime.utcnow().isoformat(),
                "execution_count": self.execution_count
            }
            
            # Update state history
            state.history.append({
                "node_id": self.node_id,
                "timestamp": datetime.utcnow().isoformat(),
                "result": result
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Node execution failed for {self.node_id}: {str(e)}")
            raise


class WorkflowEdge:
    """Represents an edge between workflow nodes."""
    
    def __init__(self, from_node: str, to_node: str, condition: Optional[Callable] = None):
        """
        Initialize a workflow edge.
        
        Args:
            from_node: Source node ID
            to_node: Target node ID  
            condition: Optional condition function for conditional edges
        """
        self.from_node = from_node
        self.to_node = to_node
        self.condition = condition
        self.traversal_count = 0
        
    def should_traverse(self, state: WorkflowState) -> bool:
        """
        Check if this edge should be traversed.
        
        Args:
            state: Current workflow state
            
        Returns:
            True if edge should be traversed
        """
        if self.condition is None:
            return True
            
        try:
            return self.condition(state)
        except Exception as e:
            logger.warning(f"Edge condition evaluation failed: {str(e)}")
            return False


class WorkflowOrchestrator:
    """
    Main orchestrator for managing agent workflows using LangGraph concepts.
    
    This class coordinates the execution of multi-agent workflows by:
    1. Managing workflow definitions and state
    2. Orchestrating agent execution order
    3. Handling workflow branching and conditions
    4. Monitoring execution progress
    """
    
    def __init__(self):
        """Initialize the workflow orchestrator."""
        self._workflows: Dict[UUID, WorkflowDefinition] = {}
        self._active_executions: Dict[UUID, WorkflowState] = {}
        self._nodes: Dict[str, WorkflowNode] = {}
        self._edges: Dict[str, List[WorkflowEdge]] = {}
        
    async def create_workflow(self, workflow_def: WorkflowDefinition) -> WorkflowDefinition:
        """
        Create a new workflow from definition.
        
        Args:
            workflow_def: The workflow definition
            
        Returns:
            The created workflow definition
            
        Raises:
            ValueError: If workflow definition is invalid
        """
        # Validate workflow definition
        await self._validate_workflow_definition(workflow_def)
        
        # Build workflow graph
        await self._build_workflow_graph(workflow_def)
        
        # Store workflow
        self._workflows[workflow_def.id] = workflow_def
        
        logger.info(f"Created workflow: {workflow_def.name} (ID: {workflow_def.id})")
        return workflow_def
    
    async def execute_workflow(self, workflow_id: UUID, initial_data: Dict[str, Any] = None) -> WorkflowState:
        """
        Execute a workflow.
        
        Args:
            workflow_id: The ID of the workflow to execute
            initial_data: Initial data for the workflow
            
        Returns:
            The final workflow state
            
        Raises:
            ValueError: If workflow doesn't exist
            RuntimeError: If execution fails
        """
        workflow_def = self._workflows.get(workflow_id)
        if not workflow_def:
            raise ValueError(f"Workflow with ID {workflow_id} not found")
            
        # Create execution state
        execution_state = WorkflowState(
            workflow_id=workflow_id,
            current_step=workflow_def.entry_point,
            data=initial_data or {},
            context={"workflow_name": workflow_def.name},
            history=[]
        )
        
        # Track active execution
        self._active_executions[execution_state.id] = execution_state
        
        try:
            # Execute workflow
            final_state = await self._execute_workflow_steps(execution_state)
            
            logger.info(f"Workflow execution completed: {workflow_def.name}")
            return final_state
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            raise RuntimeError(f"Workflow execution failed: {str(e)}") from e
            
        finally:
            # Clean up active execution
            if execution_state.id in self._active_executions:
                del self._active_executions[execution_state.id]
    
    async def get_workflow_status(self, execution_id: UUID) -> Optional[WorkflowState]:
        """
        Get the current status of a workflow execution.
        
        Args:
            execution_id: The execution ID
            
        Returns:
            The current workflow state if found
        """
        return self._active_executions.get(execution_id)
    
    async def stop_workflow(self, execution_id: UUID) -> bool:
        """
        Stop a running workflow execution.
        
        Args:
            execution_id: The execution ID to stop
            
        Returns:
            True if stopped successfully
        """
        if execution_id in self._active_executions:
            state = self._active_executions[execution_id]
            state.context["status"] = "stopped"
            state.context["stop_time"] = datetime.utcnow().isoformat()
            
            del self._active_executions[execution_id]
            logger.info(f"Stopped workflow execution: {execution_id}")
            return True
            
        return False
    
    def list_workflows(self, active_only: bool = True) -> List[WorkflowDefinition]:
        """
        List all workflows.
        
        Args:
            active_only: If True, only return active workflows
            
        Returns:
            List of workflow definitions
        """
        workflows = list(self._workflows.values())
        if active_only:
            workflows = [wf for wf in workflows if wf.is_active]
        return workflows
    
    def list_active_executions(self) -> List[WorkflowState]:
        """
        List all active workflow executions.
        
        Returns:
            List of active workflow states
        """
        return list(self._active_executions.values())
    
    async def _validate_workflow_definition(self, workflow_def: WorkflowDefinition) -> None:
        """
        Validate a workflow definition.
        
        Args:
            workflow_def: The workflow definition to validate
            
        Raises:
            ValueError: If validation fails
        """
        if not workflow_def.name or not workflow_def.name.strip():
            raise ValueError("Workflow name cannot be empty")
            
        if not workflow_def.entry_point:
            raise ValueError("Workflow must have an entry point")
            
        if not workflow_def.agent_specs:
            raise ValueError("Workflow must have at least one agent")
            
        # Validate workflow graph structure
        if not workflow_def.workflow_graph:
            raise ValueError("Workflow must have a graph definition")
            
        # Check that entry point exists in graph
        nodes = workflow_def.workflow_graph.get("nodes", {})
        if workflow_def.entry_point not in nodes:
            raise ValueError(f"Entry point '{workflow_def.entry_point}' not found in workflow graph")
    
    async def _build_workflow_graph(self, workflow_def: WorkflowDefinition) -> None:
        """
        Build the internal workflow graph from definition.
        
        Args:
            workflow_def: The workflow definition
        """
        # Create nodes from agents
        agent_map = {agent.name: agent for agent in workflow_def.agent_specs}
        
        nodes = workflow_def.workflow_graph.get("nodes", {})
        for node_id, node_config in nodes.items():
            agent_name = node_config.get("agent")
            if agent_name and agent_name in agent_map:
                agent_spec = agent_map[agent_name]
                node = WorkflowNode(node_id, agent_spec, node_config.get("type", "agent"))
                self._nodes[f"{workflow_def.id}:{node_id}"] = node
        
        # Create edges
        edges = workflow_def.workflow_graph.get("edges", [])
        for edge_config in edges:
            from_node = edge_config.get("from")
            to_node = edge_config.get("to")
            
            if from_node and to_node:
                edge = WorkflowEdge(from_node, to_node)
                edge_key = f"{workflow_def.id}:{from_node}"
                
                if edge_key not in self._edges:
                    self._edges[edge_key] = []
                self._edges[edge_key].append(edge)
    
    async def _execute_workflow_steps(self, state: WorkflowState) -> WorkflowState:
        """
        Execute workflow steps until completion.
        
        Args:
            state: The workflow state
            
        Returns:
            The final workflow state
        """
        max_iterations = 100  # Prevent infinite loops
        iteration = 0
        
        while state.current_step and iteration < max_iterations:
            iteration += 1
            
            # Get current node
            node_key = f"{state.workflow_id}:{state.current_step}"
            node = self._nodes.get(node_key)
            
            if not node:
                logger.warning(f"Node not found: {state.current_step}")
                break
                
            # Execute current node
            result = await node.execute(state)
            
            # Update state
            state.data.update(result)
            state.updated_at = datetime.utcnow()
            
            # Find next step
            next_step = await self._get_next_step(state)
            if next_step == state.current_step:
                # Prevent infinite loops
                break
                
            state.current_step = next_step
        
        # Mark as completed
        state.context["status"] = "completed"
        state.context["completion_time"] = datetime.utcnow().isoformat()
        state.context["iterations"] = iteration
        
        return state
    
    async def _get_next_step(self, state: WorkflowState) -> Optional[str]:
        """
        Determine the next step in the workflow.
        
        Args:
            state: Current workflow state
            
        Returns:
            The next step ID, or None if workflow is complete
        """
        edge_key = f"{state.workflow_id}:{state.current_step}"
        edges = self._edges.get(edge_key, [])
        
        for edge in edges:
            if edge.should_traverse(state):
                edge.traversal_count += 1
                return edge.to_node
                
        # No valid edges found, workflow complete
        return None
