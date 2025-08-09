"""
Agent Builder - Main component for creating and managing agents.

This module provides the AgentBuilder class which orchestrates the creation
of agents based on provided specifications.
"""

import logging
from typing import Dict, List, Optional
from uuid import UUID

from agent_factory.core.models import AgentSpec, ToolSpec, ConnectionSpec
from .validators import AgentValidator
from .tool_manager import ToolManager
from .connection_manager import ConnectionManager

logger = logging.getLogger(__name__)


class AgentBuilder:
    """
    Main class for building agents from specifications.
    
    The AgentBuilder coordinates the creation of agents by:
    1. Validating agent specifications
    2. Managing tools and connections
    3. Creating executable agent instances
    """
    
    def __init__(self):
        """Initialize the agent builder with necessary managers."""
        self.validator = AgentValidator()
        self.tool_manager = ToolManager()
        self.connection_manager = ConnectionManager()
        self._agents: Dict[UUID, AgentSpec] = {}
        
    async def create_agent(self, agent_spec: AgentSpec) -> AgentSpec:
        """
        Create an agent from the provided specification.
        
        Args:
            agent_spec: The agent specification containing all configuration
            
        Returns:
            The validated and processed agent specification
            
        Raises:
            ValueError: If the agent specification is invalid
            RuntimeError: If agent creation fails
        """
        try:
            # Validate the agent specification
            await self.validator.validate_agent_spec(agent_spec)
            
            # Register tools for the agent
            for tool_spec in agent_spec.tool_specs:
                await self.tool_manager.register_tool(tool_spec)
                
            # Register connections for the agent
            for connection_spec in agent_spec.connection_specs:
                await self.connection_manager.register_connection(connection_spec)
                
            # Store the agent specification
            self._agents[agent_spec.id] = agent_spec
            
            logger.info(f"Successfully created agent: {agent_spec.name} (ID: {agent_spec.id})")
            return agent_spec
            
        except Exception as e:
            logger.error(f"Failed to create agent {agent_spec.name}: {str(e)}")
            raise RuntimeError(f"Agent creation failed: {str(e)}") from e
    
    async def update_agent(self, agent_id: UUID, updated_spec: AgentSpec) -> AgentSpec:
        """
        Update an existing agent with new specifications.
        
        Args:
            agent_id: The ID of the agent to update
            updated_spec: The updated agent specification
            
        Returns:
            The updated agent specification
            
        Raises:
            ValueError: If the agent doesn't exist or specification is invalid
        """
        if agent_id not in self._agents:
            raise ValueError(f"Agent with ID {agent_id} not found")
            
        # Validate the updated specification
        await self.validator.validate_agent_spec(updated_spec)
        
        # Update the stored specification
        self._agents[agent_id] = updated_spec
        
        logger.info(f"Successfully updated agent: {updated_spec.name} (ID: {agent_id})")
        return updated_spec
    
    async def delete_agent(self, agent_id: UUID) -> bool:
        """
        Delete an agent and clean up its resources.
        
        Args:
            agent_id: The ID of the agent to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        if agent_id not in self._agents:
            return False
            
        agent_spec = self._agents[agent_id]
        
        # Clean up tools and connections
        for tool_spec in agent_spec.tool_specs:
            await self.tool_manager.unregister_tool(tool_spec.id)
            
        for connection_spec in agent_spec.connection_specs:
            await self.connection_manager.unregister_connection(connection_spec.id)
            
        # Remove the agent
        del self._agents[agent_id]
        
        logger.info(f"Successfully deleted agent: {agent_spec.name} (ID: {agent_id})")
        return True
    
    def get_agent(self, agent_id: UUID) -> Optional[AgentSpec]:
        """
        Retrieve an agent specification by ID.
        
        Args:
            agent_id: The ID of the agent to retrieve
            
        Returns:
            The agent specification if found, None otherwise
        """
        return self._agents.get(agent_id)
    
    def list_agents(self, active_only: bool = True) -> List[AgentSpec]:
        """
        List all registered agents.
        
        Args:
            active_only: If True, only return active agents
            
        Returns:
            List of agent specifications
        """
        agents = list(self._agents.values())
        if active_only:
            agents = [agent for agent in agents if agent.is_active]
        return agents
    
    async def add_tool_to_agent(self, agent_id: UUID, tool_spec: ToolSpec) -> bool:
        """
        Add a tool to an existing agent.
        
        Args:
            agent_id: The ID of the agent
            tool_spec: The tool specification to add
            
        Returns:
            True if successful, False otherwise
        """
        if agent_id not in self._agents:
            return False
            
        agent = self._agents[agent_id]
        
        # Register the tool
        await self.tool_manager.register_tool(tool_spec)
        
        # Add to agent's tool list
        agent.tool_specs.append(tool_spec)
        
        logger.info(f"Added tool {tool_spec.name} to agent {agent.name}")
        return True
    
    async def remove_tool_from_agent(self, agent_id: UUID, tool_id: UUID) -> bool:
        """
        Remove a tool from an existing agent.
        
        Args:
            agent_id: The ID of the agent
            tool_id: The ID of the tool to remove
            
        Returns:
            True if successful, False otherwise
        """
        if agent_id not in self._agents:
            return False
            
        agent = self._agents[agent_id]
        
        # Find and remove the tool
        tool_to_remove = None
        for tool in agent.tool_specs:
            if tool.id == tool_id:
                tool_to_remove = tool
                break
                
        if tool_to_remove:
            agent.tool_specs.remove(tool_to_remove)
            await self.tool_manager.unregister_tool(tool_id)
            logger.info(f"Removed tool {tool_to_remove.name} from agent {agent.name}")
            return True
            
        return False
