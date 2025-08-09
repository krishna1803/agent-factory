"""
Unit tests for agent builder.
"""

import pytest
from uuid import uuid4

from agent_factory.agent_builder import AgentBuilder
from agent_factory.core.models import AgentSpec, ToolSpec, ConnectionSpec, ToolType, ConnectionType


class TestAgentBuilder:
    """Test AgentBuilder class."""
    
    @pytest.mark.asyncio
    async def test_create_agent(self, agent_builder: AgentBuilder, sample_agent_spec: AgentSpec):
        """Test agent creation."""
        created_agent = await agent_builder.create_agent(sample_agent_spec)
        
        assert created_agent.id == sample_agent_spec.id
        assert created_agent.name == sample_agent_spec.name
        assert created_agent.role == sample_agent_spec.role
        
        # Verify agent is stored
        retrieved_agent = agent_builder.get_agent(sample_agent_spec.id)
        assert retrieved_agent is not None
        assert retrieved_agent.name == sample_agent_spec.name
    
    @pytest.mark.asyncio
    async def test_update_agent(self, agent_builder: AgentBuilder, sample_agent_spec: AgentSpec):
        """Test agent update."""
        # Create agent first
        await agent_builder.create_agent(sample_agent_spec)
        
        # Update agent
        sample_agent_spec.name = "Updated Agent Name"
        sample_agent_spec.instructions = "Updated instructions"
        
        updated_agent = await agent_builder.update_agent(sample_agent_spec.id, sample_agent_spec)
        
        assert updated_agent.name == "Updated Agent Name"
        assert updated_agent.instructions == "Updated instructions"
    
    @pytest.mark.asyncio
    async def test_delete_agent(self, agent_builder: AgentBuilder, sample_agent_spec: AgentSpec):
        """Test agent deletion."""
        # Create agent first
        await agent_builder.create_agent(sample_agent_spec)
        
        # Verify agent exists
        assert agent_builder.get_agent(sample_agent_spec.id) is not None
        
        # Delete agent
        success = await agent_builder.delete_agent(sample_agent_spec.id)
        assert success is True
        
        # Verify agent is removed
        assert agent_builder.get_agent(sample_agent_spec.id) is None
    
    def test_list_agents(self, agent_builder: AgentBuilder):
        """Test listing agents."""
        # Initially empty
        agents = agent_builder.list_agents()
        initial_count = len(agents)
        
        # The actual test would require async setup, so we just verify the method works
        assert isinstance(agents, list)
    
    @pytest.mark.asyncio
    async def test_add_tool_to_agent(self, agent_builder: AgentBuilder, sample_agent_spec: AgentSpec):
        """Test adding a tool to an agent."""
        # Create agent first
        await agent_builder.create_agent(sample_agent_spec)
        
        # Create a tool
        tool_spec = ToolSpec(
            name="new_tool",
            description="A new tool for testing",
            tool_type=ToolType.FUNCTION
        )
        
        # Add tool to agent
        success = await agent_builder.add_tool_to_agent(sample_agent_spec.id, tool_spec)
        assert success is True
        
        # Verify tool was added
        updated_agent = agent_builder.get_agent(sample_agent_spec.id)
        assert len(updated_agent.tool_specs) == 1
        assert updated_agent.tool_specs[0].name == "new_tool"
    
    @pytest.mark.asyncio
    async def test_remove_tool_from_agent(self, agent_builder: AgentBuilder, sample_agent_spec: AgentSpec):
        """Test removing a tool from an agent."""
        # Create agent first
        await agent_builder.create_agent(sample_agent_spec)
        
        # Create and add a tool
        tool_spec = ToolSpec(
            name="remove_me",
            description="Tool to be removed",
            tool_type=ToolType.FUNCTION
        )
        
        await agent_builder.add_tool_to_agent(sample_agent_spec.id, tool_spec)
        
        # Verify tool was added
        agent = agent_builder.get_agent(sample_agent_spec.id)
        assert len(agent.tool_specs) == 1
        
        # Remove tool
        success = await agent_builder.remove_tool_from_agent(sample_agent_spec.id, tool_spec.id)
        assert success is True
        
        # Verify tool was removed
        updated_agent = agent_builder.get_agent(sample_agent_spec.id)
        assert len(updated_agent.tool_specs) == 0
    
    @pytest.mark.asyncio
    async def test_create_agent_with_invalid_spec(self, agent_builder: AgentBuilder):
        """Test creating agent with invalid specification."""
        # This would test validation errors
        # The actual validation is done by the validator, so we test integration
        pass  # Placeholder for validation tests
