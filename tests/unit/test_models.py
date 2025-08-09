"""
Unit tests for core models.
"""

import pytest
from uuid import uuid4
from datetime import datetime

from agent_factory.core.models import (
    AgentSpec, ToolSpec, ConnectionSpec, ModelProfile,
    AgentRole, ModelProvider, ToolType, ConnectionType
)


class TestModelProfile:
    """Test ModelProfile class."""
    
    def test_model_profile_creation(self):
        """Test basic model profile creation."""
        profile = ModelProfile(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4",
            temperature=0.5,
            max_tokens=2000
        )
        
        assert profile.provider == ModelProvider.OPENAI
        assert profile.model_name == "gpt-4"
        assert profile.temperature == 0.5
        assert profile.max_tokens == 2000
    
    def test_model_profile_defaults(self):
        """Test model profile default values."""
        profile = ModelProfile(
            provider=ModelProvider.OPENAI,
            model_name="gpt-3.5-turbo"
        )
        
        assert profile.temperature == 0.7
        assert profile.max_tokens == 1000
        assert profile.top_p == 1.0


class TestToolSpec:
    """Test ToolSpec class."""
    
    def test_tool_spec_creation(self):
        """Test basic tool spec creation."""
        tool = ToolSpec(
            name="test_tool",
            description="A test tool",
            tool_type=ToolType.API,
            endpoint_url="https://api.example.com"
        )
        
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.tool_type == ToolType.API
        assert tool.endpoint_url == "https://api.example.com"
        assert tool.is_active is True
    
    def test_tool_spec_validation(self):
        """Test tool spec validation."""
        # Valid tool name
        tool = ToolSpec(
            name="valid_tool_name",
            description="Valid tool",
            tool_type=ToolType.FUNCTION
        )
        assert tool.name == "valid_tool_name"
        
        # Test that invalid names would be caught by Pydantic validators
        # (The actual validation logic is in validators.py)


class TestConnectionSpec:
    """Test ConnectionSpec class."""
    
    def test_connection_spec_creation(self):
        """Test basic connection spec creation."""
        connection = ConnectionSpec(
            name="test_db",
            description="Test database connection",
            connection_type=ConnectionType.DATABASE,
            endpoint="postgresql://localhost:5432/testdb"
        )
        
        assert connection.name == "test_db"
        assert connection.description == "Test database connection"
        assert connection.connection_type == ConnectionType.DATABASE
        assert connection.endpoint == "postgresql://localhost:5432/testdb"
        assert connection.is_active is True


class TestAgentSpec:
    """Test AgentSpec class."""
    
    def test_agent_spec_creation(self, sample_model_profile):
        """Test basic agent spec creation."""
        agent = AgentSpec(
            name="Test Agent",
            role=AgentRole.RESEARCHER,
            instructions="Test instructions for the agent",
            model_profile=sample_model_profile
        )
        
        assert agent.name == "Test Agent"
        assert agent.role == AgentRole.RESEARCHER
        assert agent.instructions == "Test instructions for the agent"
        assert agent.model_profile == sample_model_profile
        assert agent.is_active is True
    
    def test_agent_spec_with_tools_and_connections(self, sample_model_profile):
        """Test agent spec with tools and connections."""
        tool = ToolSpec(
            name="search_tool",
            description="Search tool",
            tool_type=ToolType.API
        )
        
        connection = ConnectionSpec(
            name="api_connection",
            description="API connection",
            connection_type=ConnectionType.API,
            endpoint="https://api.example.com"
        )
        
        agent = AgentSpec(
            name="Complex Agent",
            role=AgentRole.ANALYST,
            instructions="Complex agent with tools and connections",
            model_profile=sample_model_profile,
            tool_specs=[tool],
            connection_specs=[connection]
        )
        
        assert len(agent.tool_specs) == 1
        assert len(agent.connection_specs) == 1
        assert agent.tool_specs[0].name == "search_tool"
        assert agent.connection_specs[0].name == "api_connection"
    
    def test_agent_spec_defaults(self, sample_model_profile):
        """Test agent spec default values."""
        agent = AgentSpec(
            name="Default Agent",
            role=AgentRole.COORDINATOR,
            instructions="Default agent configuration",
            model_profile=sample_model_profile
        )
        
        assert agent.max_iterations == 10
        assert agent.context_window == 4000
        assert agent.tool_specs == []
        assert agent.connection_specs == []
        assert agent.tags == []
        assert isinstance(agent.created_at, datetime)
        assert isinstance(agent.updated_at, datetime)
