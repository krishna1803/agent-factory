"""
Agent validation utilities.

This module provides validation functionality for agent specifications,
ensuring they meet all requirements before being processed.
"""

import re
from typing import Set
from uuid import UUID

from agent_factory.core.models import AgentSpec, ToolSpec, ConnectionSpec


class AgentValidator:
    """Validator for agent specifications and related components."""
    
    def __init__(self):
        """Initialize the validator with validation rules."""
        self.reserved_names: Set[str] = {
            "system", "admin", "root", "default", "null", "undefined"
        }
        
    async def validate_agent_spec(self, agent_spec: AgentSpec) -> bool:
        """
        Validate an agent specification.
        
        Args:
            agent_spec: The agent specification to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If validation fails
        """
        # Basic field validation (handled by Pydantic, but we can add custom rules)
        self._validate_agent_name(agent_spec.name)
        self._validate_instructions(agent_spec.instructions)
        
        # Validate tools
        for tool_spec in agent_spec.tool_specs:
            await self.validate_tool_spec(tool_spec)
            
        # Validate connections
        for connection_spec in agent_spec.connection_specs:
            await self.validate_connection_spec(connection_spec)
            
        # Validate model profile
        self._validate_model_profile(agent_spec.model_profile)
        
        return True
    
    async def validate_tool_spec(self, tool_spec: ToolSpec) -> bool:
        """
        Validate a tool specification.
        
        Args:
            tool_spec: The tool specification to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If validation fails
        """
        self._validate_tool_name(tool_spec.name)
        self._validate_schema(tool_spec.input_schema, "input")
        self._validate_schema(tool_spec.output_schema, "output")
        
        if tool_spec.timeout_seconds <= 0:
            raise ValueError("Tool timeout must be positive")
            
        return True
    
    async def validate_connection_spec(self, connection_spec: ConnectionSpec) -> bool:
        """
        Validate a connection specification.
        
        Args:
            connection_spec: The connection specification to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If validation fails
        """
        self._validate_connection_name(connection_spec.name)
        self._validate_endpoint(connection_spec.endpoint)
        
        if connection_spec.timeout_seconds <= 0:
            raise ValueError("Connection timeout must be positive")
            
        return True
    
    def _validate_agent_name(self, name: str) -> None:
        """Validate agent name."""
        if not name or not name.strip():
            raise ValueError("Agent name cannot be empty")
            
        if name.lower() in self.reserved_names:
            raise ValueError(f"Agent name '{name}' is reserved")
            
        if len(name) > 100:
            raise ValueError("Agent name cannot exceed 100 characters")
    
    def _validate_tool_name(self, name: str) -> None:
        """Validate tool name."""
        if not name or not name.strip():
            raise ValueError("Tool name cannot be empty")
            
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', name):
            raise ValueError("Tool name must start with a letter and contain only alphanumeric characters, hyphens, and underscores")
            
        if len(name) > 50:
            raise ValueError("Tool name cannot exceed 50 characters")
    
    def _validate_connection_name(self, name: str) -> None:
        """Validate connection name."""
        if not name or not name.strip():
            raise ValueError("Connection name cannot be empty")
            
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', name):
            raise ValueError("Connection name must start with a letter and contain only alphanumeric characters, hyphens, and underscores")
            
        if len(name) > 50:
            raise ValueError("Connection name cannot exceed 50 characters")
    
    def _validate_instructions(self, instructions: str) -> None:
        """Validate agent instructions."""
        if not instructions or not instructions.strip():
            raise ValueError("Agent instructions cannot be empty")
            
        if len(instructions.strip()) < 10:
            raise ValueError("Agent instructions must be at least 10 characters long")
            
        if len(instructions) > 10000:
            raise ValueError("Agent instructions cannot exceed 10,000 characters")
    
    def _validate_schema(self, schema: dict, schema_type: str) -> None:
        """Validate JSON schema."""
        if not isinstance(schema, dict):
            raise ValueError(f"{schema_type} schema must be a dictionary")
            
        # Basic JSON schema validation
        if schema and "type" not in schema:
            raise ValueError(f"{schema_type} schema must include a 'type' field")
    
    def _validate_endpoint(self, endpoint: str) -> None:
        """Validate connection endpoint."""
        if not endpoint or not endpoint.strip():
            raise ValueError("Connection endpoint cannot be empty")
            
        # Basic URL validation
        url_pattern = re.compile(
            r'^(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]*[-A-Za-z0-9+&@#/%=~_|]$'
        )
        
        if not url_pattern.match(endpoint):
            raise ValueError("Connection endpoint must be a valid URL")
    
    def _validate_model_profile(self, model_profile) -> None:
        """Validate model profile configuration."""
        if not model_profile.model_name or not model_profile.model_name.strip():
            raise ValueError("Model name cannot be empty")
            
        if model_profile.temperature < 0 or model_profile.temperature > 2:
            raise ValueError("Model temperature must be between 0 and 2")
            
        if model_profile.max_tokens <= 0:
            raise ValueError("Model max_tokens must be positive")
