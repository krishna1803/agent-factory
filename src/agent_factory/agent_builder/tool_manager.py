"""
Tool management for agent builder.

This module handles the registration, validation, and lifecycle management
of tools that can be used by agents.
"""

import logging
from typing import Dict, List, Optional
from uuid import UUID

from agent_factory.core.models import ToolSpec, ToolType

logger = logging.getLogger(__name__)


class ToolManager:
    """Manages tools for the agent factory."""
    
    def __init__(self):
        """Initialize the tool manager."""
        self._tools: Dict[UUID, ToolSpec] = {}
        self._tool_registry: Dict[str, UUID] = {}  # name -> id mapping
        
    async def register_tool(self, tool_spec: ToolSpec) -> bool:
        """
        Register a tool for use by agents.
        
        Args:
            tool_spec: The tool specification to register
            
        Returns:
            True if registration successful
            
        Raises:
            ValueError: If tool name already exists or spec is invalid
        """
        # Check if tool name already exists
        if tool_spec.name in self._tool_registry:
            existing_id = self._tool_registry[tool_spec.name]
            if existing_id != tool_spec.id:
                raise ValueError(f"Tool name '{tool_spec.name}' already exists")
        
        # Validate tool specification
        await self._validate_tool_spec(tool_spec)
        
        # Register the tool
        self._tools[tool_spec.id] = tool_spec
        self._tool_registry[tool_spec.name] = tool_spec.id
        
        logger.info(f"Registered tool: {tool_spec.name} (ID: {tool_spec.id})")
        return True
    
    async def unregister_tool(self, tool_id: UUID) -> bool:
        """
        Unregister a tool.
        
        Args:
            tool_id: The ID of the tool to unregister
            
        Returns:
            True if unregistration successful
        """
        if tool_id not in self._tools:
            return False
            
        tool_spec = self._tools[tool_id]
        
        # Remove from registry
        if tool_spec.name in self._tool_registry:
            del self._tool_registry[tool_spec.name]
            
        del self._tools[tool_id]
        
        logger.info(f"Unregistered tool: {tool_spec.name} (ID: {tool_id})")
        return True
    
    def get_tool(self, tool_id: UUID) -> Optional[ToolSpec]:
        """
        Get a tool by ID.
        
        Args:
            tool_id: The ID of the tool
            
        Returns:
            The tool specification if found, None otherwise
        """
        return self._tools.get(tool_id)
    
    def get_tool_by_name(self, name: str) -> Optional[ToolSpec]:
        """
        Get a tool by name.
        
        Args:
            name: The name of the tool
            
        Returns:
            The tool specification if found, None otherwise
        """
        tool_id = self._tool_registry.get(name)
        if tool_id:
            return self._tools.get(tool_id)
        return None
    
    def list_tools(self, tool_type: Optional[ToolType] = None, active_only: bool = True) -> List[ToolSpec]:
        """
        List all registered tools.
        
        Args:
            tool_type: Filter by tool type (optional)
            active_only: If True, only return active tools
            
        Returns:
            List of tool specifications
        """
        tools = list(self._tools.values())
        
        if active_only:
            tools = [tool for tool in tools if tool.is_active]
            
        if tool_type:
            tools = [tool for tool in tools if tool.tool_type == tool_type]
            
        return tools
    
    async def update_tool(self, tool_id: UUID, updated_spec: ToolSpec) -> bool:
        """
        Update an existing tool.
        
        Args:
            tool_id: The ID of the tool to update
            updated_spec: The updated tool specification
            
        Returns:
            True if update successful
            
        Raises:
            ValueError: If tool doesn't exist or spec is invalid
        """
        if tool_id not in self._tools:
            raise ValueError(f"Tool with ID {tool_id} not found")
            
        old_spec = self._tools[tool_id]
        
        # If name changed, check for conflicts
        if old_spec.name != updated_spec.name:
            if updated_spec.name in self._tool_registry:
                existing_id = self._tool_registry[updated_spec.name]
                if existing_id != tool_id:
                    raise ValueError(f"Tool name '{updated_spec.name}' already exists")
            
            # Update name registry
            del self._tool_registry[old_spec.name]
            self._tool_registry[updated_spec.name] = tool_id
        
        # Validate updated specification
        await self._validate_tool_spec(updated_spec)
        
        # Update the tool
        self._tools[tool_id] = updated_spec
        
        logger.info(f"Updated tool: {updated_spec.name} (ID: {tool_id})")
        return True
    
    async def test_tool(self, tool_id: UUID, test_input: dict) -> dict:
        """
        Test a tool with given input.
        
        Args:
            tool_id: The ID of the tool to test
            test_input: The test input data
            
        Returns:
            The test result
            
        Raises:
            ValueError: If tool doesn't exist
            RuntimeError: If test fails
        """
        tool_spec = self.get_tool(tool_id)
        if not tool_spec:
            raise ValueError(f"Tool with ID {tool_id} not found")
            
        try:
            # TODO: Implement actual tool execution based on tool type
            # For now, return a mock response
            return {
                "status": "success",
                "tool_id": str(tool_id),
                "tool_name": tool_spec.name,
                "input": test_input,
                "output": {"message": "Tool test successful"},
                "execution_time_ms": 100
            }
            
        except Exception as e:
            logger.error(f"Tool test failed for {tool_spec.name}: {str(e)}")
            raise RuntimeError(f"Tool test failed: {str(e)}") from e
    
    async def _validate_tool_spec(self, tool_spec: ToolSpec) -> None:
        """
        Validate a tool specification.
        
        Args:
            tool_spec: The tool specification to validate
            
        Raises:
            ValueError: If specification is invalid
        """
        # Basic validation (Pydantic handles most of this)
        if not tool_spec.name or not tool_spec.name.strip():
            raise ValueError("Tool name cannot be empty")
            
        if not tool_spec.description or not tool_spec.description.strip():
            raise ValueError("Tool description cannot be empty")
            
        # Type-specific validation
        if tool_spec.tool_type == ToolType.API:
            if not tool_spec.endpoint_url:
                raise ValueError("API tools must have an endpoint URL")
                
        elif tool_spec.tool_type == ToolType.WEBHOOK:
            if not tool_spec.endpoint_url:
                raise ValueError("Webhook tools must have an endpoint URL")
                
        # Schema validation
        if tool_spec.input_schema and not isinstance(tool_spec.input_schema, dict):
            raise ValueError("Input schema must be a dictionary")
            
        if tool_spec.output_schema and not isinstance(tool_spec.output_schema, dict):
            raise ValueError("Output schema must be a dictionary")
