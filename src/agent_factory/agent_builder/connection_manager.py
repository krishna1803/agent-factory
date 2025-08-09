"""
Connection management for agent builder.

This module handles the registration, validation, and lifecycle management
of external connections that can be used by agents.
"""

import logging
from typing import Dict, List, Optional
from uuid import UUID

from agent_factory.core.models import ConnectionSpec, ConnectionType

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages external connections for the agent factory."""
    
    def __init__(self):
        """Initialize the connection manager."""
        self._connections: Dict[UUID, ConnectionSpec] = {}
        self._connection_registry: Dict[str, UUID] = {}  # name -> id mapping
        self._health_status: Dict[UUID, bool] = {}  # connection health status
        
    async def register_connection(self, connection_spec: ConnectionSpec) -> bool:
        """
        Register a connection for use by agents.
        
        Args:
            connection_spec: The connection specification to register
            
        Returns:
            True if registration successful
            
        Raises:
            ValueError: If connection name already exists or spec is invalid
        """
        # Check if connection name already exists
        if connection_spec.name in self._connection_registry:
            existing_id = self._connection_registry[connection_spec.name]
            if existing_id != connection_spec.id:
                raise ValueError(f"Connection name '{connection_spec.name}' already exists")
        
        # Validate connection specification
        await self._validate_connection_spec(connection_spec)
        
        # Register the connection
        self._connections[connection_spec.id] = connection_spec
        self._connection_registry[connection_spec.name] = connection_spec.id
        self._health_status[connection_spec.id] = False  # Will be updated on health check
        
        # Perform initial health check
        await self._health_check(connection_spec.id)
        
        logger.info(f"Registered connection: {connection_spec.name} (ID: {connection_spec.id})")
        return True
    
    async def unregister_connection(self, connection_id: UUID) -> bool:
        """
        Unregister a connection.
        
        Args:
            connection_id: The ID of the connection to unregister
            
        Returns:
            True if unregistration successful
        """
        if connection_id not in self._connections:
            return False
            
        connection_spec = self._connections[connection_id]
        
        # Remove from registries
        if connection_spec.name in self._connection_registry:
            del self._connection_registry[connection_spec.name]
            
        if connection_id in self._health_status:
            del self._health_status[connection_id]
            
        del self._connections[connection_id]
        
        logger.info(f"Unregistered connection: {connection_spec.name} (ID: {connection_id})")
        return True
    
    def get_connection(self, connection_id: UUID) -> Optional[ConnectionSpec]:
        """
        Get a connection by ID.
        
        Args:
            connection_id: The ID of the connection
            
        Returns:
            The connection specification if found, None otherwise
        """
        return self._connections.get(connection_id)
    
    def get_connection_by_name(self, name: str) -> Optional[ConnectionSpec]:
        """
        Get a connection by name.
        
        Args:
            name: The name of the connection
            
        Returns:
            The connection specification if found, None otherwise
        """
        connection_id = self._connection_registry.get(name)
        if connection_id:
            return self._connections.get(connection_id)
        return None
    
    def list_connections(self, connection_type: Optional[ConnectionType] = None, active_only: bool = True) -> List[ConnectionSpec]:
        """
        List all registered connections.
        
        Args:
            connection_type: Filter by connection type (optional)
            active_only: If True, only return active connections
            
        Returns:
            List of connection specifications
        """
        connections = list(self._connections.values())
        
        if active_only:
            connections = [conn for conn in connections if conn.is_active]
            
        if connection_type:
            connections = [conn for conn in connections if conn.connection_type == connection_type]
            
        return connections
    
    async def update_connection(self, connection_id: UUID, updated_spec: ConnectionSpec) -> bool:
        """
        Update an existing connection.
        
        Args:
            connection_id: The ID of the connection to update
            updated_spec: The updated connection specification
            
        Returns:
            True if update successful
            
        Raises:
            ValueError: If connection doesn't exist or spec is invalid
        """
        if connection_id not in self._connections:
            raise ValueError(f"Connection with ID {connection_id} not found")
            
        old_spec = self._connections[connection_id]
        
        # If name changed, check for conflicts
        if old_spec.name != updated_spec.name:
            if updated_spec.name in self._connection_registry:
                existing_id = self._connection_registry[updated_spec.name]
                if existing_id != connection_id:
                    raise ValueError(f"Connection name '{updated_spec.name}' already exists")
            
            # Update name registry
            del self._connection_registry[old_spec.name]
            self._connection_registry[updated_spec.name] = connection_id
        
        # Validate updated specification
        await self._validate_connection_spec(updated_spec)
        
        # Update the connection
        self._connections[connection_id] = updated_spec
        
        # Perform health check with new configuration
        await self._health_check(connection_id)
        
        logger.info(f"Updated connection: {updated_spec.name} (ID: {connection_id})")
        return True
    
    async def test_connection(self, connection_id: UUID) -> dict:
        """
        Test a connection.
        
        Args:
            connection_id: The ID of the connection to test
            
        Returns:
            The test result
            
        Raises:
            ValueError: If connection doesn't exist
            RuntimeError: If test fails
        """
        connection_spec = self.get_connection(connection_id)
        if not connection_spec:
            raise ValueError(f"Connection with ID {connection_id} not found")
            
        try:
            # Perform health check
            is_healthy = await self._health_check(connection_id)
            
            return {
                "status": "success" if is_healthy else "failed",
                "connection_id": str(connection_id),
                "connection_name": connection_spec.name,
                "connection_type": connection_spec.connection_type.value,
                "endpoint": connection_spec.endpoint,
                "healthy": is_healthy,
                "test_time": "2024-01-01T00:00:00Z"  # TODO: Use actual timestamp
            }
            
        except Exception as e:
            logger.error(f"Connection test failed for {connection_spec.name}: {str(e)}")
            raise RuntimeError(f"Connection test failed: {str(e)}") from e
    
    def get_connection_health(self, connection_id: UUID) -> Optional[bool]:
        """
        Get the health status of a connection.
        
        Args:
            connection_id: The ID of the connection
            
        Returns:
            True if healthy, False if unhealthy, None if unknown
        """
        return self._health_status.get(connection_id)
    
    async def _health_check(self, connection_id: UUID) -> bool:
        """
        Perform a health check on a connection.
        
        Args:
            connection_id: The ID of the connection to check
            
        Returns:
            True if healthy, False otherwise
        """
        connection_spec = self._connections.get(connection_id)
        if not connection_spec:
            return False
            
        try:
            # TODO: Implement actual health checks based on connection type
            # For now, assume all connections are healthy
            self._health_status[connection_id] = True
            return True
            
        except Exception as e:
            logger.warning(f"Health check failed for connection {connection_spec.name}: {str(e)}")
            self._health_status[connection_id] = False
            return False
    
    async def _validate_connection_spec(self, connection_spec: ConnectionSpec) -> None:
        """
        Validate a connection specification.
        
        Args:
            connection_spec: The connection specification to validate
            
        Raises:
            ValueError: If specification is invalid
        """
        # Basic validation (Pydantic handles most of this)
        if not connection_spec.name or not connection_spec.name.strip():
            raise ValueError("Connection name cannot be empty")
            
        if not connection_spec.description or not connection_spec.description.strip():
            raise ValueError("Connection description cannot be empty")
            
        if not connection_spec.endpoint or not connection_spec.endpoint.strip():
            raise ValueError("Connection endpoint cannot be empty")
            
        # Type-specific validation
        if connection_spec.connection_type == ConnectionType.DATABASE:
            if not connection_spec.credentials:
                raise ValueError("Database connections must include credentials")
                
        elif connection_spec.connection_type == ConnectionType.API:
            if not connection_spec.endpoint.startswith(('http://', 'https://')):
                raise ValueError("API connections must have HTTP/HTTPS endpoint")
                
        # Configuration validation
        if connection_spec.configuration and not isinstance(connection_spec.configuration, dict):
            raise ValueError("Connection configuration must be a dictionary")
            
        if connection_spec.credentials and not isinstance(connection_spec.credentials, dict):
            raise ValueError("Connection credentials must be a dictionary")
