"""
Core models and data structures for the Agent Factory.

This module defines the fundamental data models used throughout the system:
- AgentSpec: Specification for creating agents
- ToolSpec: Specification for tools and their schemas
- ConnectionSpec: Specification for external connections
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class AgentRole(str, Enum):
    """Predefined agent roles."""
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"
    REVIEWER = "reviewer"
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"


class ModelProvider(str, Enum):
    """Supported model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE = "azure"
    LOCAL = "local"


class ToolType(str, Enum):
    """Types of tools available."""
    API = "api"
    FUNCTION = "function"
    WEBHOOK = "webhook"
    DATABASE = "database"
    FILE_SYSTEM = "file_system"
    CUSTOM = "custom"


class ConnectionType(str, Enum):
    """Types of external connections."""
    DATABASE = "database"
    API = "api"
    WEBHOOK = "webhook"
    MESSAGE_QUEUE = "message_queue"
    STORAGE = "storage"
    CACHE = "cache"


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""
    requests_per_minute: int = Field(default=60, ge=1)
    requests_per_hour: int = Field(default=1000, ge=1)
    burst_size: int = Field(default=10, ge=1)


class ModelProfile(BaseModel):
    """Model configuration profile."""
    provider: ModelProvider
    model_name: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, ge=1)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    additional_config: Dict[str, Any] = Field(default_factory=dict)


class ToolSpec(BaseModel):
    """Specification for a tool."""
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    tool_type: ToolType
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int = Field(default=30, ge=1)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    endpoint_url: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)
    authentication: Dict[str, Any] = Field(default_factory=dict)
    retry_config: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True
    
    @validator('name')
    def validate_name(cls, v):
        """Validate tool name format."""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Tool name must contain only alphanumeric characters, hyphens, and underscores')
        return v


class ConnectionSpec(BaseModel):
    """Specification for external connections."""
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    connection_type: ConnectionType
    endpoint: str = Field(..., min_length=1)
    credentials: Dict[str, Any] = Field(default_factory=dict)
    configuration: Dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int = Field(default=30, ge=1)
    retry_attempts: int = Field(default=3, ge=0)
    health_check_interval: int = Field(default=300, ge=60)  # seconds
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True
    
    @validator('name')
    def validate_name(cls, v):
        """Validate connection name format."""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Connection name must contain only alphanumeric characters, hyphens, and underscores')
        return v


class AgentSpec(BaseModel):
    """Specification for creating an agent."""
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1)
    role: Union[AgentRole, str]
    instructions: str = Field(..., min_length=1)
    model_profile: ModelProfile
    tool_specs: List[ToolSpec] = Field(default_factory=list)
    connection_specs: List[ConnectionSpec] = Field(default_factory=list)
    system_prompt: Optional[str] = None
    max_iterations: int = Field(default=10, ge=1)
    memory_config: Dict[str, Any] = Field(default_factory=dict)
    context_window: int = Field(default=4000, ge=100)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True
    
    @validator('name')
    def validate_name(cls, v):
        """Validate agent name format."""
        if not v.replace('_', '').replace('-', '').replace(' ', '').isalnum():
            raise ValueError('Agent name must contain only alphanumeric characters, hyphens, underscores, and spaces')
        return v
    
    @validator('instructions')
    def validate_instructions(cls, v):
        """Validate instructions are meaningful."""
        if len(v.strip()) < 10:
            raise ValueError('Instructions must be at least 10 characters long')
        return v


class WorkflowState(BaseModel):
    """State representation for workflow execution."""
    id: UUID = Field(default_factory=uuid4)
    workflow_id: UUID
    current_step: str
    data: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    history: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class WorkflowDefinition(BaseModel):
    """Definition of a workflow with agents and their connections."""
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    agent_specs: List[AgentSpec] = Field(default_factory=list)
    workflow_graph: Dict[str, Any] = Field(default_factory=dict)
    entry_point: str
    exit_conditions: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    version: str = "1.0.0"
    is_active: bool = True
