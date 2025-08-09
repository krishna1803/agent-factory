# Agent Factory API Reference

This document provides a comprehensive reference for the Agent Factory REST API.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

Currently, the API does not require authentication for development. In production, JWT tokens will be required.

## Content Type

All requests and responses use `application/json` content type.

## Rate Limiting

- **Default**: 100 requests per minute per IP
- **Headers**: Rate limit information is included in response headers:
  - `X-RateLimit-Limit`: Request limit
  - `X-RateLimit-Remaining`: Remaining requests
  - `X-RateLimit-Reset`: Reset timestamp

## Error Responses

All error responses follow this format:

```json
{
  "error": "Error Type",
  "message": "Detailed error message",
  "details": {}
}
```

Common HTTP status codes:
- `400`: Bad Request - Invalid input
- `401`: Unauthorized - Authentication required
- `403`: Forbidden - Insufficient permissions
- `404`: Not Found - Resource doesn't exist
- `429`: Too Many Requests - Rate limit exceeded
- `500`: Internal Server Error - Server error

## Endpoints

### Health Check

#### GET /health

Basic health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "service": "Agent Factory API",
  "version": "0.1.0"
}
```

#### GET /health/detailed

Detailed health check with component status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "service": "Agent Factory API",
  "version": "0.1.0",
  "components": {
    "api": {"status": "healthy", "response_time_ms": 1},
    "database": {"status": "healthy", "response_time_ms": 5}
  }
}
```

### Agents

#### POST /agents

Create a new agent.

**Request Body:**
```json
{
  "name": "Research Assistant",
  "role": "researcher",
  "instructions": "You are a research assistant that helps find and analyze information.",
  "model_profile": {
    "provider": "openai",
    "model_name": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 2000
  },
  "tool_specs": [],
  "connection_specs": [],
  "tags": ["research", "assistant"]
}
```

**Response:**
```json
{
  "success": true,
  "message": "Agent 'Research Assistant' created successfully",
  "agent": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "Research Assistant",
    "role": "researcher",
    "instructions": "You are a research assistant...",
    "model_profile": {...},
    "created_at": "2024-01-01T00:00:00Z",
    "is_active": true
  }
}
```

#### GET /agents

List all agents.

**Query Parameters:**
- `active_only` (boolean): Filter to active agents only (default: true)

**Response:**
```json
{
  "success": true,
  "message": "Agents retrieved successfully",
  "agents": [...],
  "total": 5
}
```

#### GET /agents/{agent_id}

Get a specific agent by ID.

**Response:**
```json
{
  "success": true,
  "message": "Agent retrieved successfully",
  "agent": {...}
}
```

#### PUT /agents/{agent_id}

Update an existing agent.

**Request Body:** Same as POST /agents

**Response:**
```json
{
  "success": true,
  "message": "Agent 'Research Assistant' updated successfully",
  "agent": {...}
}
```

#### DELETE /agents/{agent_id}

Delete an agent.

**Response:**
```json
{
  "success": true,
  "message": "Agent deleted successfully"
}
```

### Tools

#### POST /tools

Register a new tool.

**Request Body:**
```json
{
  "name": "web_search",
  "description": "Search the web for information",
  "tool_type": "api",
  "input_schema": {
    "type": "object",
    "properties": {
      "query": {"type": "string"},
      "max_results": {"type": "integer", "default": 10}
    },
    "required": ["query"]
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "results": {"type": "array"}
    }
  },
  "endpoint_url": "https://api.search.com/search",
  "timeout_seconds": 30
}
```

#### GET /tools

List all tools.

**Query Parameters:**
- `tool_type` (string): Filter by tool type
- `active_only` (boolean): Filter to active tools only

#### GET /tools/{tool_id}

Get a specific tool by ID.

#### POST /tools/{tool_id}/test

Test a tool with sample input.

**Request Body:**
```json
{
  "query": "artificial intelligence",
  "max_results": 5
}
```

### Connections

#### POST /connections

Register a new connection.

**Request Body:**
```json
{
  "name": "postgres_db",
  "description": "Main PostgreSQL database",
  "connection_type": "database",
  "endpoint": "postgresql://localhost:5432/mydb",
  "credentials": {
    "username": "user",
    "password": "pass"
  },
  "timeout_seconds": 30
}
```

#### GET /connections

List all connections.

#### POST /connections/{connection_id}/test

Test a connection.

### Workflows

#### POST /workflows

Create a new workflow.

**Request Body:**
```json
{
  "name": "Research Pipeline",
  "description": "Multi-step research workflow",
  "agent_specs": [...],
  "workflow_graph": {
    "nodes": {
      "start": {"type": "start"},
      "research": {"agent": "Research Assistant"},
      "analyze": {"agent": "Data Analyst"},
      "end": {"type": "end"}
    },
    "edges": [
      {"from": "start", "to": "research"},
      {"from": "research", "to": "analyze"},
      {"from": "analyze", "to": "end"}
    ]
  },
  "entry_point": "start"
}
```

#### POST /workflows/{workflow_id}/execute

Execute a workflow.

**Request Body:**
```json
{
  "research_topic": "machine learning trends",
  "depth": "comprehensive"
}
```

#### GET /workflows/{workflow_id}/status/{execution_id}

Get workflow execution status.

## WebSocket Endpoints

### /ws/executions/{execution_id}

Real-time workflow execution updates.

**Message Format:**
```json
{
  "type": "status_update",
  "execution_id": "...",
  "current_step": "research",
  "progress": 50,
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## SDK Examples

### Python

```python
import requests

# Create an agent
agent_data = {
    "name": "My Agent",
    "role": "assistant",
    "instructions": "Help with tasks",
    "model_profile": {
        "provider": "openai",
        "model_name": "gpt-3.5-turbo"
    }
}

response = requests.post(
    "http://localhost:8000/api/v1/agents",
    json=agent_data
)

agent = response.json()["agent"]
print(f"Created agent: {agent['name']}")
```

### JavaScript

```javascript
// Create an agent
const agentData = {
  name: "My Agent",
  role: "assistant", 
  instructions: "Help with tasks",
  model_profile: {
    provider: "openai",
    model_name: "gpt-3.5-turbo"
  }
};

const response = await fetch('http://localhost:8000/api/v1/agents', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify(agentData)
});

const result = await response.json();
console.log(`Created agent: ${result.agent.name}`);
```

## OpenAPI Specification

The full OpenAPI specification is available at:
- **JSON**: http://localhost:8000/openapi.json
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
