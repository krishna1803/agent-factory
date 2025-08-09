"""
Integration tests for the Agent Factory API.
"""

import pytest
from fastapi.testclient import TestClient

from agent_factory.api.main import app


@pytest.fixture
def client():
    """Create a test client for the API."""
    with TestClient(app) as test_client:
        yield test_client


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_basic_health_check(self, client: TestClient):
        """Test basic health check endpoint."""
        response = client.get("/health/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["service"] == "Agent Factory API"
    
    def test_detailed_health_check(self, client: TestClient):
        """Test detailed health check endpoint."""
        response = client.get("/health/detailed")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "components" in data
        assert "api" in data["components"]
        assert "database" in data["components"]


class TestAgentEndpoints:
    """Test agent management endpoints."""
    
    def test_list_agents_empty(self, client: TestClient):
        """Test listing agents when none exist."""
        response = client.get("/api/v1/agents/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["total"] == 0
        assert data["agents"] == []
    
    def test_create_agent(self, client: TestClient):
        """Test creating a new agent."""
        agent_data = {
            "name": "Test API Agent",
            "role": "researcher",
            "instructions": "This is a test agent created via API",
            "model_profile": {
                "provider": "openai",
                "model_name": "gpt-3.5-turbo",
                "temperature": 0.7,
                "max_tokens": 1000
            }
        }
        
        response = client.post("/api/v1/agents/", json=agent_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["agent"]["name"] == "Test API Agent"
        assert data["agent"]["role"] == "researcher"
    
    def test_get_nonexistent_agent(self, client: TestClient):
        """Test getting an agent that doesn't exist."""
        fake_id = "550e8400-e29b-41d4-a716-446655440000"
        response = client.get(f"/api/v1/agents/{fake_id}")
        assert response.status_code == 404


class TestWorkflowEndpoints:
    """Test workflow management endpoints."""
    
    def test_list_workflows_empty(self, client: TestClient):
        """Test listing workflows when none exist."""
        response = client.get("/api/v1/workflows/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["total"] == 0
        assert data["workflows"] == []


class TestToolEndpoints:
    """Test tool management endpoints."""
    
    def test_list_tools_empty(self, client: TestClient):
        """Test listing tools when none exist."""
        response = client.get("/api/v1/tools/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["total"] == 0
        assert data["tools"] == []


class TestConnectionEndpoints:
    """Test connection management endpoints."""
    
    def test_list_connections_empty(self, client: TestClient):
        """Test listing connections when none exist."""
        response = client.get("/api/v1/connections/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["total"] == 0
        assert data["connections"] == []


class TestAPIInfo:
    """Test API information endpoints."""
    
    def test_root_endpoint(self, client: TestClient):
        """Test the root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["version"] == "0.1.0"
    
    def test_api_info_endpoint(self, client: TestClient):
        """Test the API info endpoint."""
        response = client.get("/api/v1/info")
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "Agent Factory API"
        assert data["version"] == "0.1.0"
        assert "endpoints" in data
