"""
Test configuration and fixtures.
"""

import pytest
import asyncio
from typing import Generator

from agent_factory.agent_builder import AgentBuilder
from agent_factory.orchestration import WorkflowOrchestrator
from agent_factory.core.models import AgentSpec, ModelProfile, ModelProvider


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def agent_builder() -> AgentBuilder:
    """Create an agent builder instance for testing."""
    return AgentBuilder()


@pytest.fixture
def workflow_orchestrator() -> WorkflowOrchestrator:
    """Create a workflow orchestrator instance for testing."""
    return WorkflowOrchestrator()


@pytest.fixture
def sample_model_profile() -> ModelProfile:
    """Create a sample model profile for testing."""
    return ModelProfile(
        provider=ModelProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=1000
    )


@pytest.fixture
def sample_agent_spec(sample_model_profile: ModelProfile) -> AgentSpec:
    """Create a sample agent spec for testing."""
    return AgentSpec(
        name="Test Agent",
        role="researcher",
        instructions="This is a test agent for unit testing",
        model_profile=sample_model_profile
    )
