# Agent Factory Documentation

Welcome to the Agent Factory documentation! This guide will help you understand and use the Agent Factory system to create, orchestrate, and manage AI agents.

## Table of Contents

1. [Getting Started](getting-started.md)
2. [User Guide](user-guide/)
3. [API Reference](api/)
4. [Development](development/)

## What is Agent Factory?

Agent Factory is a comprehensive platform for creating, orchestrating, and managing AI agents. It provides:

- **Agent Builder**: Create agents with defined specifications
- **Orchestration Runtime**: LangGraph/LangChain-based workflow orchestration
- **UI Component**: StreamLit-based visualization and management
- **API Component**: FastAPI-based REST API for programmatic access

## Quick Start

### Prerequisites

- Python 3.9 or higher
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/agent-factory/agent-factory.git
cd agent-factory
```

2. Run the setup script:
```bash
./scripts/setup.sh
```

3. Start the services:
```bash
./scripts/run.sh all
```

4. Open your browser:
- UI: http://localhost:8501
- API: http://localhost:8000/docs

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   StreamLit UI  │    │   FastAPI API   │    │ Agent Builder   │
│                 │◄───┤                 │◄───┤                 │
│ - Agent Creation│    │ - REST Endpoints│    │ - Agent Specs   │
│ - Workflow UI   │    │ - Authentication│    │ - Tool Specs    │
│ - Monitoring    │    │ - Rate Limiting │    │ - Validation    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  Orchestration  │
                       │                 │
                       │ - LangGraph     │
                       │ - Workflows     │
                       │ - State Mgmt    │
                       └─────────────────┘
```

## Core Concepts

### Agent Specifications

An Agent Specification defines:
- **Name & Role**: Identity and purpose
- **Instructions**: What the agent should do
- **Model Profile**: LLM configuration
- **Tools**: Available functions and APIs
- **Connections**: External service connections

### Tool Specifications

Tool Specifications define:
- **Name & Description**: Tool identity
- **Type**: API, Function, Webhook, etc.
- **Schemas**: Input/output data structures
- **Configuration**: Timeouts, rate limits, auth

### Connection Specifications

Connection Specifications define:
- **Type**: Database, API, Cache, etc.
- **Endpoint**: Connection details
- **Credentials**: Authentication information
- **Configuration**: Timeouts, retries, health checks

### Workflows

Workflows orchestrate multiple agents:
- **Graph Structure**: Nodes and edges
- **State Management**: Data flow between agents
- **Execution Control**: Branching, loops, conditions
- **Monitoring**: Progress tracking and logging

## Getting Help

- **Documentation**: Browse the user guide and API reference
- **Examples**: Check the examples directory
- **Issues**: Report bugs on GitHub Issues
- **Discussions**: Join community discussions

## Contributing

We welcome contributions! Please see our [Contributing Guide](development/contributing.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
