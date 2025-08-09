# Agent Factory

A comprehensive platform for creating, orchestrating, and managing AI agents using LangGraph/LangChain frameworks with support for multiple LLM providers, database connectivity, and RAG capabilities.

## 🚀 Features

### Core Capabilities
- **Multi-Agent Orchestration**: Built on LangGraph/LangChain for sophisticated workflow management
- **Multiple LLM Providers**: OpenAI, Ollama (local), and OCI GenAI Service support
- **Database Connectivity**: Oracle 23ai with AI Vector Search and PostgreSQL with pgvector
- **RAG System**: Complete Retrieval-Augmented Generation with source references
- **Interactive Workflows**: Text-based interaction with deployed agentic workflows
- **Professional Architecture**: Modular, testable, and production-ready code structure

### LLM Provider Support
- **OpenAI**: GPT-4, GPT-3.5-turbo, embedding models
- **Ollama**: Local model execution (Llama 2, Mistral, etc.)
- **OCI GenAI**: Oracle Cloud Infrastructure Generative AI Service

### Database Integration
- **Oracle 23ai**: AI Vector Search, JSON support, advanced analytics
- **PostgreSQL**: pgvector extension for vector similarity search
- **Vector Operations**: Embedding storage, similarity search, hybrid queries

### RAG (Retrieval-Augmented Generation)
- **Document Processing**: PDF, DOCX, TXT, MD, HTML, JSON support
- **Vector Stores**: ChromaDB, FAISS, database-backed options
- **Source References**: Transparent citation of retrieved sources
- **Interactive RAG**: Conversational interface with context retention
- **Production Ready**: Docker and Kubernetes deployment configurations

## 🏗️ Architecture

### Components

1. **Agent Builder**
   - Agent Spec: Define agent name, role, instructions, model profile, tool list
   - Tool Spec: Configure tool name, input/output schemas, timeout, rate limiting
   - Connection Spec: Manage external service connections

2. **Orchestration Runtime**
   - LangGraph/LangChain workflow framework
   - State management and edge definition
   - Workflow execution engine

3. **UI Component**
   - StreamLit-based dashboard
   - Real-time agent creation visualization
   - Workflow monitoring and execution tracking

4. **API Component**
   - FastAPI REST endpoints
   - Authentication and authorization
   - Rate limiting and monitoring

## 📁 Project Structure

```
agent-factory/
├── src/agent_factory/           # Main source code
│   ├── agent_builder/          # Agent creation components
│   ├── orchestration/          # Workflow orchestration
│   ├── ui/                     # StreamLit UI components
│   ├── api/                    # FastAPI endpoints
│   ├── core/                   # Core utilities and models
│   └── __init__.py
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── e2e/                    # End-to-end tests
├── docs/                       # Documentation
│   ├── api/                    # API documentation
│   ├── user-guide/             # User guides
│   └── development/            # Development docs
├── docker/                     # Docker configurations
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── docker-compose.dev.yml
├── kubernetes/                 # Kubernetes manifests
│   ├── base/                   # Base configurations
│   └── overlays/               # Environment overlays
├── scripts/                    # Utility scripts
├── requirements/               # Dependency files
└── pyproject.toml             # Project configuration
```

## 🛠️ Development Setup

### Prerequisites

- Python 3.9+
- Docker (optional)
- Kubernetes (optional)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/agent-factory/agent-factory.git
cd agent-factory
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e ".[dev]"
```

### Running the Application

#### API Server
```bash
uvicorn agent_factory.api.main:app --reload
```

#### UI Dashboard
```bash
streamlit run src/agent_factory/ui/main.py
```

#### Development with Docker
```bash
docker-compose -f docker/docker-compose.dev.yml up
```

## 🧪 Testing

Run the test suite:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=src/agent_factory --cov-report=html
```

## 📚 Documentation

Build the documentation:
```bash
mkdocs serve
```

## 🚀 Deployment

### Docker
```bash
docker build -f docker/Dockerfile -t agent-factory .
docker run -p 8000:8000 agent-factory
```

### Kubernetes
```bash
kubectl apply -k kubernetes/overlays/production
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- Documentation: [https://agent-factory.readthedocs.io/](https://agent-factory.readthedocs.io/)
- Issues: [GitHub Issues](https://github.com/agent-factory/agent-factory/issues)
- Discussions: [GitHub Discussions](https://github.com/agent-factory/agent-factory/discussions)
Repository for building an Agent Factory
