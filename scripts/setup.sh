#!/bin/bash

# Development setup script for Agent Factory

set -e

echo "🚀 Setting up Agent Factory development environment..."

# Check if Python 3.9+ is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
REQUIRED_VERSION="3.9"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python $REQUIRED_VERSION or higher is required. Found $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python $PYTHON_VERSION found"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📈 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📋 Installing dependencies..."
pip install -e ".[dev]"

# Set up pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file..."
    cat > .env << EOF
# Agent Factory Environment Variables
DEBUG=true
LOG_LEVEL=DEBUG
SECRET_KEY=dev-secret-key-change-in-production
DATABASE_URL=sqlite:///./agent_factory_dev.db
REDIS_URL=redis://localhost:6379
EOF
fi

# Create logs directory
mkdir -p logs

echo "✅ Development environment setup complete!"
echo ""
echo "🚀 Quick start commands:"
echo "  source venv/bin/activate          # Activate virtual environment"
echo "  uvicorn agent_factory.api.main:app --reload  # Start API server"
echo "  streamlit run src/agent_factory/ui/main.py   # Start UI"
echo "  pytest                            # Run tests"
echo "  pre-commit run --all-files        # Run code quality checks"
echo ""
echo "📚 Documentation:"
echo "  API docs: http://localhost:8000/docs"
echo "  UI app: http://localhost:8501"
