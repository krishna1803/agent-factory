#!/bin/bash

# Agent Factory Complete Setup and Demo Script
# This script sets up the environment and runs a comprehensive demo

set -e

echo "🚀 Agent Factory Complete Setup and Demo"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if Python 3.9+ is available
check_python() {
    print_header "Checking Python version..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 9 ]; then
            print_status "Python $PYTHON_VERSION found ✅"
            PYTHON_CMD="python3"
        else
            print_error "Python 3.9+ required, found $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 not found"
        exit 1
    fi
}

# Setup virtual environment
setup_venv() {
    print_header "Setting up virtual environment..."
    
    if [ ! -d "venv" ]; then
        $PYTHON_CMD -m venv venv
        print_status "Virtual environment created"
    else
        print_status "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    print_status "Virtual environment activated"
    
    # Upgrade pip
    pip install --upgrade pip
}

# Install dependencies
install_dependencies() {
    print_header "Installing dependencies..."
    
    # Install main package in development mode
    pip install -e .
    print_status "Main package installed"
    
    # Install development dependencies
    pip install -e ".[dev]"
    print_status "Development dependencies installed"
    
    # Install optional dependencies for full functionality
    print_status "Installing optional dependencies..."
    
    # Document processing
    pip install PyPDF2 python-docx beautifulsoup4 html2text || print_warning "Document processing dependencies failed"
    
    # Vector stores
    pip install chromadb faiss-cpu || print_warning "Vector store dependencies failed"
    
    # Additional utilities
    pip install numpy pandas || print_warning "Utility dependencies failed"
    
    print_status "Dependencies installation completed"
}

# Setup environment variables
setup_environment() {
    print_header "Setting up environment variables..."
    
    # Create .env file if it doesn't exist
    if [ ! -f ".env" ]; then
        cp .env.template .env 2>/dev/null || cat > .env << EOF
# Agent Factory Environment Configuration

# API Configuration
API_HOST=localhost
API_PORT=8000
DEBUG=true

# Database Configuration (Optional)
# POSTGRES_CONNECTION_STRING=postgresql://user:password@localhost:5432/agentfactory
# ORACLE_CONNECTION_STRING=oracle://user:password@localhost:1521/XEPDB1

# LLM Provider Configuration (Optional)
# OPENAI_API_KEY=your-openai-api-key
# OCI_CONFIG_FILE=~/.oci/config
# OCI_COMPARTMENT_ID=your-compartment-id

# Vector Store Configuration
VECTOR_STORE_TYPE=chroma
VECTOR_STORE_PATH=./data/vectorstore

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/agent_factory.log
EOF
        print_status "Environment file created (.env)"
    else
        print_status "Environment file already exists"
    fi
    
    # Create necessary directories
    mkdir -p data/vectorstore
    mkdir -p logs
    mkdir -p temp/uploads
    
    print_status "Directory structure created"
}

# Check optional services
check_optional_services() {
    print_header "Checking optional services..."
    
    # Check Ollama
    if command -v ollama &> /dev/null; then
        print_status "Ollama found ✅"
        if ollama list | grep -q "llama2"; then
            print_status "Llama2 model available ✅"
        else
            print_warning "Llama2 model not found. Run: ollama pull llama2"
        fi
    else
        print_warning "Ollama not found. Install from: https://ollama.ai"
    fi
    
    # Check PostgreSQL
    if command -v psql &> /dev/null; then
        print_status "PostgreSQL client found ✅"
    else
        print_warning "PostgreSQL client not found"
    fi
    
    # Check Docker
    if command -v docker &> /dev/null; then
        print_status "Docker found ✅"
    else
        print_warning "Docker not found"
    fi
}

# Run tests
run_tests() {
    print_header "Running tests..."
    
    # Run quick tests
    python -m pytest tests/ -v --tb=short -x
    
    if [ $? -eq 0 ]; then
        print_status "Tests passed ✅"
    else
        print_warning "Some tests failed. This is normal if optional dependencies are missing."
    fi
}

# Start services
start_services() {
    print_header "Starting Agent Factory services..."
    
    # Check if services are already running
    if lsof -i:8000 &> /dev/null; then
        print_warning "Port 8000 already in use. Stopping existing service..."
        pkill -f "uvicorn.*agent_factory" || true
        sleep 2
    fi
    
    # Start API server in background
    print_status "Starting API server..."
    uvicorn agent_factory.api.main:app --host localhost --port 8000 --reload &
    API_PID=$!
    echo $API_PID > .api.pid
    
    # Wait for API to start
    sleep 5
    
    # Check if API is running
    if curl -s http://localhost:8000/health > /dev/null; then
        print_status "API server started ✅ (http://localhost:8000)"
    else
        print_error "Failed to start API server"
        return 1
    fi
    
    # Start StreamLit UI in background
    print_status "Starting StreamLit UI..."
    streamlit run src/agent_factory/ui/main.py --server.port 8501 --server.headless true &
    UI_PID=$!
    echo $UI_PID > .ui.pid
    
    # Wait for UI to start
    sleep 5
    
    print_status "StreamLit UI started ✅ (http://localhost:8501)"
    
    # Start interactive UI
    print_status "Starting Interactive UI..."
    streamlit run src/agent_factory/ui/interactive.py --server.port 8502 --server.headless true &
    INTERACTIVE_PID=$!
    echo $INTERACTIVE_PID > .interactive.pid
    
    sleep 3
    print_status "Interactive UI started ✅ (http://localhost:8502)"
}

# Run comprehensive demo
run_demo() {
    print_header "Running comprehensive demo..."
    
    # Run the demo script
    python examples/comprehensive_demo.py
    
    if [ $? -eq 0 ]; then
        print_status "Demo completed successfully ✅"
    else
        print_warning "Demo completed with warnings. Check output above."
    fi
}

# Display final information
show_summary() {
    print_header "Setup Complete! 🎉"
    
    echo ""
    echo "Agent Factory is now running with the following services:"
    echo ""
    echo "📡 API Server:      http://localhost:8000"
    echo "   📚 API Docs:     http://localhost:8000/docs"
    echo "   💚 Health Check: http://localhost:8000/health"
    echo ""
    echo "🖥️  Main UI:        http://localhost:8501"
    echo "💬 Interactive UI: http://localhost:8502"
    echo ""
    echo "🔧 Configuration:"
    echo "   📁 Environment:  .env"
    echo "   📁 Data:         ./data/"
    echo "   📁 Logs:         ./logs/"
    echo ""
    echo "📖 Next Steps:"
    echo "   1. Configure LLM providers in .env file"
    echo "   2. Set up database connections (optional)"
    echo "   3. Visit the web interfaces to start creating agents"
    echo "   4. Check out examples/ directory for code samples"
    echo ""
    echo "🛑 To stop services:"
    echo "   ./scripts/stop.sh"
    echo ""
    echo "📚 Documentation:"
    echo "   - README.md for detailed setup"
    echo "   - docs/ directory for technical documentation"
    echo "   - examples/ directory for usage examples"
}

# Cleanup function
cleanup() {
    if [ -f ".api.pid" ]; then
        kill $(cat .api.pid) 2>/dev/null || true
        rm .api.pid
    fi
    if [ -f ".ui.pid" ]; then
        kill $(cat .ui.pid) 2>/dev/null || true
        rm .ui.pid
    fi
    if [ -f ".interactive.pid" ]; then
        kill $(cat .interactive.pid) 2>/dev/null || true
        rm .interactive.pid
    fi
}

# Handle script interruption
trap cleanup EXIT

# Main execution
main() {
    echo "Starting Agent Factory setup..."
    echo ""
    
    check_python
    setup_venv
    install_dependencies
    setup_environment
    check_optional_services
    
    # Ask user if they want to run tests
    echo ""
    read -p "Run tests? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_tests
    fi
    
    # Ask user if they want to start services
    echo ""
    read -p "Start services? (Y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        start_services
        
        # Ask if they want to run demo
        echo ""
        read -p "Run comprehensive demo? (y/N): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            run_demo
        fi
        
        show_summary
        
        # Keep services running
        echo "Press Ctrl+C to stop all services..."
        wait
    else
        print_status "Services not started. Run './scripts/run.sh' to start them later."
    fi
}

# Run main function
main "$@"
