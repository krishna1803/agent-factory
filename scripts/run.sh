#!/bin/bash

# Run Agent Factory services

set -e

# Function to check if port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        return 0
    else
        return 1
    fi
}

# Function to start API server
start_api() {
    echo "🚀 Starting Agent Factory API server..."
    if check_port 8000; then
        echo "⚠️  Port 8000 is already in use"
        return 1
    fi
    
    cd "$(dirname "$0")/.."
    source venv/bin/activate
    export PYTHONPATH=$PWD/src
    uvicorn agent_factory.api.main:app --host 0.0.0.0 --port 8000 --reload &
    API_PID=$!
    echo "API_PID=$API_PID" > .api.pid
    echo "✅ API server started (PID: $API_PID)"
    echo "📊 API docs available at: http://localhost:8000/docs"
}

# Function to start UI
start_ui() {
    echo "🖥️  Starting Agent Factory UI..."
    if check_port 8501; then
        echo "⚠️  Port 8501 is already in use"
        return 1
    fi
    
    cd "$(dirname "$0")/.."
    source venv/bin/activate
    export PYTHONPATH=$PWD/src
    streamlit run src/agent_factory/ui/main.py --server.port=8501 &
    UI_PID=$!
    echo "UI_PID=$UI_PID" > .ui.pid
    echo "✅ UI started (PID: $UI_PID)"
    echo "🎨 UI available at: http://localhost:8501"
}

# Function to stop services
stop_services() {
    echo "🛑 Stopping Agent Factory services..."
    
    if [ -f .api.pid ]; then
        API_PID=$(cat .api.pid | grep API_PID | cut -d= -f2)
        if kill -0 $API_PID 2>/dev/null; then
            kill $API_PID
            echo "✅ API server stopped"
        fi
        rm .api.pid
    fi
    
    if [ -f .ui.pid ]; then
        UI_PID=$(cat .ui.pid | grep UI_PID | cut -d= -f2)
        if kill -0 $UI_PID 2>/dev/null; then
            kill $UI_PID
            echo "✅ UI stopped"
        fi
        rm .ui.pid
    fi
}

# Function to check status
check_status() {
    echo "📊 Agent Factory Status:"
    
    if check_port 8000; then
        echo "✅ API server running on port 8000"
    else
        echo "❌ API server not running"
    fi
    
    if check_port 8501; then
        echo "✅ UI running on port 8501"
    else
        echo "❌ UI not running"
    fi
}

# Handle command line arguments
case "$1" in
    "api")
        start_api
        ;;
    "ui")
        start_ui
        ;;
    "all")
        start_api
        sleep 3
        start_ui
        echo ""
        echo "🎉 Agent Factory is now running!"
        echo "📊 API: http://localhost:8000/docs"
        echo "🎨 UI:  http://localhost:8501"
        echo ""
        echo "Press Ctrl+C to stop all services"
        wait
        ;;
    "stop")
        stop_services
        ;;
    "status")
        check_status
        ;;
    "restart")
        stop_services
        sleep 2
        start_api
        sleep 3
        start_ui
        ;;
    *)
        echo "Usage: $0 {api|ui|all|stop|status|restart}"
        echo ""
        echo "Commands:"
        echo "  api     - Start only the API server"
        echo "  ui      - Start only the UI"
        echo "  all     - Start both API and UI"
        echo "  stop    - Stop all services"
        echo "  status  - Check service status"
        echo "  restart - Restart all services"
        exit 1
        ;;
esac
