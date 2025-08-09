"""
FastAPI main application for Agent Factory.

This module provides the main FastAPI application with all routes
and middleware configured for the Agent Factory system.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import logging

from .routers import agents, workflows, tools, connections, health, interactive
from .middleware import LoggingMiddleware, RateLimitMiddleware
from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    
    Handles startup and shutdown events for the FastAPI application.
    """
    # Startup
    logger.info("Starting Agent Factory API server...")
    
    # Initialize services here if needed
    # await initialize_database()
    # await initialize_cache()
    
    logger.info("Agent Factory API server started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Agent Factory API server...")
    
    # Cleanup services here if needed
    # await cleanup_database()
    # await cleanup_cache()
    
    logger.info("Agent Factory API server shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Agent Factory API",
    description="A comprehensive API for creating, orchestrating, and managing AI agents",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware)

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(agents.router, prefix="/api/v1/agents", tags=["agents"])
app.include_router(workflows.router, prefix="/api/v1/workflows", tags=["workflows"])
app.include_router(tools.router, prefix="/api/v1/tools", tags=["tools"])
app.include_router(connections.router, prefix="/api/v1/connections", tags=["connections"])
app.include_router(interactive.router, prefix="/api/v1/interactive", tags=["interactive"])


@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested resource was not found",
            "path": str(request.url.path)
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error", 
            "message": "An internal server error occurred"
        }
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to Agent Factory API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/api/v1/info")
async def api_info():
    """Get API information."""
    return {
        "name": "Agent Factory API",
        "version": "0.1.0",
        "description": "A comprehensive API for creating, orchestrating, and managing AI agents",
        "endpoints": {
            "agents": "/api/v1/agents",
            "workflows": "/api/v1/workflows", 
            "tools": "/api/v1/tools",
            "connections": "/api/v1/connections",
            "health": "/health"
        }
    }
