#!/usr/bin/env python3
"""
Comprehensive example demonstrating Agent Factory capabilities.

This script showcases:
1. Creating agents with different LLM providers (OpenAI, Ollama, OCI GenAI)
2. Setting up database connections (Oracle 23ai, PostgreSQL)
3. Creating RAG-enabled workflows
4. Interactive prompt processing with reference returns
5. End-to-end workflow execution
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, Any, List
from uuid import uuid4

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Agent Factory imports
from agent_factory.core.models import (
    AgentSpec, ModelProfile, ToolSpec, ConnectionSpec, WorkflowDefinition,
    AgentRole, ModelProvider, ToolType, ConnectionType
)
from agent_factory.providers.llm import LLMProviderFactory, LLMProviderManager
from agent_factory.connections.database import DatabaseManager
from agent_factory.rag.pipeline import RAGManager, RAGPipeline
from agent_factory.orchestration.langgraph_integration import LangGraphOrchestrator
from agent_factory.interactive.prompt_engine import InteractiveWorkflowEngine


class AgentFactoryDemo:
    """Comprehensive demonstration of Agent Factory capabilities."""
    
    def __init__(self):
        """Initialize the demo."""
        self.llm_manager = LLMProviderManager()
        self.db_manager = DatabaseManager()
        self.rag_manager = RAGManager()
        self.orchestrator = LangGraphOrchestrator()
        self.interactive_engine = InteractiveWorkflowEngine()
        
        self.demo_agents = []
        self.demo_workflows = []
        self.demo_sessions = []
    
    async def run_comprehensive_demo(self):
        """Run the comprehensive demonstration."""
        print("🚀 Starting Agent Factory Comprehensive Demo")
        print("=" * 60)
        
        try:
            # 1. LLM Provider Setup
            await self.demo_llm_providers()
            
            # 2. Database Connectivity
            await self.demo_database_connectivity()
            
            # 3. RAG System Setup
            await self.demo_rag_system()
            
            # 4. Agent Creation
            await self.demo_agent_creation()
            
            # 5. Workflow Orchestration
            await self.demo_workflow_orchestration()
            
            # 6. Interactive Prompt Processing
            await self.demo_interactive_prompts()
            
            print("✅ Demo completed successfully!")
            
        except Exception as e:
            logger.error(f"Demo failed: {str(e)}")
            raise
    
    async def demo_llm_providers(self):
        """Demonstrate LLM provider capabilities."""
        print("\n📡 LLM Provider Demonstration")
        print("-" * 40)
        
        # OpenAI provider
        if os.getenv("OPENAI_API_KEY"):
            print("🤖 Setting up OpenAI provider...")
            openai_profile = ModelProfile(
                provider=ModelProvider.OPENAI,
                model_name="gpt-4",
                temperature=0.7,
                max_tokens=1000,
                api_key=os.getenv("OPENAI_API_KEY")
            )
            
            openai_provider = LLMProviderFactory.create_provider(openai_profile)
            await openai_provider.initialize()
            self.llm_manager.register_provider("openai", openai_provider)
            
            # Test OpenAI
            from langchain_core.messages import HumanMessage
            try:
                response = await openai_provider.generate_response([
                    HumanMessage(content="Hello! This is a test message.")
                ])
                print(f"✅ OpenAI Response: {response.content[:100]}...")
            except Exception as e:
                print(f"❌ OpenAI test failed: {str(e)}")
        else:
            print("⚠️ OpenAI API key not found, skipping OpenAI demo")
        
        # Ollama provider (if available)
        print("\n🦙 Setting up Ollama provider...")
        ollama_profile = ModelProfile(
            provider=ModelProvider.LOCAL,
            model_name="llama2",
            temperature=0.7,
            api_base="http://localhost:11434"
        )
        
        try:
            ollama_provider = LLMProviderFactory.create_provider(ollama_profile)
            await ollama_provider.initialize()
            self.llm_manager.register_provider("ollama", ollama_provider)
            print("✅ Ollama provider initialized")
        except Exception as e:
            print(f"⚠️ Ollama not available: {str(e)}")
        
        # OCI GenAI provider (if configured)
        if os.getenv("OCI_CONFIG_FILE"):
            print("\n☁️ Setting up OCI GenAI provider...")
            oci_profile = ModelProfile(
                provider=ModelProvider.OPENAI,  # Using as fallback
                model_name="cohere.command",
                temperature=0.7,
                additional_config={
                    "provider": "oci_genai",
                    "config_file": os.getenv("OCI_CONFIG_FILE"),
                    "compartment_id": os.getenv("OCI_COMPARTMENT_ID")
                }
            )
            
            try:
                oci_provider = LLMProviderFactory.create_provider(oci_profile)
                await oci_provider.initialize()
                self.llm_manager.register_provider("oci", oci_provider)
                print("✅ OCI GenAI provider initialized")
            except Exception as e:
                print(f"⚠️ OCI GenAI not available: {str(e)}")
        else:
            print("⚠️ OCI configuration not found, skipping OCI demo")
        
        # Health check all providers
        health_status = await self.llm_manager.health_check_all()
        print(f"\n🏥 Provider Health Status: {health_status}")
    
    async def demo_database_connectivity(self):
        """Demonstrate database connectivity."""
        print("\n🗄️ Database Connectivity Demonstration")
        print("-" * 40)
        
        # PostgreSQL with pgvector
        if os.getenv("POSTGRES_CONNECTION_STRING"):
            print("🐘 Setting up PostgreSQL connection...")
            postgres_spec = ConnectionSpec(
                name="demo_postgres",
                description="Demo PostgreSQL connection with pgvector",
                connection_type=ConnectionType.DATABASE,
                endpoint=os.getenv("POSTGRES_CONNECTION_STRING"),
                credentials={
                    "username": os.getenv("POSTGRES_USER", "postgres"),
                    "password": os.getenv("POSTGRES_PASSWORD", "")
                },
                configuration={
                    "database_type": "postgresql",
                    "host": os.getenv("POSTGRES_HOST", "localhost"),
                    "port": int(os.getenv("POSTGRES_PORT", "5432")),
                    "database": os.getenv("POSTGRES_DB", "postgres"),
                    "pool_size": 5
                }
            )
            
            try:
                self.db_manager.register_connection("postgres", postgres_spec)
                connection = await self.db_manager.get_connection("postgres")
                
                # Test connection
                result = await connection.execute_query("SELECT version()")
                print(f"✅ PostgreSQL connected: {result[0] if result else 'OK'}")
                
                # Create vector table for demo
                await connection.create_vector_table("demo_documents", dimension=1536)
                print("✅ Vector table created")
                
            except Exception as e:
                print(f"❌ PostgreSQL setup failed: {str(e)}")
        else:
            print("⚠️ PostgreSQL connection string not found")
        
        # Oracle 23ai
        if os.getenv("ORACLE_CONNECTION_STRING"):
            print("\n🏛️ Setting up Oracle 23ai connection...")
            oracle_spec = ConnectionSpec(
                name="demo_oracle",
                description="Demo Oracle 23ai connection with AI Vector Search",
                connection_type=ConnectionType.DATABASE,
                endpoint=os.getenv("ORACLE_CONNECTION_STRING"),
                credentials={
                    "username": os.getenv("ORACLE_USER", "admin"),
                    "password": os.getenv("ORACLE_PASSWORD", "")
                },
                configuration={
                    "database_type": "oracle",
                    "host": os.getenv("ORACLE_HOST", "localhost"),
                    "port": int(os.getenv("ORACLE_PORT", "1521")),
                    "service_name": os.getenv("ORACLE_SERVICE", "XEPDB1"),
                    "min_connections": 1,
                    "max_connections": 5
                }
            )
            
            try:
                self.db_manager.register_connection("oracle", oracle_spec)
                connection = await self.db_manager.get_connection("oracle")
                
                # Test connection
                result = await connection.execute_query("SELECT 'Oracle 23ai Connected' FROM DUAL")
                print(f"✅ Oracle 23ai connected: {result[0] if result else 'OK'}")
                
                # Create vector table for demo
                await connection.create_vector_table("demo_docs_vector", dimension=1536)
                print("✅ AI Vector Search table created")
                
            except Exception as e:
                print(f"❌ Oracle 23ai setup failed: {str(e)}")
        else:
            print("⚠️ Oracle connection string not found")
        
        # Health check all connections
        health_status = await self.db_manager.health_check_all()
        print(f"\n🏥 Database Health Status: {health_status}")
    
    async def demo_rag_system(self):
        """Demonstrate RAG system capabilities."""
        print("\n📚 RAG System Demonstration")
        print("-" * 40)
        
        # Create sample documents
        sample_docs = [
            {
                "id": "doc1",
                "content": "Agent Factory is a comprehensive platform for creating and orchestrating AI agents using LangGraph and LangChain frameworks.",
                "metadata": {"source": "documentation.md", "type": "technical"},
                "embedding": [0.1] * 1536  # Mock embedding
            },
            {
                "id": "doc2", 
                "content": "The system supports multiple LLM providers including OpenAI, Ollama, and OCI GenAI Service for flexible model selection.",
                "metadata": {"source": "architecture.md", "type": "technical"},
                "embedding": [0.2] * 1536  # Mock embedding
            },
            {
                "id": "doc3",
                "content": "RAG (Retrieval-Augmented Generation) workflows can return source references alongside responses for enhanced transparency.",
                "metadata": {"source": "features.md", "type": "feature"},
                "embedding": [0.3] * 1536  # Mock embedding
            }
        ]
        
        # Create RAG agent spec
        rag_agent_spec = AgentSpec(
            name="demo_rag_agent",
            role="rag_specialist",
            instructions="You are a helpful assistant that uses retrieved context to answer questions accurately.",
            model_profile=ModelProfile(
                provider=ModelProvider.OPENAI,
                model_name="gpt-4",
                temperature=0.7,
                api_key=os.getenv("OPENAI_API_KEY", "demo_key")
            )
        )
        
        try:
            # Create RAG pipeline
            pipeline = await self.rag_manager.create_rag_pipeline(
                "demo_pipeline",
                rag_agent_spec,
                store_type="chroma"
            )
            
            print("✅ RAG pipeline created")
            
            # Create RAG agent
            rag_agent = await self.rag_manager.create_rag_agent(
                "demo_rag_agent",
                "demo_pipeline", 
                rag_agent_spec
            )
            
            print("✅ RAG agent created")
            
            # Simulate document indexing (in real scenario, would process actual files)
            print("📄 Indexing sample documents...")
            
            # Test RAG query
            print("🔍 Testing RAG query...")
            result = await self.rag_manager.query_agent(
                "demo_rag_agent",
                "What LLM providers does Agent Factory support?",
                session_id="demo_session"
            )
            
            print(f"✅ RAG Query Result:")
            print(f"   Response: {result['response'][:100]}...")
            print(f"   Sources: {len(result.get('sources', []))} references")
            
        except Exception as e:
            print(f"❌ RAG setup failed: {str(e)}")
    
    async def demo_agent_creation(self):
        """Demonstrate agent creation with different capabilities."""
        print("\n🤖 Agent Creation Demonstration")
        print("-" * 40)
        
        # Research Agent
        research_agent = AgentSpec(
            name="research_agent",
            role=AgentRole.RESEARCHER,
            instructions="You are a research specialist who gathers and analyzes information.",
            model_profile=ModelProfile(
                provider=ModelProvider.OPENAI,
                model_name="gpt-4",
                temperature=0.3,
                api_key=os.getenv("OPENAI_API_KEY", "demo_key")
            ),
            tool_specs=[
                ToolSpec(
                    name="web_search",
                    description="Search the web for information",
                    tool_type=ToolType.API,
                    endpoint_url="https://api.example.com/search"
                )
            ]
        )
        
        # Database Agent
        db_agent = AgentSpec(
            name="database_agent",
            role="database_specialist",
            instructions="You are a database expert who can query and analyze data.",
            model_profile=ModelProfile(
                provider=ModelProvider.OPENAI,
                model_name="gpt-4",
                temperature=0.1,
                api_key=os.getenv("OPENAI_API_KEY", "demo_key")
            ),
            connection_specs=[
                ConnectionSpec(
                    name="main_db",
                    description="Main database connection",
                    connection_type=ConnectionType.DATABASE,
                    endpoint="postgresql://localhost:5432/demo",
                    credentials={"username": "demo", "password": "demo"}
                )
            ]
        )
        
        # RAG Agent
        rag_agent = AgentSpec(
            name="rag_agent",
            role="rag_specialist",
            instructions="You are a knowledge assistant that uses retrieved context to answer questions.",
            model_profile=ModelProfile(
                provider=ModelProvider.OPENAI,
                model_name="gpt-4",
                temperature=0.7,
                api_key=os.getenv("OPENAI_API_KEY", "demo_key")
            )
        )
        
        self.demo_agents = [research_agent, db_agent, rag_agent]
        
        print(f"✅ Created {len(self.demo_agents)} demo agents:")
        for agent in self.demo_agents:
            print(f"   - {agent.name} ({agent.role})")
    
    async def demo_workflow_orchestration(self):
        """Demonstrate workflow orchestration."""
        print("\n🔄 Workflow Orchestration Demonstration")
        print("-" * 40)
        
        # Create a multi-agent workflow
        workflow_def = WorkflowDefinition(
            name="comprehensive_analysis_workflow",
            description="A workflow that combines research, database analysis, and knowledge retrieval",
            agent_specs=self.demo_agents,
            workflow_graph={
                "nodes": ["research_agent", "database_agent", "rag_agent"],
                "edges": [
                    {"from": "research_agent", "to": "database_agent"},
                    {"from": "database_agent", "to": "rag_agent"}
                ]
            },
            entry_point="research_agent"
        )
        
        try:
            # Deploy workflow
            await self.interactive_engine.deploy_workflow(workflow_def)
            print("✅ Workflow deployed successfully")
            
            # Create compiled graph
            compiled_graph = await self.orchestrator.create_langgraph_workflow(workflow_def)
            print("✅ LangGraph workflow compiled")
            
            self.demo_workflows.append(workflow_def)
            
        except Exception as e:
            print(f"❌ Workflow deployment failed: {str(e)}")
    
    async def demo_interactive_prompts(self):
        """Demonstrate interactive prompt processing."""
        print("\n💬 Interactive Prompt Demonstration")
        print("-" * 40)
        
        if not self.demo_workflows:
            print("⚠️ No workflows available for interaction")
            return
        
        workflow = self.demo_workflows[0]
        
        try:
            # Create conversation session
            session_id = await self.interactive_engine.create_session(
                workflow.id,
                user_id="demo_user"
            )
            print(f"✅ Created session: {session_id[:8]}...")
            
            # Test queries
            test_queries = [
                "What is Agent Factory and what are its main capabilities?",
                "How does the RAG system work in this platform?",
                "Can you analyze some sample data for me?",
                "What are the benefits of using multiple LLM providers?"
            ]
            
            for i, query in enumerate(test_queries, 1):
                print(f"\n🔍 Test Query {i}: {query}")
                
                result = await self.interactive_engine.process_prompt(
                    session_id,
                    query,
                    include_sources=True
                )
                
                print(f"✅ Response: {result.response[:150]}...")
                if result.sources:
                    print(f"📚 Sources: {len(result.sources)} references returned")
                print(f"⏱️ Execution time: {result.execution_time:.2f}s")
                print(f"🛤️ Agent path: {' → '.join(result.agent_path)}")
            
            # Get conversation history
            history = await self.interactive_engine.get_session_history(session_id)
            print(f"\n📋 Conversation history: {len(history)} interactions")
            
            # Close session
            await self.interactive_engine.close_session(session_id)
            print("✅ Session closed")
            
        except Exception as e:
            print(f"❌ Interactive demo failed: {str(e)}")
    
    async def demo_rag_with_references(self):
        """Demonstrate RAG with reference returns."""
        print("\n📖 RAG Reference Demonstration")
        print("-" * 40)
        
        # This would be called as part of the interactive demo
        # but separated here for clarity
        
        sample_query = "How does Agent Factory handle different LLM providers?"
        
        try:
            # Direct RAG query
            rag_result = await self.rag_manager.query_agent(
                "demo_rag_agent",
                sample_query,
                session_id="demo_rag_session"
            )
            
            print(f"Query: {sample_query}")
            print(f"Response: {rag_result['response']}")
            
            if rag_result.get('sources'):
                print("\n📚 Retrieved Sources:")
                for i, source in enumerate(rag_result['sources'], 1):
                    print(f"\nSource {i}:")
                    print(f"  Content: {source['content'][:100]}...")
                    print(f"  Metadata: {source['metadata']}")
            
        except Exception as e:
            print(f"❌ RAG reference demo failed: {str(e)}")


async def main():
    """Main demonstration function."""
    demo = AgentFactoryDemo()
    await demo.run_comprehensive_demo()


if __name__ == "__main__":
    # Set up environment variables for demo (optional)
    # os.environ["OPENAI_API_KEY"] = "your-openai-key"
    # os.environ["POSTGRES_CONNECTION_STRING"] = "postgresql://user:pass@localhost:5432/db"
    # os.environ["ORACLE_CONNECTION_STRING"] = "oracle://user:pass@localhost:1521/service"
    
    print("Agent Factory Comprehensive Demo")
    print("================================")
    print("This demo showcases all major features:")
    print("• LLM Provider Integration (OpenAI, Ollama, OCI GenAI)")
    print("• Database Connectivity (Oracle 23ai, PostgreSQL)")
    print("• RAG System with Reference Returns")
    print("• Interactive Workflow Execution")
    print("• Multi-Agent Orchestration")
    print()
    
    asyncio.run(main())
