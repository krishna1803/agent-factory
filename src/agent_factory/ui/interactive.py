"""
Enhanced StreamLit UI for interactive workflow execution.

This module provides UI components for interacting with deployed
agentic workflows through textual prompts, supporting RAG workflows
with reference displays.
"""

import streamlit as st
import asyncio
import json
import requests
from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import UUID, uuid4

# Configure page
st.set_page_config(
    page_title="Agent Factory - Interactive",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API base URL (should be configurable)
API_BASE_URL = "http://localhost:8000/api/v1"

class InteractiveUI:
    """UI class for interactive workflow execution."""
    
    def __init__(self):
        """Initialize the interactive UI."""
        self.session_state_keys = [
            "deployed_workflows",
            "active_sessions",
            "current_session",
            "conversation_history"
        ]
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state variables."""
        for key in self.session_state_keys:
            if key not in st.session_state:
                if key == "deployed_workflows":
                    st.session_state[key] = {}
                elif key == "active_sessions":
                    st.session_state[key] = {}
                elif key == "current_session":
                    st.session_state[key] = None
                elif key == "conversation_history":
                    st.session_state[key] = []
    
    def render_main_interface(self):
        """Render the main interactive interface."""
        st.title("💬 Interactive Agent Workflows")
        st.markdown("*Chat with deployed agentic workflows and get intelligent responses*")
        
        # Create tabs for different functionalities
        tab1, tab2, tab3, tab4 = st.tabs([
            "🚀 Deploy Workflows", 
            "💬 Chat Interface", 
            "📊 Session Management",
            "🔍 RAG & Database Tools"
        ])
        
        with tab1:
            self.render_deployment_tab()
        
        with tab2:
            self.render_chat_tab()
        
        with tab3:
            self.render_session_management_tab()
        
        with tab4:
            self.render_tools_tab()
    
    def render_deployment_tab(self):
        """Render workflow deployment interface."""
        st.header("Deploy Workflows for Interactive Use")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Workflow Configuration")
            
            # Workflow selection/creation
            workflow_option = st.radio(
                "Choose workflow source:",
                ["Upload JSON", "Create New", "Select Existing"]
            )
            
            workflow_def = None
            
            if workflow_option == "Upload JSON":
                uploaded_file = st.file_uploader(
                    "Upload workflow definition (JSON)",
                    type=["json"],
                    help="Upload a JSON file containing a workflow definition"
                )
                
                if uploaded_file:
                    try:
                        workflow_data = json.load(uploaded_file)
                        workflow_def = self._parse_workflow_definition(workflow_data)
                        st.success("Workflow definition loaded successfully!")
                        st.json(workflow_data)
                    except Exception as e:
                        st.error(f"Error parsing workflow: {str(e)}")
            
            elif workflow_option == "Create New":
                workflow_def = self._render_workflow_creator()
            
            elif workflow_option == "Select Existing":
                # This would integrate with the main workflow management
                st.info("Integration with existing workflows coming soon...")
            
            # Deployment configuration
            if workflow_def:
                st.subheader("Deployment Configuration")
                
                config = {}
                
                # RAG configuration
                if self._is_rag_workflow(workflow_def):
                    st.markdown("**RAG Configuration**")
                    config["rag_enabled"] = True
                    config["vector_store_type"] = st.selectbox(
                        "Vector Store Type",
                        ["chroma", "faiss", "database"]
                    )
                    
                    if config["vector_store_type"] == "database":
                        config["db_connection"] = st.text_input("Database Connection Name")
                        config["vector_table"] = st.text_input("Vector Table Name")
                
                # Database configuration
                if self._is_database_workflow(workflow_def):
                    st.markdown("**Database Configuration**")
                    config["db_enabled"] = True
                    config["auto_connect"] = st.checkbox("Auto-connect on session start")
                
                # Performance configuration
                st.markdown("**Performance Settings**")
                config["max_concurrent_sessions"] = st.number_input(
                    "Max Concurrent Sessions", 
                    min_value=1, 
                    max_value=100, 
                    value=10
                )
                config["session_timeout_hours"] = st.number_input(
                    "Session Timeout (hours)", 
                    min_value=1, 
                    max_value=168, 
                    value=24
                )
                
                # Deploy button
                if st.button("🚀 Deploy Workflow", type="primary"):
                    success = self._deploy_workflow(workflow_def, config)
                    if success:
                        st.success(f"Workflow '{workflow_def['name']}' deployed successfully!")
                        st.rerun()
        
        with col2:
            self.render_deployed_workflows_sidebar()
    
    def render_chat_tab(self):
        """Render the chat interface."""
        st.header("Chat with Deployed Workflows")
        
        # Session selection/creation
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.session_state.deployed_workflows:
                workflow_ids = list(st.session_state.deployed_workflows.keys())
                workflow_names = [
                    st.session_state.deployed_workflows[wid]["name"] 
                    for wid in workflow_ids
                ]
                
                selected_workflow = st.selectbox(
                    "Select Workflow",
                    options=workflow_ids,
                    format_func=lambda x: st.session_state.deployed_workflows[x]["name"]
                )
            else:
                st.warning("No workflows deployed. Please deploy a workflow first.")
                return
        
        with col2:
            if st.button("🆕 New Session"):
                session_id = self._create_new_session(selected_workflow)
                if session_id:
                    st.session_state.current_session = session_id
                    st.success(f"Created session: {session_id[:8]}...")
                    st.rerun()
        
        with col3:
            if st.session_state.active_sessions:
                current_session = st.selectbox(
                    "Active Sessions",
                    options=list(st.session_state.active_sessions.keys()),
                    format_func=lambda x: f"{x[:8]}..."
                )
                if current_session != st.session_state.current_session:
                    st.session_state.current_session = current_session
                    self._load_conversation_history(current_session)
                    st.rerun()
        
        # Chat interface
        if st.session_state.current_session:
            self.render_chat_interface()
        else:
            st.info("Create or select a session to start chatting.")
    
    def render_chat_interface(self):
        """Render the main chat interface."""
        session_id = st.session_state.current_session
        
        # Chat history display
        chat_container = st.container()
        
        with chat_container:
            st.subheader(f"Conversation - Session: {session_id[:8]}...")
            
            # Display conversation history
            for i, interaction in enumerate(st.session_state.conversation_history):
                with st.chat_message("user"):
                    st.write(interaction["user_input"])
                
                with st.chat_message("assistant"):
                    st.write(interaction["agent_response"])
                    
                    # Show metadata if available
                    if interaction.get("metadata"):
                        with st.expander("📊 Response Details"):
                            st.json(interaction["metadata"])
                
                # Show sources for RAG responses
                if interaction.get("sources"):
                    with st.expander(f"📚 Sources ({len(interaction['sources'])})"):
                        for j, source in enumerate(interaction["sources"]):
                            st.markdown(f"**Source {j+1}:**")
                            st.markdown(source.get("content", "")[:300] + "...")
                            if source.get("metadata"):
                                st.caption(f"From: {source['metadata'].get('source', 'Unknown')}")
        
        # Chat input
        st.divider()
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_area(
                "Your message:",
                height=100,
                placeholder="Type your message here..."
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            
            include_sources = st.checkbox("Include sources", value=True)
            
            if st.button("📤 Send", type="primary"):
                if user_input.strip():
                    self._send_message(session_id, user_input, include_sources)
                    st.rerun()
                else:
                    st.warning("Please enter a message.")
    
    def render_session_management_tab(self):
        """Render session management interface."""
        st.header("Session Management")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Active Sessions")
            
            if st.session_state.active_sessions:
                for session_id, session_info in st.session_state.active_sessions.items():
                    with st.expander(f"Session: {session_id[:8]}... - {session_info.get('workflow_name', 'Unknown')}"):
                        st.write(f"**Created:** {session_info.get('created_at', 'Unknown')}")
                        st.write(f"**Last Activity:** {session_info.get('last_activity', 'Unknown')}")
                        st.write(f"**Interactions:** {session_info.get('interaction_count', 0)}")
                        
                        col1_inner, col2_inner = st.columns(2)
                        
                        with col1_inner:
                            if st.button(f"📋 View History", key=f"history_{session_id}"):
                                self._load_conversation_history(session_id)
                                st.session_state.current_session = session_id
                                st.rerun()
                        
                        with col2_inner:
                            if st.button(f"🗑️ Close Session", key=f"close_{session_id}"):
                                self._close_session(session_id)
                                st.rerun()
            else:
                st.info("No active sessions.")
        
        with col2:
            st.subheader("Cleanup Tools")
            
            max_age = st.number_input(
                "Cleanup sessions older than (hours):",
                min_value=1,
                max_value=168,
                value=24
            )
            
            if st.button("🧹 Cleanup Inactive Sessions"):
                cleaned_count = self._cleanup_sessions(max_age)
                st.success(f"Cleaned up {cleaned_count} inactive sessions")
                st.rerun()
    
    def render_tools_tab(self):
        """Render RAG and database tools interface."""
        st.header("RAG & Database Tools")
        
        tab1, tab2 = st.tabs(["📚 RAG Tools", "🗄️ Database Tools"])
        
        with tab1:
            self.render_rag_tools()
        
        with tab2:
            self.render_database_tools()
    
    def render_rag_tools(self):
        """Render RAG-specific tools."""
        st.subheader("Document Indexing and RAG Management")
        
        # Document indexing
        st.markdown("**Index Documents**")
        
        workflow_id = st.selectbox(
            "Select RAG-enabled workflow:",
            options=[
                wid for wid, info in st.session_state.deployed_workflows.items()
                if info.get("rag_enabled", False)
            ],
            format_func=lambda x: st.session_state.deployed_workflows[x]["name"]
        )
        
        if workflow_id:
            uploaded_files = st.file_uploader(
                "Upload documents to index:",
                accept_multiple_files=True,
                type=["txt", "pdf", "docx", "md", "json"]
            )
            
            if uploaded_files and st.button("📚 Index Documents"):
                # Save uploaded files temporarily and index them
                file_paths = []
                for uploaded_file in uploaded_files:
                    # In a real implementation, save to temporary directory
                    file_paths.append(uploaded_file.name)
                
                success = self._index_documents(workflow_id, file_paths)
                if success:
                    st.success(f"Indexed {len(file_paths)} documents successfully!")
        
        # Direct RAG query
        st.divider()
        st.markdown("**Direct RAG Query**")
        
        pipeline_name = st.text_input("Pipeline Name:")
        rag_query = st.text_area("Query:", placeholder="Enter your question...")
        max_docs = st.slider("Max documents to retrieve:", 1, 20, 5)
        
        if st.button("🔍 Query RAG Pipeline") and pipeline_name and rag_query:
            result = self._query_rag_pipeline(pipeline_name, rag_query, max_docs)
            if result:
                st.success("Query executed successfully!")
                st.write("**Response:**")
                st.write(result["response"])
                
                if result["sources"]:
                    st.write("**Sources:**")
                    for i, source in enumerate(result["sources"]):
                        with st.expander(f"Source {i+1}"):
                            st.write(source["content"])
                            if source.get("metadata"):
                                st.caption(f"From: {source['metadata'].get('source', 'Unknown')}")
    
    def render_database_tools(self):
        """Render database tools."""
        st.subheader("Database Query and Management")
        
        # Database query interface
        connection_name = st.text_input("Database Connection Name:")
        
        query_type = st.radio("Query Type:", ["SQL Query", "Vector Search"])
        
        if query_type == "SQL Query":
            sql_query = st.text_area(
                "SQL Query:",
                placeholder="SELECT * FROM table_name LIMIT 10;"
            )
            
            if st.button("🔍 Execute Query") and connection_name and sql_query:
                result = self._execute_database_query(connection_name, sql_query)
                if result:
                    st.success(f"Query executed successfully! ({result['row_count']} rows)")
                    if result["results"]:
                        st.dataframe(result["results"])
        
        elif query_type == "Vector Search":
            search_query = st.text_input("Search Query:")
            table_name = st.text_input("Table Name:")
            limit = st.number_input("Limit:", min_value=1, max_value=100, value=10)
            
            if st.button("🔍 Vector Search") and all([connection_name, search_query, table_name]):
                # This would require vector embedding of the search query
                st.info("Vector search functionality to be implemented")
        
        # Database health check
        st.divider()
        st.markdown("**Database Health Check**")
        
        health_connection = st.text_input("Connection Name for Health Check:")
        
        if st.button("🏥 Check Health") and health_connection:
            is_healthy = self._check_database_health(health_connection)
            if is_healthy:
                st.success("Database connection is healthy! ✅")
            else:
                st.error("Database connection is unhealthy! ❌")
    
    def render_deployed_workflows_sidebar(self):
        """Render sidebar with deployed workflows."""
        st.subheader("Deployed Workflows")
        
        if st.session_state.deployed_workflows:
            for workflow_id, workflow_info in st.session_state.deployed_workflows.items():
                with st.expander(f"🤖 {workflow_info['name']}"):
                    st.write(f"**Type:** {workflow_info.get('type', 'Standard')}")
                    st.write(f"**RAG Enabled:** {'✅' if workflow_info.get('rag_enabled') else '❌'}")
                    st.write(f"**DB Enabled:** {'✅' if workflow_info.get('db_enabled') else '❌'}")
                    st.write(f"**Sessions:** {workflow_info.get('active_sessions', 0)}")
                    
                    if st.button(f"📊 View Details", key=f"details_{workflow_id}"):
                        self._show_workflow_details(workflow_id)
        else:
            st.info("No workflows deployed yet.")
    
    # Helper methods
    
    def _parse_workflow_definition(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse workflow definition from JSON data."""
        # This would parse and validate the workflow definition
        return workflow_data
    
    def _render_workflow_creator(self) -> Optional[Dict[str, Any]]:
        """Render simple workflow creator."""
        st.info("Quick workflow creator - for demo purposes")
        
        name = st.text_input("Workflow Name:")
        description = st.text_area("Description:")
        
        workflow_type = st.selectbox(
            "Workflow Type:",
            ["Standard", "RAG", "Database", "RAG + Database"]
        )
        
        if name and description:
            return {
                "name": name,
                "description": description,
                "type": workflow_type,
                "id": str(uuid4())
            }
        
        return None
    
    def _is_rag_workflow(self, workflow_def: Dict[str, Any]) -> bool:
        """Check if workflow is RAG-enabled."""
        return "rag" in workflow_def.get("type", "").lower()
    
    def _is_database_workflow(self, workflow_def: Dict[str, Any]) -> bool:
        """Check if workflow is database-enabled."""
        return "database" in workflow_def.get("type", "").lower()
    
    def _deploy_workflow(self, workflow_def: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """Deploy a workflow."""
        try:
            # Simulate API call to deploy workflow
            workflow_id = workflow_def["id"]
            
            # Store in session state
            st.session_state.deployed_workflows[workflow_id] = {
                "name": workflow_def["name"],
                "type": workflow_def.get("type", "Standard"),
                "rag_enabled": "rag" in workflow_def.get("type", "").lower(),
                "db_enabled": "database" in workflow_def.get("type", "").lower(),
                "deployed_at": datetime.now().isoformat(),
                "active_sessions": 0,
                "config": config
            }
            
            return True
        except Exception as e:
            st.error(f"Deployment failed: {str(e)}")
            return False
    
    def _create_new_session(self, workflow_id: str) -> Optional[str]:
        """Create a new conversation session."""
        try:
            session_id = str(uuid4())
            
            # Store in session state
            st.session_state.active_sessions[session_id] = {
                "workflow_id": workflow_id,
                "workflow_name": st.session_state.deployed_workflows[workflow_id]["name"],
                "created_at": datetime.now().isoformat(),
                "last_activity": datetime.now().isoformat(),
                "interaction_count": 0
            }
            
            # Update workflow session count
            st.session_state.deployed_workflows[workflow_id]["active_sessions"] += 1
            
            return session_id
        except Exception as e:
            st.error(f"Failed to create session: {str(e)}")
            return None
    
    def _send_message(self, session_id: str, message: str, include_sources: bool = True):
        """Send a message to the workflow."""
        try:
            # Simulate API call to process prompt
            # In real implementation, this would call the interactive API
            
            # Mock response
            response = f"Thank you for your message: '{message}'. This is a simulated response from the workflow."
            
            sources = []
            if include_sources and "rag" in st.session_state.active_sessions[session_id]["workflow_name"].lower():
                sources = [
                    {
                        "content": "This is a sample source document that would be retrieved for RAG workflows...",
                        "metadata": {"source": "sample_document.pdf", "page": 1}
                    }
                ]
            
            # Add to conversation history
            interaction = {
                "user_input": message,
                "agent_response": response,
                "sources": sources,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "execution_time": 0.5,
                    "include_sources": include_sources
                }
            }
            
            st.session_state.conversation_history.append(interaction)
            
            # Update session info
            st.session_state.active_sessions[session_id]["last_activity"] = datetime.now().isoformat()
            st.session_state.active_sessions[session_id]["interaction_count"] += 1
            
        except Exception as e:
            st.error(f"Failed to send message: {str(e)}")
    
    def _load_conversation_history(self, session_id: str):
        """Load conversation history for a session."""
        # In real implementation, this would fetch from API
        # For now, maintain in session state
        pass
    
    def _close_session(self, session_id: str):
        """Close a conversation session."""
        if session_id in st.session_state.active_sessions:
            # Update workflow session count
            workflow_id = st.session_state.active_sessions[session_id]["workflow_id"]
            if workflow_id in st.session_state.deployed_workflows:
                st.session_state.deployed_workflows[workflow_id]["active_sessions"] -= 1
            
            # Remove session
            del st.session_state.active_sessions[session_id]
            
            # Clear current session if it was the closed one
            if st.session_state.current_session == session_id:
                st.session_state.current_session = None
                st.session_state.conversation_history = []
    
    def _cleanup_sessions(self, max_age_hours: int) -> int:
        """Cleanup inactive sessions."""
        # Simulate cleanup - in real implementation would call API
        return 0
    
    def _index_documents(self, workflow_id: str, file_paths: List[str]) -> bool:
        """Index documents for RAG workflow."""
        # Simulate document indexing
        return True
    
    def _query_rag_pipeline(self, pipeline_name: str, query: str, max_docs: int) -> Optional[Dict[str, Any]]:
        """Query RAG pipeline directly."""
        # Simulate RAG query
        return {
            "query": query,
            "response": f"This is a simulated RAG response for query: '{query}'",
            "sources": [
                {
                    "content": "Sample source content that would be retrieved...",
                    "metadata": {"source": "sample.pdf"}
                }
            ]
        }
    
    def _execute_database_query(self, connection_name: str, query: str) -> Optional[Dict[str, Any]]:
        """Execute database query."""
        # Simulate database query
        return {
            "results": [{"column1": "value1", "column2": "value2"}],
            "row_count": 1,
            "execution_time": 0.1
        }
    
    def _check_database_health(self, connection_name: str) -> bool:
        """Check database health."""
        # Simulate health check
        return True
    
    def _show_workflow_details(self, workflow_id: str):
        """Show detailed workflow information."""
        workflow_info = st.session_state.deployed_workflows.get(workflow_id)
        if workflow_info:
            st.json(workflow_info)


def main():
    """Main application function."""
    ui = InteractiveUI()
    ui.render_main_interface()


if __name__ == "__main__":
    main()
