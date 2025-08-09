"""
Main StreamLit application for Agent Factory UI.

This module provides a comprehensive web interface for creating,
managing, and monitoring agents and workflows.
"""

import streamlit as st
import asyncio
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import uuid4

# Import Agent Factory components
from agent_factory.core.models import (
    AgentSpec, ToolSpec, ConnectionSpec, WorkflowDefinition,
    AgentRole, ModelProvider, ToolType, ConnectionType, ModelProfile
)
from agent_factory.agent_builder import AgentBuilder
from agent_factory.orchestration import WorkflowOrchestrator

# Configure Streamlit page
st.set_page_config(
    page_title="Agent Factory",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "agent_builder" not in st.session_state:
    st.session_state.agent_builder = AgentBuilder()

if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = WorkflowOrchestrator()

if "created_agents" not in st.session_state:
    st.session_state.created_agents = []

if "created_workflows" not in st.session_state:
    st.session_state.created_workflows = []


def main():
    """Main application function."""
    st.title("🤖 Agent Factory")
    st.markdown("*Create, orchestrate, and manage AI agents with ease*")
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Navigate",
        ["Dashboard", "Agent Builder", "Tool Manager", "Connection Manager", 
         "Workflow Designer", "Execution Monitor", "Settings"]
    )
    
    # Route to selected page
    if page == "Dashboard":
        show_dashboard()
    elif page == "Agent Builder":
        show_agent_builder()
    elif page == "Tool Manager":
        show_tool_manager()
    elif page == "Connection Manager":
        show_connection_manager()
    elif page == "Workflow Designer":
        show_workflow_designer()
    elif page == "Execution Monitor":
        show_execution_monitor()
    elif page == "Settings":
        show_settings()


def show_dashboard():
    """Display the main dashboard."""
    st.header("📊 Dashboard")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Agents Created", len(st.session_state.created_agents))
    
    with col2:
        st.metric("Workflows", len(st.session_state.created_workflows))
    
    with col3:
        st.metric("Active Executions", 0)  # TODO: Get from orchestrator
    
    with col4:
        st.metric("Success Rate", "100%")  # TODO: Calculate actual rate
    
    st.divider()
    
    # Recent activity
    st.subheader("Recent Activity")
    
    if st.session_state.created_agents:
        st.write("**Recently Created Agents:**")
        for agent in st.session_state.created_agents[-5:]:
            st.write(f"• {agent.name} ({agent.role}) - {agent.created_at.strftime('%Y-%m-%d %H:%M')}")
    else:
        st.info("No agents created yet. Start by creating your first agent!")
    
    # Quick actions
    st.subheader("Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🤖 Create New Agent", use_container_width=True):
            st.switch_page("pages/agent_builder.py")
    
    with col2:
        if st.button("🔧 Manage Tools", use_container_width=True):
            st.switch_page("pages/tool_manager.py")
    
    with col3:
        if st.button("🌊 Design Workflow", use_container_width=True):
            st.switch_page("pages/workflow_designer.py")


def show_agent_builder():
    """Display the agent builder interface."""
    st.header("🤖 Agent Builder")
    
    with st.form("agent_form"):
        st.subheader("Agent Configuration")
        
        # Basic agent information
        col1, col2 = st.columns(2)
        
        with col1:
            agent_name = st.text_input("Agent Name", placeholder="e.g., Research Assistant")
            agent_role = st.selectbox("Agent Role", [role.value for role in AgentRole])
        
        with col2:
            created_by = st.text_input("Created By", placeholder="Your name")
            tags = st.text_input("Tags (comma-separated)", placeholder="research, analysis")
        
        # Instructions
        instructions = st.text_area(
            "Agent Instructions", 
            placeholder="Describe what this agent should do...",
            height=100
        )
        
        # Model configuration
        st.subheader("Model Configuration")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            model_provider = st.selectbox("Provider", [provider.value for provider in ModelProvider])
            model_name = st.text_input("Model Name", value="gpt-3.5-turbo")
        
        with col2:
            temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
            max_tokens = st.number_input("Max Tokens", min_value=1, value=1000)
        
        with col3:
            top_p = st.slider("Top P", 0.0, 1.0, 1.0, 0.1)
            context_window = st.number_input("Context Window", min_value=100, value=4000)
        
        # Submit button
        submitted = st.form_submit_button("Create Agent", use_container_width=True)
        
        if submitted:
            if agent_name and instructions:
                try:
                    # Create model profile
                    model_profile = ModelProfile(
                        provider=ModelProvider(model_provider),
                        model_name=model_name,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p
                    )
                    
                    # Create agent spec
                    agent_spec = AgentSpec(
                        name=agent_name,
                        role=agent_role,
                        instructions=instructions,
                        model_profile=model_profile,
                        created_by=created_by,
                        tags=tags.split(",") if tags else [],
                        context_window=context_window
                    )
                    
                    # Create agent
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    created_agent = loop.run_until_complete(
                        st.session_state.agent_builder.create_agent(agent_spec)
                    )
                    
                    # Store in session state
                    st.session_state.created_agents.append(created_agent)
                    
                    st.success(f"✅ Agent '{agent_name}' created successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error creating agent: {str(e)}")
            else:
                st.error("❌ Please fill in all required fields")
    
    # Display existing agents
    if st.session_state.created_agents:
        st.divider()
        st.subheader("Created Agents")
        
        for agent in st.session_state.created_agents:
            with st.expander(f"{agent.name} ({agent.role})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Instructions:** {agent.instructions}")
                    st.write(f"**Model:** {agent.model_profile.model_name}")
                    st.write(f"**Temperature:** {agent.model_profile.temperature}")
                
                with col2:
                    st.write(f"**Created:** {agent.created_at.strftime('%Y-%m-%d %H:%M')}")
                    st.write(f"**Created By:** {agent.created_by or 'Unknown'}")
                    st.write(f"**Tags:** {', '.join(agent.tags)}")


def show_tool_manager():
    """Display the tool management interface."""
    st.header("🔧 Tool Manager")
    
    tab1, tab2 = st.tabs(["Create Tool", "Manage Tools"])
    
    with tab1:
        with st.form("tool_form"):
            st.subheader("Tool Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                tool_name = st.text_input("Tool Name", placeholder="e.g., web_search")
                tool_type = st.selectbox("Tool Type", [t.value for t in ToolType])
            
            with col2:
                timeout = st.number_input("Timeout (seconds)", min_value=1, value=30)
                endpoint_url = st.text_input("Endpoint URL (if applicable)")
            
            description = st.text_area("Description", placeholder="What does this tool do?")
            
            # Schema configuration
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Input Schema")
                input_schema = st.text_area(
                    "Input Schema (JSON)", 
                    value='{"type": "object", "properties": {}}',
                    height=100
                )
            
            with col2:
                st.subheader("Output Schema")
                output_schema = st.text_area(
                    "Output Schema (JSON)",
                    value='{"type": "object", "properties": {}}',
                    height=100
                )
            
            submitted = st.form_submit_button("Create Tool", use_container_width=True)
            
            if submitted:
                if tool_name and description:
                    try:
                        # Parse schemas
                        input_schema_dict = json.loads(input_schema)
                        output_schema_dict = json.loads(output_schema)
                        
                        # Create tool spec
                        tool_spec = ToolSpec(
                            name=tool_name,
                            description=description,
                            tool_type=ToolType(tool_type),
                            input_schema=input_schema_dict,
                            output_schema=output_schema_dict,
                            timeout_seconds=timeout,
                            endpoint_url=endpoint_url if endpoint_url else None
                        )
                        
                        st.success(f"✅ Tool '{tool_name}' created successfully!")
                        st.json(tool_spec.dict())
                        
                    except json.JSONDecodeError:
                        st.error("❌ Invalid JSON in schema fields")
                    except Exception as e:
                        st.error(f"❌ Error creating tool: {str(e)}")
                else:
                    st.error("❌ Please fill in all required fields")
    
    with tab2:
        st.subheader("Existing Tools")
        st.info("Tool management functionality will be added here")


def show_connection_manager():
    """Display the connection management interface."""
    st.header("🔗 Connection Manager")
    
    with st.form("connection_form"):
        st.subheader("Connection Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            conn_name = st.text_input("Connection Name", placeholder="e.g., postgres_db")
            conn_type = st.selectbox("Connection Type", [t.value for t in ConnectionType])
        
        with col2:
            endpoint = st.text_input("Endpoint", placeholder="e.g., localhost:5432")
            timeout = st.number_input("Timeout (seconds)", min_value=1, value=30)
        
        description = st.text_area("Description", placeholder="What is this connection for?")
        
        # Credentials (simplified)
        st.subheader("Credentials")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        submitted = st.form_submit_button("Create Connection", use_container_width=True)
        
        if submitted:
            if conn_name and endpoint:
                try:
                    connection_spec = ConnectionSpec(
                        name=conn_name,
                        description=description,
                        connection_type=ConnectionType(conn_type),
                        endpoint=endpoint,
                        credentials={"username": username, "password": password} if username else {},
                        timeout_seconds=timeout
                    )
                    
                    st.success(f"✅ Connection '{conn_name}' created successfully!")
                    st.json(connection_spec.dict(exclude={"credentials"}))
                    
                except Exception as e:
                    st.error(f"❌ Error creating connection: {str(e)}")
            else:
                st.error("❌ Please fill in all required fields")


def show_workflow_designer():
    """Display the workflow designer interface."""
    st.header("🌊 Workflow Designer")
    
    with st.form("workflow_form"):
        st.subheader("Workflow Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            workflow_name = st.text_input("Workflow Name", placeholder="e.g., Research Pipeline")
            entry_point = st.text_input("Entry Point", placeholder="start_node")
        
        with col2:
            created_by = st.text_input("Created By", placeholder="Your name")
            version = st.text_input("Version", value="1.0.0")
        
        description = st.text_area("Description", placeholder="What does this workflow do?")
        
        # Agent selection
        st.subheader("Select Agents")
        if st.session_state.created_agents:
            selected_agents = st.multiselect(
                "Agents to include",
                options=[agent.name for agent in st.session_state.created_agents],
                default=[]
            )
        else:
            st.info("Create agents first to add them to workflows")
            selected_agents = []
        
        # Simple workflow graph definition
        st.subheader("Workflow Graph")
        graph_definition = st.text_area(
            "Graph Definition (JSON)",
            value=json.dumps({
                "nodes": {"start": {"type": "start"}, "end": {"type": "end"}},
                "edges": [{"from": "start", "to": "end"}]
            }, indent=2),
            height=200
        )
        
        submitted = st.form_submit_button("Create Workflow", use_container_width=True)
        
        if submitted:
            if workflow_name and entry_point:
                try:
                    # Parse graph definition
                    graph_dict = json.loads(graph_definition)
                    
                    # Get selected agent specs
                    agent_specs = [
                        agent for agent in st.session_state.created_agents 
                        if agent.name in selected_agents
                    ]
                    
                    # Create workflow definition
                    workflow_def = WorkflowDefinition(
                        name=workflow_name,
                        description=description,
                        agent_specs=agent_specs,
                        workflow_graph=graph_dict,
                        entry_point=entry_point,
                        created_by=created_by,
                        version=version
                    )
                    
                    # Create workflow
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    created_workflow = loop.run_until_complete(
                        st.session_state.orchestrator.create_workflow(workflow_def)
                    )
                    
                    # Store in session state
                    st.session_state.created_workflows.append(created_workflow)
                    
                    st.success(f"✅ Workflow '{workflow_name}' created successfully!")
                    
                except json.JSONDecodeError:
                    st.error("❌ Invalid JSON in graph definition")
                except Exception as e:
                    st.error(f"❌ Error creating workflow: {str(e)}")
            else:
                st.error("❌ Please fill in all required fields")


def show_execution_monitor():
    """Display the execution monitoring interface."""
    st.header("📊 Execution Monitor")
    
    # Active executions
    st.subheader("Active Executions")
    
    # TODO: Get actual active executions from orchestrator
    active_executions = []
    
    if active_executions:
        for execution in active_executions:
            with st.expander(f"Execution {execution.id}"):
                st.write(f"**Workflow:** {execution.workflow_id}")
                st.write(f"**Current Step:** {execution.current_step}")
                st.write(f"**Started:** {execution.created_at}")
    else:
        st.info("No active executions")
    
    # Execution history
    st.subheader("Recent Executions")
    st.info("Execution history will be displayed here")


def show_settings():
    """Display the settings interface."""
    st.header("⚙️ Settings")
    
    st.subheader("Application Settings")
    
    # Theme
    theme = st.selectbox("Theme", ["Auto", "Light", "Dark"])
    
    # API Settings
    st.subheader("API Configuration")
    api_host = st.text_input("API Host", value="localhost")
    api_port = st.number_input("API Port", value=8000)
    
    # Default model settings
    st.subheader("Default Model Settings")
    default_provider = st.selectbox("Default Provider", [p.value for p in ModelProvider])
    default_model = st.text_input("Default Model", value="gpt-3.5-turbo")
    
    if st.button("Save Settings"):
        st.success("✅ Settings saved successfully!")


if __name__ == "__main__":
    main()
