"""
Database Copilot - Hybrid Version
Working functionality from original + improved UI styling.
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add project root to path - same as working version
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from app_utils.database import db_manager
    from app_utils.ai_assistant import DatabaseAssistant
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()


def set_modern_styling():
    """Add modern styling without breaking functionality."""
    st.markdown("""
    <style>
    /* Modern styling that doesn't interfere with functionality */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .status-connected {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 0.8rem;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .status-disconnected {
        background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
        color: white;
        padding: 0.8rem;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .example-section {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)


def initialize_app():
    """Initialize app - exact same as working version."""
    st.set_page_config(
        page_title="Database Copilot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    set_modern_styling()
    
    # Initialize session state - exact same as working version
    if 'db_connected' not in st.session_state:
        st.session_state.db_connected = False
    if 'assistant' not in st.session_state:
        st.session_state.assistant = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'schema_discovered' not in st.session_state:
        st.session_state.schema_discovered = False


def connect_to_database():
    """Handle database connection - exact same logic as working version."""
    with st.sidebar:
        st.header("🔌 Database Connection")
        
        if not st.session_state.db_connected:
            st.markdown("""
            <div class="status-disconnected">
                <h4>⚠️ Not connected to database</h4>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🔗 Connect to Database", type="primary"):
                with st.spinner("Connecting to database..."):
                    success = db_manager.connect()
                    
                if success:
                    st.session_state.db_connected = True
                    st.session_state.assistant = DatabaseAssistant(db_manager)
                    st.success("✅ Connected successfully!")
                    st.rerun()
                else:
                    st.error("❌ Connection failed. Check your .env configuration.")
        else:
            st.markdown("""
            <div class="status-connected">
                <h4>✅ Connected to database</h4>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🔄 Refresh Connection"):
                db_manager.close()
                st.session_state.db_connected = False
                st.session_state.assistant = None
                st.session_state.schema_discovered = False
                st.rerun()


def discover_schema():
    """Handle schema discovery - exact same as working version."""
    if st.session_state.db_connected and not st.session_state.schema_discovered:
        with st.sidebar:
            st.header("📊 Database Schema")
            
            if st.button("🔍 Discover Schema"):
                with st.spinner("Analyzing database schema..."):
                    try:
                        tables = db_manager.discover_schema()
                        st.session_state.schema_discovered = True
                        st.success(f"Found {len(tables)} tables")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Schema discovery failed: {str(e)}")
    
    elif st.session_state.schema_discovered:
        with st.sidebar:
            st.header("📊 Database Schema")
            
            # Show table summary
            tables = list(db_manager.schema_cache.values())
            
            with st.expander(f"📋 Tables ({len(tables)})", expanded=False):
                for table in tables[:10]:  # Limit display
                    st.write(f"**{table.name}** ({table.row_count:,} rows)" if table.row_count else f"**{table.name}**")
                    
                    # Show first few columns
                    col_names = [col['name'] for col in table.columns[:5]]
                    st.caption(f"Columns: {', '.join(col_names)}{'...' if len(table.columns) > 5 else ''}")


def configuration_panel():
    """Show configuration status - exact same as working version."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Check environment variables
        env_vars = {
            "OpenAI API Key": os.getenv('OPENAI_API_KEY', '').replace('your_openai_api_key_here', ''),
            "Database Host": os.getenv('DATABASE_HOST'),
            "Database Name": os.getenv('DATABASE_NAME'),
            "Database Type": os.getenv('DATABASE_TYPE')
        }
        
        for var_name, var_value in env_vars.items():
            if var_value:
                st.success(f"✅ {var_name}")
            else:
                st.error(f"❌ {var_name}")


def chat_interface():
    """Chat interface - exact same logic as working version but with better styling."""
    if not st.session_state.db_connected:
        st.warning("Please connect to your database first using the sidebar.")
        
        # Show example questions with better styling
        st.markdown("""
        <div class="example-section">
            <h3>💡 Example Questions You Can Ask:</h3>
            <p>Connect to your database to try these:</p>
        </div>
        """, unsafe_allow_html=True)
        
        examples = [
            "What was the production for well ABC-123 in January 2023?",
            "Show me the top 10 producing wells last month",
            "What's the total oil production for Chesapeake Energy in 2023?", 
            "Which county had the highest gas production last year?",
            "How did production change for well XYZ-456 over the last 6 months?"
        ]
        
        cols = st.columns(2)
        for i, example in enumerate(examples):
            with cols[i % 2]:
                st.info(f"*\"{example}\"*")
        
        return
    
    if not st.session_state.schema_discovered:
        st.info("💡 Tip: Click 'Discover Schema' in the sidebar to analyze your database structure first.")
    
    st.title("🤖 Database Copilot")
    st.markdown("Ask questions about your oil & gas data in plain English. I'll query your database and provide intelligent answers.")
    
    # Example questions with better styling - but only if connected
    st.markdown("""
    <div class="example-section">
        <h3>💡 Quick Start - Try These Questions:</h3>
    </div>
    """, unsafe_allow_html=True)
    
    examples = [
        "What was the production for well ABC-123 in January 2023?",
        "Show me the top 10 producing wells last month",
        "What's the total oil production for Chesapeake Energy in 2023?", 
        "Which county had the highest gas production last year?",
        "How did production change for well XYZ-456 over the last 6 months?"
    ]
    
    cols = st.columns(2)
    for i, example in enumerate(examples):
        with cols[i % 2]:
            if st.button(example, key=f"example_{i}", use_container_width=True):
                st.session_state.current_question = example
    
    # Chat history display - exact same as working version
    chat_container = st.container()
    
    with chat_container:
        for i, (question, response) in enumerate(st.session_state.chat_history):
            # User question
            with st.chat_message("user"):
                st.write(question)
            
            # Assistant response
            with st.chat_message("assistant"):
                st.write(response)
    
    # Question input - exact same as working version
    question = st.chat_input("Ask a question about your database...")
    
    # Handle example button clicks - exact same as working version
    if 'current_question' in st.session_state:
        question = st.session_state.current_question
        del st.session_state.current_question
    
    if question:
        # Add user question to chat - exact same as working version
        with chat_container:
            with st.chat_message("user"):
                st.write(question)
        
        # Generate response - exact same as working version
        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.assistant.ask_question(question)
                        st.write(response)
                        
                        # Add to chat history
                        st.session_state.chat_history.append((question, response))
                        
                    except Exception as e:
                        error_msg = f"Sorry, I encountered an error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history.append((question, error_msg))


def main():
    """Main application entry point - exact same structure as working version."""
    initialize_app()
    
    # Header with better styling
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Database Copilot</h1>
        <p>AI-Powered Database Assistant</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar components - exact same as working version
    connect_to_database()
    discover_schema()
    configuration_panel()
    
    # Main chat interface - exact same as working version
    chat_interface()
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"**Database Copilot** - AI-powered database assistant for {os.getenv('COMPANY_NAME', 'your company')}"
    )


if __name__ == "__main__":
    main()