"""
Database Copilot - AI-Powered Database Assistant
Ask questions about your oil & gas database in natural language.
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from app_utils.database import db_manager
    from app_utils.ai_assistant import DatabaseAssistant
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all files are in the correct directory structure.")
    st.stop()


def initialize_app():
    """Initialize the application and database connection."""
    st.set_page_config(
        page_title="Database Copilot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'db_connected' not in st.session_state:
        st.session_state.db_connected = False
    if 'assistant' not in st.session_state:
        st.session_state.assistant = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'schema_discovered' not in st.session_state:
        st.session_state.schema_discovered = False


def connect_to_database():
    """Handle database connection."""
    with st.sidebar:
        st.header("🔌 Database Connection")
        
        if not st.session_state.db_connected:
            st.warning("Not connected to database")
            
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
            st.success("✅ Connected to database")
            
            if st.button("🔄 Refresh Connection"):
                db_manager.close()
                st.session_state.db_connected = False
                st.session_state.assistant = None
                st.session_state.schema_discovered = False
                st.rerun()


def discover_schema():
    """Handle schema discovery."""
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


def chat_interface():
    """Main chat interface."""
    if not st.session_state.db_connected:
        st.warning("Please connect to your database first using the sidebar.")
        return
    
    if not st.session_state.schema_discovered:
        st.info("💡 Tip: Click 'Discover Schema' in the sidebar to analyze your database structure first.")
    
    st.title("🤖 Database Copilot")
    st.markdown("Ask questions about your oil & gas data in plain English. I'll query your database and provide intelligent answers.")
    
    # Updated example questions that match your actual database
    with st.expander("💡 Example Questions", expanded=True):
        examples = [
            "How many wells are in the database?",
            "Show me the top 10 producing wells in 2023",
            "What was the total oil production in 2023?",
            "What was the production for well XTO-6313H in 2023?",
            "Compare production between XTO-6313H and WPX-8369H",
            "What's the total production for XTO Energy Inc in 2023?",
            "How many wells does XTO Energy Inc operate?",
            "Show me all wells in Genesee County",
        ]
        
        cols = st.columns(2)
        for i, example in enumerate(examples):
            with cols[i % 2]:
                if st.button(example, key=f"example_{i}", use_container_width=True):
                    st.session_state.current_question = example
    
    # Chat history display
    chat_container = st.container()
    
    with chat_container:
        for i, (question, response) in enumerate(st.session_state.chat_history):
            # User question
            with st.chat_message("user"):
                st.write(question)
            
            # Assistant response
            with st.chat_message("assistant"):
                st.write(response)
    
    # Question input
    question = st.chat_input("Ask a question about your database...")
    
    # Handle example button clicks
    if 'current_question' in st.session_state:
        question = st.session_state.current_question
        del st.session_state.current_question
    
    if question:
        # Add user question to chat
        with chat_container:
            with st.chat_message("user"):
                st.write(question)
        
        # Generate response
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


def configuration_panel():
    """Show configuration status and tips."""
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
        
        if st.button("📋 Show Config Help"):
            show_config_help()


def show_config_help():
    """Show configuration help modal."""
    st.info("""
    **Required Environment Variables (.env file):**
    
    ```
    OPENAI_API_KEY=your_actual_api_key
    DATABASE_TYPE=sqlserver
    DATABASE_HOST=your_server.com
    DATABASE_NAME=ProductionDB
    DATABASE_USER=your_username
    DATABASE_PASSWORD=your_password
    ```
    
    **Supported Database Types:**
    - sqlserver (SQL Server)
    - postgresql (PostgreSQL)
    - mysql (MySQL)
    - oracle (Oracle)
    - sqlite (Local testing)
    """)


def main():
    """Main application entry point."""
    initialize_app()
    
    # Sidebar components
    connect_to_database()
    discover_schema()
    configuration_panel()
    
    # Main chat interface
    chat_interface()
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"**Database Copilot** - AI-powered database assistant for {os.getenv('COMPANY_NAME', 'your company')}"
    )


if __name__ == "__main__":
    main()