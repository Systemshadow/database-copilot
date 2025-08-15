"""
Database Copilot - Modern UI Version
Beautiful, engaging interface for AI-powered database queries.
"""

import streamlit as st
import sys
import os
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from app_utils.database import db_manager
    from app_utils.ai_assistant import DatabaseAssistant
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()


def set_modern_theme():
    """Apply modern, attractive styling."""
    st.markdown("""
    <style>
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom CSS for modern look */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin: 0;
    }
    
    .stats-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .chat-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #e1e5e9;
    }
    
    .sidebar-header {
        background: linear-gradient(135deg, #2196F3 0%, #21CBF3 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
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
    
    .example-question {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border: none;
        border-radius: 20px;
        padding: 0.8rem 1.2rem;
        margin: 0.3rem;
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 0.9rem;
        color: #1565c0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .example-question:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
        color: white;
    }
    
    .feature-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #2196F3;
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* Chat messages styling */
    .user-message {
        background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 1rem 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #f5f5f5 0%, #eeeeee 100%);
        color: #333;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 1rem 0;
        max-width: 80%;
        margin-right: auto;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #2196F3;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    /* Loading spinner */
    .loading-spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #2196F3;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
    """, unsafe_allow_html=True)


def initialize_app():
    """Initialize the application with modern theme."""
    st.set_page_config(
        page_title="Database Copilot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    set_modern_theme()
    
    # Initialize session state
    if 'db_connected' not in st.session_state:
        st.session_state.db_connected = False
    if 'assistant' not in st.session_state:
        st.session_state.assistant = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'schema_discovered' not in st.session_state:
        st.session_state.schema_discovered = False


def modern_sidebar():
    """Create modern, attractive sidebar."""
    with st.sidebar:
        # Sidebar header
        st.markdown("""
        <div class="sidebar-header">
            <h2>🤖 Database Copilot</h2>
            <p>AI-Powered Data Assistant</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Connection status
        if st.session_state.db_connected:
            st.markdown("""
            <div class="status-connected">
                <h4>✅ Connected</h4>
                <p>Ready to answer questions</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🔄 Refresh Connection", use_container_width=True):
                disconnect_database()
        else:
            st.markdown("""
            <div class="status-disconnected">
                <h4>⚠️ Not Connected</h4>
                <p>Click to connect to database</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🔗 Connect to Database", type="primary", use_container_width=True):
                connect_database()
        
        # Schema info
        if st.session_state.db_connected:
            st.markdown("---")
            st.markdown("### 📊 Database Info")
            
            if not st.session_state.schema_discovered:
                if st.button("🔍 Discover Schema", use_container_width=True):
                    discover_schema()
            else:
                show_schema_info()
        
        # Configuration status
        st.markdown("---")
        st.markdown("### ⚙️ Configuration")
        show_config_status()


def connect_database():
    """Handle database connection with loading animation."""
    with st.spinner("🔌 Connecting to database..."):
        time.sleep(1)  # Brief pause for UX
        success = db_manager.connect()
        
    if success:
        st.session_state.db_connected = True
        st.session_state.assistant = DatabaseAssistant(db_manager)
        st.success("✅ Connected successfully!")
        st.rerun()
    else:
        st.error("❌ Connection failed. Check your configuration.")


def disconnect_database():
    """Disconnect from database."""
    db_manager.close()
    st.session_state.db_connected = False
    st.session_state.assistant = None
    st.session_state.schema_discovered = False
    st.session_state.chat_history = []
    st.rerun()


def discover_schema():
    """Discover database schema with loading animation."""
    with st.spinner("🔍 Analyzing database schema..."):
        try:
            tables = db_manager.discover_schema()
            st.session_state.schema_discovered = True
            st.success(f"✅ Found {len(tables)} tables")
            st.rerun()
        except Exception as e:
            st.error(f"Schema discovery failed: {str(e)}")


def show_schema_info():
    """Show database schema information."""
    tables = list(db_manager.schema_cache.values())
    
    with st.expander(f"📋 Tables ({len(tables)})", expanded=False):
        for table in tables[:5]:  # Show first 5 tables
            col_count = len(table.columns)
            row_count = f"{table.row_count:,}" if table.row_count else "Unknown"
            
            st.markdown(f"""
            **{table.name}**  
            📊 {row_count} rows • {col_count} columns
            """)


def show_config_status():
    """Show configuration status with icons."""
    env_vars = {
        "🔑 OpenAI API": os.getenv('OPENAI_API_KEY', '').replace('your_openai_api_key_here', ''),
        "🗄️ Database Host": os.getenv('DATABASE_HOST'),
        "📊 Database Name": os.getenv('DATABASE_NAME'),
        "🔧 Database Type": os.getenv('DATABASE_TYPE')
    }
    
    for var_name, var_value in env_vars.items():
        if var_value:
            st.success(f"{var_name}")
        else:
            st.error(f"{var_name}")


def hero_section():
    """Create stunning hero section."""
    st.markdown("""
    <div class="main-header fade-in">
        <h1>🤖 Database Copilot</h1>
        <p>Ask questions about your oil & gas data in plain English</p>
    </div>
    """, unsafe_allow_html=True)


def stats_section():
    """Show database statistics if connected."""
    if st.session_state.db_connected and st.session_state.schema_discovered:
        tables = list(db_manager.schema_cache.values())
        total_rows = sum(table.row_count or 0 for table in tables)
        total_cols = sum(len(table.columns) for table in tables)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(tables)}</div>
                <div class="metric-label">📊 Tables</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_rows:,}</div>
                <div class="metric-label">📈 Records</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_cols}</div>
                <div class="metric-label">📋 Columns</div>
            </div>
            """, unsafe_allow_html=True)


def example_questions():
    """Show interactive example questions with beautiful styling."""
    st.markdown("### 💡 Quick Start - Try These Questions")
    
    examples = [
        "What was the oil production for well Smith 1H in January 2023?",
        "Show me the top 10 producing wells last month",
        "What's the total gas production for Chesapeake Energy?",
        "Which county had the highest production last year?",
        "How did production change over the last 6 months?"
    ]
    
    # Create a more prominent display
    st.markdown("""
    <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); 
                padding: 1.5rem; border-radius: 15px; margin: 1rem 0;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
        <h4 style="color: #1565c0; margin-bottom: 1rem;">🚀 Click any question to get started:</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Display questions in a grid with better styling
    cols = st.columns(2)
    for i, example in enumerate(examples):
        with cols[i % 2]:
            # Custom HTML button with better styling
            button_html = f"""
            <div style="margin: 0.5rem 0;">
                <button onclick="document.querySelector('input[type=text]').value='{example}'; 
                               document.querySelector('input[type=text]').dispatchEvent(new Event('input'));"
                        style="width: 100%; padding: 1rem; border: none; border-radius: 15px;
                               background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
                               color: white; cursor: pointer; font-size: 0.9rem;
                               box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
                               transition: all 0.3s ease;">
                    {example}
                </button>
            </div>
            """
            
            # Use both HTML and Streamlit button for better compatibility
            if st.button(example, key=f"example_{i}", use_container_width=True, 
                        help="Click to ask this question"):
                st.session_state.current_question = example
                st.rerun()


def chat_interface():
    """Modern chat interface with beautiful styling."""
    # Always show example questions prominently at the top
    example_questions()
    
    if not st.session_state.db_connected:
        st.info("💡 Connect to your database in the sidebar to start asking questions!")
        return
    
    st.markdown("---")
    st.markdown("### 💬 Chat with Your Database")
    
    # Chat history display with custom styling
    for i, (question, response) in enumerate(st.session_state.chat_history):
        # User question
        st.markdown(f"""
        <div class="user-message">
            <strong>You:</strong> {question}
        </div>
        """, unsafe_allow_html=True)
        
        # Assistant response
        st.markdown(f"""
        <div class="assistant-message">
            <strong>🤖 Assistant:</strong> {response}
        </div>
        """, unsafe_allow_html=True)
    
    # Question input
    question = st.chat_input("Ask a question about your database...", key="chat_input")
    
    # Handle example button clicks
    if 'current_question' in st.session_state:
        question = st.session_state.current_question
        del st.session_state.current_question
    
    if question:
        # Add question to history immediately for better UX
        st.session_state.chat_history.append((question, "🤖 Thinking..."))
        st.rerun()
        
        # Generate response with better error handling
        try:
            # Show progress
            with st.spinner("🧠 Analyzing your question..."):
                print(f"DEBUG: Processing question: {question}")  # Debug log
                response = st.session_state.assistant.ask_question(question)
                print(f"DEBUG: Got response: {response[:100]}...")  # Debug log
            
            # Update the last response
            st.session_state.chat_history[-1] = (question, response)
            st.rerun()
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            print(f"ERROR: {error_msg}")  # Debug log
            st.session_state.chat_history[-1] = (question, f"❌ {error_msg}")
            st.error(f"Debug info: {str(e)}")  # Show error in UI too
            st.rerun()


def main():
    """Main application entry point."""
    initialize_app()
    
    # Sidebar
    modern_sidebar()
    
    # Main content
    hero_section()
    stats_section()
    
    # Chat interface
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    chat_interface()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    company_name = os.getenv('COMPANY_NAME', 'Your Company')
    st.markdown(f"**Database Copilot** - AI-powered data assistant for {company_name}")


if __name__ == "__main__":
    main()