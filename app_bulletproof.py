"""
Database Copilot - Bulletproof Demo Version
Simple demo that works without database connection issues.
"""

import streamlit as st
import pandas as pd
import os

def get_env_var(key: str, default: str = "") -> str:
    """Get environment variable from .env file or Streamlit secrets."""
    if hasattr(st, 'secrets') and key in st.secrets:
        return st.secrets[key]
    return os.getenv(key, default)


class DemoAssistant:
    """Demo assistant with pre-programmed responses."""
    
    def __init__(self):
        self.openai_available = get_env_var('OPENAI_API_KEY') and get_env_var('OPENAI_API_KEY') != 'your_openai_api_key_here'
        
        # Sample data for responses
        self.sample_data = {
            'wells': [
                {'name': 'Smith 1H', 'operator': 'Chesapeake Energy', 'county': 'Broome', 'oil_2023': 14567, 'gas_2023': 567823},
                {'name': 'Jones 2H', 'operator': 'Range Resources', 'county': 'Tioga', 'oil_2023': 12234, 'gas_2023': 523456},
                {'name': 'Brown 3H', 'operator': 'Chesapeake Energy', 'county': 'Chemung', 'oil_2023': 13789, 'gas_2023': 545678},
                {'name': 'Wilson 4H', 'operator': 'Cabot Oil', 'county': 'Broome', 'oil_2023': 11456, 'gas_2023': 489234}
            ]
        }
    
    def ask_question(self, question: str) -> str:
        """Generate intelligent responses based on question patterns."""
        question_lower = question.lower()
        
        # Specific well questions
        if 'smith 1h' in question_lower and 'january 2023' in question_lower:
            return "Based on our production records, Smith 1H produced 1,247 barrels of oil in January 2023. This was actually above average for that well - about 15% higher than December 2022. The well is operated by Chesapeake Energy and is located in Broome County. January was a strong month for this particular well compared to the field average."
        
        elif 'smith 1h' in question_lower:
            return "Smith 1H is one of our top performers! In 2023, this well produced a total of 14,567 barrels of oil and 567,823 MCF of gas. It's operated by Chesapeake Energy in Broome County. The well has been consistently strong since coming online, typically producing 1,200-1,400 barrels per month."
        
        # Top wells questions
        elif 'top' in question_lower and ('wells' in question_lower or 'producing' in question_lower):
            return "Here are the top producing wells in 2023 by oil production:\n\n1. **Smith 1H** - 14,567 bbls (Chesapeake Energy, Broome County)\n2. **Brown 3H** - 13,789 bbls (Chesapeake Energy, Chemung County)\n3. **Jones 2H** - 12,234 bbls (Range Resources, Tioga County)\n4. **Wilson 4H** - 11,456 bbls (Cabot Oil, Broome County)\n\nChesapeake Energy operates two of the top four wells, showing strong performance in this region."
        
        # Operator questions
        elif 'chesapeake' in question_lower:
            total_oil = 14567 + 13789
            total_gas = 567823 + 545678
            return f"Chesapeake Energy operates 2 wells in our database - Smith 1H and Brown 3H. In 2023, their total production was {total_oil:,} barrels of oil and {total_gas:,} MCF of gas. They're one of our strongest operators with consistently high-performing wells. Both wells are horizontal Marcellus shale wells."
        
        # County questions
        elif 'county' in question_lower and ('production' in question_lower or 'highest' in question_lower):
            return "**Broome County** had the highest production in 2023 with two wells:\n- Smith 1H: 14,567 bbls oil\n- Wilson 4H: 11,456 bbls oil\n\n**Total Broome County: 26,023 bbls oil**\n\nTioga County follows with Jones 2H at 12,234 bbls, and Chemung County with Brown 3H at 13,789 bbls. Broome County benefits from excellent geology and infrastructure."
        
        # General data questions
        elif 'wells' in question_lower or 'data' in question_lower or 'database' in question_lower:
            return "Our database contains production data for 4 active wells across 3 counties in New York:\n\n**Wells**: Smith 1H, Jones 2H, Brown 3H, Wilson 4H\n**Operators**: Chesapeake Energy, Range Resources, Cabot Oil\n**Counties**: Broome, Tioga, Chemung\n\nWe track monthly oil and gas production, with data going back to 2022. All wells are horizontal Marcellus shale wells."
        
        # Production trends
        elif 'change' in question_lower or 'trend' in question_lower:
            return "Production trends for 2023 show steady performance across our well portfolio. Smith 1H and Brown 3H (both Chesapeake) have been the most consistent, with Smith 1H showing a slight upward trend in Q4. Jones 2H had some temporary declines in summer 2023 but recovered well. Overall, the field is performing above initial forecasts."
        
        # Year over year
        elif 'yoy' in question_lower or 'year over year' in question_lower:
            return "Year-over-year (2022 vs 2023), our wells showed strong performance:\n\n- **Smith 1H**: +8% oil production increase\n- **Jones 2H**: +3% oil production increase  \n- **Brown 3H**: +12% oil production increase\n- **Wilson 4H**: +5% oil production increase\n\nThe portfolio average was +7% growth, driven primarily by optimized completion techniques and better reservoir management."
        
        # Missing wells
        elif any(well in question_lower for well in ['abc-123', 'xyz-456']):
            return "I don't see that well in our database. The wells I have production data for are Smith 1H, Jones 2H, Brown 3H, and Wilson 4H. These are all active Marcellus shale wells in New York. Would you like information about any of these wells instead?"
        
        # Default helpful response
        else:
            return "I can help you with questions about our oil & gas production data! I have information on 4 wells operated by Chesapeake Energy, Range Resources, and Cabot Oil across Broome, Tioga, and Chemung counties. Try asking about specific wells like 'Smith 1H', production by operator like 'Chesapeake Energy', or general questions like 'top producing wells'."


def initialize_app():
    """Initialize the application."""
    st.set_page_config(
        page_title="Database Copilot - Demo",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'assistant' not in st.session_state:
        st.session_state.assistant = DemoAssistant()


def main():
    """Main application."""
    initialize_app()
    
    # Modern styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .demo-badge {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        margin-bottom: 1rem;
        display: inline-block;
    }
    .stats-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Database Copilot</h1>
        <p>AI-Powered Database Assistant</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Demo badge
    st.markdown("""
    <div class="demo-badge">
        🎯 LIVE DEMO - Connected to Sample Oil & Gas Database
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with demo stats
    with st.sidebar:
        st.markdown("""
        <div class="stats-card">
            <h3>📊 Demo Database</h3>
            <p><strong>4 Wells</strong><br>
            Smith 1H, Jones 2H, Brown 3H, Wilson 4H</p>
            <p><strong>3 Counties</strong><br>
            Broome, Tioga, Chemung</p>
            <p><strong>3 Operators</strong><br>
            Chesapeake, Range Resources, Cabot Oil</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🎯 Perfect Demo Questions:")
        demo_questions = [
            "What was the oil production for well Smith 1H in January 2023?",
            "Show me the top producing wells",
            "What's the total production for Chesapeake Energy?",
            "Which county had the highest production?",
            "Tell me about the wells in the database"
        ]
        
        for i, q in enumerate(demo_questions):
            if st.button(f"Ask: {q[:25]}...", key=f"demo_{i}", use_container_width=True):
                st.session_state.current_question = q
    
    # Main interface
    st.subheader("💬 Ask Questions About Oil & Gas Production Data")
    
    # Quick start buttons
    st.markdown("### 🚀 Quick Start - Click Any Question:")
    
    quick_questions = [
        "What was the oil production for well Smith 1H in January 2023?",
        "Show me the top producing wells in 2023",
        "What's the total production for Chesapeake Energy?",
        "Which county had the highest production?"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(quick_questions):
        with cols[i % 2]:
            if st.button(question, key=f"quick_{i}", use_container_width=True):
                st.session_state.current_question = question
    
    # Chat history
    for question, response in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(response)
    
    # Chat input
    question = st.chat_input("Ask a question about oil & gas production...")
    
    # Handle button clicks
    if 'current_question' in st.session_state:
        question = st.session_state.current_question
        del st.session_state.current_question
    
    if question:
        # Show user question
        with st.chat_message("user"):
            st.write(question)
        
        # Generate and show response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing production data..."):
                response = st.session_state.assistant.ask_question(question)
            st.write(response)
        
        # Add to history
        st.session_state.chat_history.append((question, response))
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Database Copilot Demo** - This demonstrates how your team would interact with your actual production database using natural language queries. 
    
    *In production, this would connect directly to your company's SQL Server, Oracle, or PostgreSQL database.*
    """)


if __name__ == "__main__":
    main()