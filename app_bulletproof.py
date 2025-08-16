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
    """Demo assistant that uses OpenAI API for intelligent responses."""
    
    def __init__(self):
        self.openai_client = None
        self._initialize_openai()
        
        # Sample data context for the AI
        self.data_context = """
        PRODUCTION DATABASE CONTEXT:
        
        Wells in database:
        - Smith 1H (Chesapeake Energy, Broome County): 2023: 14,567 bbls oil, 567,823 MCF gas | 2022: 13,412 bbls oil
        - Jones 2H (Range Resources, Tioga County): 2023: 12,234 bbls oil, 523,456 MCF gas | 2022: 11,876 bbls oil  
        - Brown 3H (Chesapeake Energy, Chemung County): 2023: 13,789 bbls oil, 545,678 MCF gas | 2022: 12,234 bbls oil
        - Wilson 4H (Cabot Oil, Broome County): 2023: 11,456 bbls oil, 489,234 MCF gas | 2022: 10,310 bbls oil
        
        TOTALS:
        - 2023: 52,046 bbls oil total, 2.13M MCF gas total
        - 2022: 47,832 bbls oil total, 1.96M MCF gas total
        - Growth: +9% oil year-over-year
        
        All wells are horizontal Marcellus shale wells in New York state.
        """
    
    def _initialize_openai(self):
        """Initialize OpenAI client."""
        api_key = get_env_var('OPENAI_API_KEY')
        if api_key and api_key != 'your_openai_api_key_here':
            try:
                import openai
                self.openai_client = openai.OpenAI(api_key=api_key)
            except Exception as e:
                print(f"OpenAI initialization failed: {e}")
    
    def ask_question(self, question: str) -> str:
        """Generate intelligent responses using OpenAI API."""
        
        # If OpenAI is available, use it for intelligent responses
        if self.openai_client:
            try:
                prompt = f"""
                You are an expert oil & gas data analyst having a conversation with a colleague about production data. 
                Respond naturally and conversationally based on the database information provided.

                QUESTION: {question}

                DATABASE INFORMATION:
                {self.data_context}

                INSTRUCTIONS:
                - Respond like a knowledgeable analyst talking to a colleague
                - Be conversational and natural, not robotic
                - Include specific numbers from the data when relevant
                - Provide insights and context, not just raw data
                - If the question is about something not in the database, explain what data you do have
                - Use formatting like **bold** for emphasis when helpful
                - Keep responses focused and helpful
                - Add relevant insights about trends, comparisons, or implications

                Respond as a human analyst would in a conversation.
                """
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a knowledgeable oil & gas production analyst. Respond conversationally and naturally, like you're talking to a colleague. Use the provided data to give intelligent, insightful answers."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.3
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                print(f"OpenAI API error: {e}")
                return self._fallback_response(question)
        
        else:
            return self._fallback_response(question)
    
    def _fallback_response(self, question: str) -> str:
        """Fallback response when OpenAI is not available."""
        return f"""I'd love to help analyze that for you! However, I need my OpenAI API connection to provide intelligent responses.
        
        **What I can tell you about our database:**
        • 4 wells: Smith 1H, Jones 2H, Brown 3H, Wilson 4H
        • 3 operators: Chesapeake Energy, Range Resources, Cabot Oil  
        • 2022-2023 production data
        • Total 2023: 52,046 bbls oil, 2.13M MCF gas
        
        With the full AI connection, I could analyze trends, compare performance, and provide detailed insights about your question: "{question}"
        
        *This demonstrates the concept - in production, the AI would provide detailed, conversational analysis of your actual database.*"""


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