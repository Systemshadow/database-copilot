"""
Database Copilot - Demo Ready Version
Creates test data in memory for reliable cloud deployment.
"""

import streamlit as st
import pandas as pd
import sqlite3
import sys
import os
from pathlib import Path
import tempfile

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from app_utils.ai_assistant import DatabaseAssistant
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()


def get_env_var(key: str, default: str = "") -> str:
    """Get environment variable from .env file or Streamlit secrets."""
    # Try Streamlit secrets first (for cloud deployment)
    if hasattr(st, 'secrets') and key in st.secrets:
        return st.secrets[key]
    # Fallback to environment variables (for local development)
    return os.getenv(key, default)


class DemoDatabase:
    """Demo database that creates test data in memory."""
    
    def __init__(self):
        self.connection = None
        self.connected = False
        
    def connect(self) -> bool:
        """Create in-memory database with sample data."""
        try:
            # Create temporary database file
            self.db_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
            self.db_file.close()
            
            self.connection = sqlite3.connect(self.db_file.name)
            self._create_sample_data()
            self.connected = True
            return True
        except Exception as e:
            st.error(f"Database connection failed: {str(e)}")
            return False
    
    def _create_sample_data(self):
        """Create sample oil & gas production data."""
        # Sample production data
        production_data = []
        wells = [
            {'name': 'Smith 1H', 'operator': 'Chesapeake Energy', 'county': 'Broome'},
            {'name': 'Jones 2H', 'operator': 'Range Resources', 'county': 'Tioga'},
            {'name': 'Brown 3H', 'operator': 'Chesapeake Energy', 'county': 'Chemung'},
            {'name': 'Wilson 4H', 'operator': 'Cabot Oil', 'county': 'Broome'}
        ]
        
        # Generate data for 2022 and 2023
        for year in [2022, 2023]:
            for month in range(1, 13):
                for well in wells:
                    # Vary production by month and well
                    base_oil = 800 + (hash(well['name']) % 400)
                    base_gas = 40000 + (hash(well['name']) % 20000)
                    base_water = 150 + (hash(well['name']) % 100)
                    
                    # Seasonal variation
                    seasonal_factor = 0.8 + 0.4 * (month / 12)
                    year_factor = 1.1 if year == 2023 else 1.0
                    
                    production_data.append({
                        'API_WellNo': f'31-001-{12345 + wells.index(well)}',
                        'Well_Name': well['name'],
                        'Operator': well['operator'],
                        'County': well['county'],
                        'Year': year,
                        'Month': month,
                        'MonthProd': month,
                        'OilProd': round(base_oil * seasonal_factor * year_factor, 1),
                        'GasProd': round(base_gas * seasonal_factor * year_factor, 1),
                        'WaterProd': round(base_water * seasonal_factor * year_factor, 1),
                        'Production_Date': f'{year}-{month:02d}-01'
                    })
        
        # Create production table
        prod_df = pd.DataFrame(production_data)
        prod_df.to_sql('well_production', self.connection, if_exists='replace', index=False)
        
        # Create wells table
        wells_data = []
        for i, well in enumerate(wells):
            wells_data.append({
                'API_WellNo': f'31-001-{12345 + i}',
                'Well_Name': well['name'],
                'Operator_Name': well['operator'],
                'County': well['county'],
                'Well_Type': 'Horizontal',
                'Status': 'Producing',
                'Spud_Date': f'2022-{3+i:02d}-15',
                'Total_Depth': 8500 + i * 100,
                'Field_Name': 'Marcellus'
            })
        
        wells_df = pd.DataFrame(wells_data)
        wells_df.to_sql('wells', self.connection, if_exists='replace', index=False)
        
        # Commit changes
        self.connection.commit()
    
    def execute_query(self, sql: str) -> pd.DataFrame:
        """Execute SQL query and return DataFrame."""
        if not self.connected:
            raise RuntimeError("Not connected to database")
        
        # Add safety limit
        if 'LIMIT' not in sql.upper() and 'TOP' not in sql.upper():
            sql += ' LIMIT 100'
        
        return pd.read_sql(sql, self.connection)
    
    def list_tables(self) -> list:
        """List all tables."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return [row[0] for row in cursor.fetchall()]
    
    def describe_table(self, table_name: str) -> pd.DataFrame:
        """Describe table structure."""
        cursor = self.connection.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        return pd.DataFrame(columns, columns=['cid', 'column_name', 'type', 'notnull', 'dflt_value', 'pk'])
    
    def close(self):
        """Close connection."""
        if self.connection:
            self.connection.close()
        self.connected = False


# Simplified AI Assistant for demo
class SimpleDemoAssistant:
    """Simplified assistant for demo that works with our demo database."""
    
    def __init__(self, db: DemoDatabase):
        self.db = db
        self.openai_client = None
        self._initialize_openai()
    
    def _initialize_openai(self):
        """Initialize OpenAI client."""
        api_key = get_env_var('OPENAI_API_KEY')
        if api_key and api_key != 'your_openai_api_key_here':
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=api_key)
            except:
                pass
    
    def ask_question(self, question: str) -> str:
        """Process a question and return a conversational response."""
        try:
            # Generate SQL from question
            sql_query = self._generate_sql(question)
            if not sql_query:
                return "I couldn't understand that question. Try asking about wells, production, or operators."
            
            # Execute query
            result_df = self.db.execute_query(sql_query)
            
            if result_df.empty:
                return self._generate_not_found_response(question)
            
            # Generate response
            return self._generate_response(question, result_df, sql_query)
            
        except Exception as e:
            return f"I encountered an error: {str(e)}"
    
    def _generate_sql(self, question: str) -> str:
        """Generate SQL from natural language."""
        question_lower = question.lower()
        
        # Simple pattern matching for demo
        if 'smith 1h' in question_lower and 'january 2023' in question_lower:
            return "SELECT * FROM well_production WHERE Well_Name = 'Smith 1H' AND Year = 2023 AND Month = 1"
        
        elif 'top' in question_lower and 'wells' in question_lower and 'oil' in question_lower:
            return "SELECT Well_Name, SUM(OilProd) as total_oil FROM well_production WHERE Year = 2023 GROUP BY Well_Name ORDER BY total_oil DESC LIMIT 10"
        
        elif 'chesapeake' in question_lower and 'total' in question_lower:
            return "SELECT SUM(OilProd) as total_oil, SUM(GasProd) as total_gas FROM well_production WHERE Operator = 'Chesapeake Energy'"
        
        elif 'county' in question_lower and 'production' in question_lower:
            return "SELECT County, SUM(OilProd) as total_oil, SUM(GasProd) as total_gas FROM well_production GROUP BY County ORDER BY total_oil DESC"
        
        elif 'wells' in question_lower or 'all' in question_lower:
            return "SELECT * FROM well_production LIMIT 10"
        
        else:
            return "SELECT Well_Name, Operator, County, OilProd, GasProd FROM well_production WHERE Year = 2023 LIMIT 10"
    
    def _generate_not_found_response(self, question: str) -> str:
        """Generate response when no data found."""
        if 'xyz-456' in question.lower() or 'abc-123' in question.lower():
            return "That well isn't in our database. The wells I have data for are Smith 1H, Jones 2H, Brown 3H, and Wilson 4H."
        return "I didn't find any data matching that criteria. Try asking about Smith 1H, Chesapeake Energy, or Broome County."
    
    def _generate_response(self, question: str, df: pd.DataFrame, sql: str) -> str:
        """Generate conversational response from data."""
        if self.openai_client:
            return self._generate_ai_response(question, df)
        else:
            return self._generate_simple_response(question, df)
    
    def _generate_simple_response(self, question: str, df: pd.DataFrame) -> str:
        """Generate simple response without AI."""
        if len(df) == 1 and 'OilProd' in df.columns:
            row = df.iloc[0]
            return f"In {row.get('Production_Date', 'that period')}, {row.get('Well_Name', 'the well')} produced {row.get('OilProd', 0):,.1f} barrels of oil and {row.get('GasProd', 0):,.1f} MCF of gas."
        
        elif 'total' in question.lower():
            oil_total = df['OilProd'].sum() if 'OilProd' in df.columns else 0
            gas_total = df['GasProd'].sum() if 'GasProd' in df.columns else 0
            return f"Total production: {oil_total:,.1f} barrels of oil and {gas_total:,.1f} MCF of gas."
        
        else:
            return f"I found {len(df)} records. The data includes wells like {', '.join(df['Well_Name'].unique()[:3])} with production data from our database."
    
    def _generate_ai_response(self, question: str, df: pd.DataFrame) -> str:
        """Generate AI response with OpenAI."""
        try:
            data_summary = df.to_string(max_rows=5, max_cols=8)
            
            prompt = f"""
            Answer this question about oil & gas production data conversationally:
            
            Question: {question}
            
            Data: {data_summary}
            
            Respond like a knowledgeable analyst. Include specific numbers and insights.
            Be conversational and helpful.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
        except:
            return self._generate_simple_response(question, df)


def initialize_app():
    """Initialize the application."""
    st.set_page_config(
        page_title="Database Copilot - Demo",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'db_connected' not in st.session_state:
        st.session_state.db_connected = False
    if 'demo_db' not in st.session_state:
        st.session_state.demo_db = None
    if 'assistant' not in st.session_state:
        st.session_state.assistant = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []


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
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Database Copilot</h1>
        <p>AI-Powered Database Assistant - Demo Version</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔌 Database Connection")
        
        if not st.session_state.db_connected:
            st.warning("⚠️ Not connected to database")
            
            if st.button("🔗 Connect to Demo Database", type="primary"):
                with st.spinner("Connecting to demo database..."):
                    demo_db = DemoDatabase()
                    success = demo_db.connect()
                    
                if success:
                    st.session_state.db_connected = True
                    st.session_state.demo_db = demo_db
                    st.session_state.assistant = SimpleDemoAssistant(demo_db)
                    st.success("✅ Connected to demo database!")
                    st.rerun()
                else:
                    st.error("❌ Connection failed")
        else:
            st.success("✅ Connected to demo database")
            
            if st.button("🔄 Refresh Connection"):
                if st.session_state.demo_db:
                    st.session_state.demo_db.close()
                st.session_state.db_connected = False
                st.session_state.demo_db = None
                st.session_state.assistant = None
                st.session_state.chat_history = []
                st.rerun()
        
        # Database info
        if st.session_state.db_connected and st.session_state.demo_db:
            st.header("📊 Database Info")
            tables = st.session_state.demo_db.list_tables()
            st.success(f"📋 Tables: {len(tables)}")
            for table in tables:
                st.write(f"• {table}")
    
    # Main interface
    if not st.session_state.db_connected:
        st.info("💡 Click 'Connect to Demo Database' in the sidebar to get started!")
        
        # Show example questions
        st.subheader("💡 Demo Questions You Can Ask:")
        examples = [
            "What was the oil production for well Smith 1H in January 2023?",
            "Show me the top producing wells",
            "What's the total production for Chesapeake Energy?",
            "Show me production by county",
            "Tell me about the wells in the database"
        ]
        
        for example in examples:
            st.info(f'"{example}"')
    
    else:
        # Chat interface
        st.subheader("💬 Ask Questions About Your Data")
        
        # Show example questions as buttons
        examples = [
            "What was the oil production for well Smith 1H in January 2023?",
            "Show me the top producing wells",
            "What's the total production for Chesapeake Energy?",
            "Show me production by county"
        ]
        
        cols = st.columns(2)
        for i, example in enumerate(examples):
            with cols[i % 2]:
                if st.button(example, key=f"ex_{i}"):
                    st.session_state.current_question = example
        
        # Chat history
        for question, response in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                st.write(response)
        
        # Chat input
        question = st.chat_input("Ask a question about the database...")
        
        # Handle example clicks
        if 'current_question' in st.session_state:
            question = st.session_state.current_question
            del st.session_state.current_question
        
        if question:
            # Show user question
            with st.chat_message("user"):
                st.write(question)
            
            # Generate and show response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.assistant.ask_question(question)
                st.write(response)
            
            # Add to history
            st.session_state.chat_history.append((question, response))
    
    # Footer
    st.markdown("---")
    st.markdown("**Database Copilot Demo** - AI-powered database assistant with sample oil & gas data")


if __name__ == "__main__":
    main()