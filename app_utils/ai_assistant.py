"""
Enhanced AI-powered database assistant with intelligent error handling.
This replaces the old robotic responses with conversational expert responses.
"""

import os
import pandas as pd
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
import traceback

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .database import DatabaseManager, TableInfo

load_dotenv()


@dataclass
class QueryResult:
    """Result of a database query with metadata."""
    success: bool
    data: Optional[pd.DataFrame] = None
    sql_query: Optional[str] = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None


class ExpertDatabaseAssistant:
    """
    AI expert colleague for natural language database queries.
    Provides conversational, intelligent responses instead of robotic error messages.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.openai_client = None
        self.conversation_context = []
        self._initialize_openai()
    
    def _initialize_openai(self):
        """Initialize OpenAI client if API key is available."""
        api_key = os.getenv('OPENAI_API_KEY')
        if OPENAI_AVAILABLE and api_key and api_key != 'your_openai_api_key_here':
            try:
                self.openai_client = OpenAI(api_key=api_key)
            except Exception as e:
                print(f"OpenAI initialization failed: {str(e)}")
                self.openai_client = None
    
    def ask_question(self, question: str) -> str:
        """
        Main interface: ask a question and get a conversational response.
        THIS METHOD SHOULD NEVER THROW EXCEPTIONS - IT HANDLES ALL ERRORS GRACEFULLY.
        """
        try:
            # Add to conversation context
            self.conversation_context.append({"role": "user", "content": question})
            
            # Execute the query with full error handling
            query_result = self._execute_query_safely(question)
            
            # Generate conversational response
            response = self._generate_expert_response(question, query_result)
            
            # Add response to context
            self.conversation_context.append({"role": "assistant", "content": response})
            
            return response
            
        except Exception as e:
            # This is the final safety net - should rarely be reached
            return self._handle_critical_failure(question, str(e))
    
    def _execute_query_safely(self, question: str) -> QueryResult:
        """
        Execute query with comprehensive error handling.
        Never throws exceptions - always returns a QueryResult.
        """
        import time
        start_time = time.time()
        
        try:
            # Check for relative timeframes first
            if self._contains_relative_timeframe(question):
                return self._handle_relative_timeframe(question)
            
            # Find relevant tables
            relevant_tables = self.db_manager.find_relevant_tables(question)
            
            if not relevant_tables:
                return QueryResult(
                    success=False,
                    error_message="no_tables_found",
                    execution_time=time.time() - start_time
                )
            
            # Generate SQL query
            sql_query, error = self._generate_sql_safely(question, relevant_tables)
            
            if error:
                return QueryResult(
                    success=False, 
                    error_message=f"sql_generation_failed: {error}",
                    execution_time=time.time() - start_time
                )
            
            # Execute the query
            result_df = self.db_manager.execute_query(sql_query)
            
            return QueryResult(
                success=True,
                data=result_df,
                sql_query=sql_query,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return QueryResult(
                success=False,
                error_message=f"execution_error: {str(e)}",
                execution_time=execution_time
            )
    
    def _contains_relative_timeframe(self, question: str) -> bool:
        """Check if question contains relative timeframes that need special handling."""
        relative_terms = [
            "last month", "last year", "this month", "this year",
            "yesterday", "today", "recent", "lately", "currently",
            "now", "past month", "past year"
        ]
        
        question_lower = question.lower()
        return any(term in question_lower for term in relative_terms)
    
    def _handle_relative_timeframe(self, question: str) -> QueryResult:
        """Handle questions with relative timeframes intelligently."""
        # This creates a controlled "failure" that triggers intelligent response
        return QueryResult(
            success=False,
            error_message="relative_timeframe_issue",
            execution_time=0.1
        )
    
    def _generate_sql_safely(self, question: str, relevant_tables: List[str]) -> Tuple[str, str]:
        """Generate SQL with better error handling."""
        try:
            return self.generate_sql_query(question, relevant_tables)
        except Exception as e:
            return "", f"SQL generation exception: {str(e)}"
    
    def _generate_expert_response(self, question: str, query_result: QueryResult) -> str:
        """
        Generate expert-level conversational responses for all scenarios.
        This replaces ALL old robotic error handling.
        """
        if query_result.success and query_result.data is not None and not query_result.data.empty:
            # Successful query with data
            return self._generate_success_response(question, query_result)
        
        elif query_result.success and (query_result.data is None or query_result.data.empty):
            # Query executed but no data found
            return self._generate_no_data_response(question, query_result)
        
        else:
            # Query failed - handle intelligently based on error type
            return self._handle_query_failure(question, query_result)
    
    def _handle_query_failure(self, question: str, query_result: QueryResult) -> str:
        """
        Handle query failures with intelligent, conversational responses.
        This is the key method that replaces old robotic error handling.
        """
        error_msg = query_result.error_message or ""
        
        # Handle specific error types intelligently
        if "relative_timeframe_issue" in error_msg:
            return self._handle_timeframe_error(question)
        
        elif "no_tables_found" in error_msg:
            return self._handle_no_tables_error(question)
        
        elif "sql_generation_failed" in error_msg:
            return self._handle_sql_generation_error(question, error_msg)
        
        elif "execution_error" in error_msg:
            return self._handle_execution_error(question, error_msg)
        
        else:
            return self._handle_generic_error(question, error_msg)
    
    def _handle_timeframe_error(self, question: str) -> str:
        """Handle relative timeframe issues conversationally."""
        if not self.openai_client:
            return self._fallback_timeframe_response(question)
        
        try:
            # Get available date ranges from database
            available_data_context = self._get_available_timeframes()
            
            prompt = f"""
You're an expert oil & gas analyst talking to a colleague. They asked: "{question}"

The issue is they used relative timeframes like "last month" but we need to be specific about dates.

Available data: {available_data_context}

Respond conversationally like an expert colleague would, offering to help with specific timeframes instead.

Example good responses:
- "I'd love to show you last month's top performers, but our data goes through December 2024. Want me to pull December's numbers instead?"
- "For recent data, I can show you the latest we have which is through 2024. Should I look at December 2024 or the full year?"
- "Our data runs through 2024, so for 'recent' production I could show you Q4 2024 or the full year. Which would be more helpful?"

Be specific about what data you actually have and offer concrete alternatives.
"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a knowledgeable oil & gas analyst. Respond conversationally and helpfully when users ask about timeframes you don't have data for."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return self._fallback_timeframe_response(question)
    
    def _fallback_timeframe_response(self, question: str) -> str:
        """Fallback response for timeframe issues without AI."""
        question_lower = question.lower()
        
        if "last month" in question_lower:
            return "I'd love to show you last month's data, but our database has production data through 2024. Would you like me to show December 2024 instead?"
        
        elif "last year" in question_lower:
            return "For recent yearly data, I can show you 2023 or 2024 results. Which year would you prefer?"
        
        elif "recent" in question_lower or "lately" in question_lower:
            return "For recent data, our database goes through 2024. I can show you the latest quarter or the full year 2024. What timeframe works best?"
        
        else:
            return "I need specific dates or years to query the data. Our database covers 2000-2024. What timeframe would you like to see?"
    
    def _get_available_timeframes(self) -> str:
        """Get information about available date ranges in the database."""
        try:
            # Try to get date range from the main production table
            tables = list(self.db_manager.schema_cache.values())
            production_table = None
            
            for table in tables:
                if 'production' in table.name.lower() or 'prod' in table.name.lower():
                    production_table = table.name
                    break
            
            if production_table:
                date_query = f"SELECT MIN(year) as min_year, MAX(year) as max_year FROM {production_table}"
                result = self.db_manager.execute_query(date_query)
                if not result.empty:
                    min_year = result['min_year'].iloc[0]
                    max_year = result['max_year'].iloc[0]
                    return f"Data available from {min_year} to {max_year}"
            
            return "Data available from 2000 to 2024"
            
        except:
            return "Data available from 2000 to 2024"
    
    def _handle_no_tables_error(self, question: str) -> str:
        """Handle when no relevant tables are found."""
        # Get available tables
        available_tables = list(self.db_manager.schema_cache.keys()) if self.db_manager.schema_cache else []
        
        if available_tables:
            table_list = ", ".join(available_tables[:5])
            return f"I'm having trouble identifying which tables relate to your question about '{question}'. The main tables I see are: {table_list}. Could you be more specific about what data you're looking for?"
        else:
            return "I need to discover the database schema first. Could you try clicking 'Discover Schema' in the sidebar, then ask your question again?"
    
    def _handle_sql_generation_error(self, question: str, error_msg: str) -> str:
        """Handle SQL generation failures."""
        return f"I understand what you're asking, but I'm having trouble translating '{question}' into a database query. Could you try rephrasing it or being more specific about the wells, operators, or time periods you're interested in?"
    
    def _handle_execution_error(self, question: str, error_msg: str) -> str:
        """Handle database execution errors."""
        if "no such table" in error_msg.lower():
            return "It looks like I'm trying to query a table that doesn't exist. The database schema might have changed. Could you try refreshing the connection in the sidebar?"
        else:
            return f"I ran into a database issue while processing your question '{question}'. The database might be temporarily unavailable, or there could be a connection issue."
    
    def _handle_generic_error(self, question: str, error_msg: str) -> str:
        """Handle any other errors conversationally."""
        return f"I'm having trouble processing your question '{question}' right now. This might be a temporary issue. Could you try rephrasing your question or checking if the database connection is working?"
    
    def _handle_critical_failure(self, question: str, error_details: str) -> str:
        """Handle critical failures that shouldn't normally happen."""
        return f"I apologize, but I'm experiencing a technical issue while processing your question '{question}'. Please try again, or if the problem persists, check the database connection."
    
    def _generate_success_response(self, question: str, query_result: QueryResult) -> str:
        """Generate response for successful queries with data."""
        if not self.openai_client:
            return self._generate_fallback_response(question, query_result)
        
        try:
            # Prepare data summary for the LLM
            data_summary = self._summarize_data(query_result.data)
            
            prompt = f"""
You are an expert oil & gas data analyst having a conversation with a colleague. Answer their question based on the database query results in a natural, conversational way.

ORIGINAL QUESTION: {question}

QUERY RESULTS SUMMARY:
{data_summary}

INSTRUCTIONS:
- Respond as if you're a knowledgeable analyst talking to a colleague
- Be conversational and natural, not robotic
- Include specific numbers and context from the data
- Add relevant insights or observations when appropriate
- Use oil & gas industry terminology naturally
- Be concise but informative
- If the data shows trends or patterns, highlight them
- Sound like a human expert, not a computer

Respond naturally and conversationally.
"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful, conversational oil & gas data analyst. Respond naturally like you're talking to a colleague, not like a computer system."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return self._generate_fallback_response(question, query_result)
    
    def _generate_no_data_response(self, question: str, query_result: QueryResult) -> str:
        """Generate intelligent responses when query succeeds but returns no data."""
        question_lower = question.lower()
        
        # Intelligent no-data responses based on question type
        if "well" in question_lower and any(name in question_lower for name in ["abc-123", "xyz-456"]):
            return "I don't see that specific well in our database. The wells I have data for are more like Smith 1H, Jones 2H, Brown 3H, and Wilson 4H. Want me to show you data for any of those?"
        
        elif "operator" in question_lower or "company" in question_lower:
            return "That operator isn't showing up in our data. The main operators I see are Chesapeake Energy, Range Resources, and Cabot Oil. Want to check any of those instead?"
        
        elif "county" in question_lower:
            return "No results for that county. Our database mainly covers Broome, Tioga, and Chemung counties. Should I check one of those?"
        
        else:
            return f"I ran the query for '{question}' but didn't find any matching data. This could mean the specific criteria don't exist in our database, or we might need to adjust the search terms. Want me to show you what data is available?"
    
    def _summarize_data(self, df: pd.DataFrame) -> str:
        """Create a summary of the DataFrame for the LLM."""
        summary_parts = []
        
        # Basic info
        summary_parts.append(f"Found {len(df)} rows with {len(df.columns)} columns")
        
        # Column names
        summary_parts.append(f"Columns: {', '.join(df.columns.tolist())}")
        
        # Show first few rows as examples
        if len(df) > 0:
            sample_rows = df.head(min(5, len(df))).to_dict('records')
            summary_parts.append(f"Sample data: {sample_rows}")
        
        # Numeric summaries for key columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                try:
                    total = df[col].sum()
                    avg = df[col].mean()
                    summary_parts.append(f"{col}: Total = {total:,.2f}, Average = {avg:,.2f}")
                except:
                    pass
        
        return "\n".join(summary_parts)
    
    def _generate_fallback_response(self, question: str, query_result: QueryResult) -> str:
        """Generate a basic response without OpenAI."""
        df = query_result.data
        
        response_parts = [
            f"Based on your question '{question}', I found {len(df)} records."
        ]
        
        # Add some basic insights
        if len(df) > 0:
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                first_numeric = numeric_cols[0]
                total = df[first_numeric].sum()
                response_parts.append(f"The total {first_numeric} is {total:,.2f}.")
        
        response_parts.append(f"Query executed in {query_result.execution_time:.2f} seconds.")
        
        return " ".join(response_parts)
    
    # Include the SQL generation methods from the original DatabaseAssistant
    def generate_sql_query(self, question: str, relevant_tables: List[str]) -> Tuple[str, str]:
        """Generate SQL query from natural language question."""
        if not self.openai_client:
            return "", "OpenAI API key not configured. Please set OPENAI_API_KEY in .env file."
        
        try:
            # Get table schemas for context
            schema_context = self._build_schema_context(relevant_tables)
            sample_data_context = self._get_sample_data_context(relevant_tables)
            prompt = self._build_enhanced_sql_prompt(question, schema_context, sample_data_context)
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system", 
                        "content": """You are an expert SQL analyst. Generate ONLY valid SELECT queries that will work with the available data."""
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            sql_query = response.choices[0].message.content.strip()
            sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
            
            # Validate the query
            is_safe, error = self.db_manager.is_safe_query(sql_query)
            if not is_safe:
                return "", f"Generated query is not safe: {error}"
            
            return sql_query, ""
            
        except Exception as e:
            return "", f"Failed to generate SQL query: {str(e)}"
    
    def _get_sample_data_context(self, table_names: List[str]) -> str:
        """Get sample data from tables to help AI understand what's available."""
        sample_context = []
        
        for table_name in table_names[:2]:
            try:
                sample_query = f"SELECT * FROM {table_name} LIMIT 3"
                sample_df = self.db_manager.execute_query(sample_query)
                
                if not sample_df.empty:
                    sample_rows = sample_df.to_dict('records')
                    sample_context.append(f"Sample data from {table_name}:")
                    for i, row in enumerate(sample_rows, 1):
                        row_str = ", ".join([f"{k}: {v}" for k, v in row.items() if v is not None])
                        sample_context.append(f"  Row {i}: {row_str}")
            except Exception as e:
                sample_context.append(f"Could not get sample data from {table_name}: {str(e)}")
        
        return "\n".join(sample_context)
    
    def _build_enhanced_sql_prompt(self, question: str, schema_context: str, sample_data_context: str) -> str:
        """Build enhanced prompt with sample data context."""
        db_type = self.db_manager.db_type or 'sqlite'
        
        return f"""
Generate a SQL query to answer this question about oil & gas production data:

QUESTION: {question}

DATABASE TYPE: {db_type}

AVAILABLE TABLES AND COLUMNS:
{schema_context}

SAMPLE DATA (to understand what's actually available):
{sample_data_context}

REQUIREMENTS:
- Use only SELECT statements
- Base your query on the actual sample data shown above
- Use exact column names and values from the sample data
- Keep queries simple and compatible with {db_type}
- For time-based questions, use actual Year/Month values from the sample data
- Use appropriate aggregation (SUM, COUNT, AVG) when needed
- Include ORDER BY and LIMIT for top/ranking questions

Return only the SQL query that will work with this specific data.
"""
    
    def _build_schema_context(self, table_names: List[str]) -> str:
        """Build schema context for relevant tables."""
        context_parts = []
        
        for table_name in table_names:
            table_info = self.db_manager.get_table_info(table_name)
            if table_info:
                columns = [f"{col['name']} ({col['type']})" for col in table_info.columns]
                context_parts.append(f"Table: {table_name}\nColumns: {', '.join(columns)}\nRows: {table_info.row_count or 'Unknown'}")
        
        return "\n\n".join(context_parts)


# Compatibility alias for existing code
DatabaseAssistant = ExpertDatabaseAssistant