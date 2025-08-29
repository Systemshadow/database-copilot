"""
Enhanced AI Assistant with Multi-Table Support and Structured Responses
Quick fix version - working for client demo.
"""

import os
import pandas as pd
import streamlit as st
from typing import Optional, List, Dict, Tuple, Any
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
    """Enhanced result with structured data support."""
    success: bool
    data: Optional[pd.DataFrame] = None
    sql_query: Optional[str] = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    display_as_table: bool = False
    table_title: Optional[str] = None


class MultiTableDatabaseAssistant:
    """Enhanced AI assistant with multi-table support and structured responses."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.openai_client = None
        self.conversation_context = []
        self._initialize_openai()
        
        # Define table relationships
        self.table_relationships = {
            'production': {
                'join_key': 'API_WellNo',
                'related_tables': ['wells'],
                'description': 'Monthly production data by well'
            },
            'wells': {
                'join_key': 'API_WellNo', 
                'related_tables': ['production'],
                'description': 'Master well data with technical details'
            }
        }
    
    def _initialize_openai(self):
        """Initialize OpenAI client if API key is available."""
        # Try Streamlit secrets first (for cloud deployment)
        api_key = None
        
        try:
            if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
                api_key = st.secrets['OPENAI_API_KEY']
        except:
            # Secrets not available (local development)
            pass
        
        if not api_key:
            # Fallback to environment variable (for local testing)
            api_key = os.getenv('OPENAI_API_KEY')
        
        if OPENAI_AVAILABLE and api_key and api_key != 'your_openai_api_key_here':
            try:
                self.openai_client = OpenAI(api_key=api_key)
            except Exception as e:
                print(f"OpenAI initialization failed: {str(e)}")
                self.openai_client = None
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Main interface: ask a question and get a response with structured data support."""
        try:
            # Add to conversation context
            self.conversation_context.append({"role": "user", "content": question})
            
            # Execute the query with full error handling
            query_result = self._execute_enhanced_query(question)
            
            # Generate response (text + structured data)
            response_data = self._generate_enhanced_response(question, query_result)
            
            # Add response to context
            self.conversation_context.append({"role": "assistant", "content": response_data['text']})
            
            return response_data
            
        except Exception as e:
            # Final safety net
            return {
                'text': f"I encountered an unexpected error while processing your question. Let me know if you'd like to try rephrasing it.",
                'table_data': None,
                'table_title': None
            }
    
    def _execute_enhanced_query(self, question: str) -> QueryResult:
        """Execute query with multi-table awareness."""
        import time
        start_time = time.time()
        
        try:
            # Determine if this needs multi-table support
            needs_join = self._requires_cross_table_data(question)
            
            # Find relevant tables
            relevant_tables = self._find_relevant_tables_enhanced(question)
            
            if not relevant_tables:
                return QueryResult(
                    success=False,
                    error_message="Could not identify relevant tables",
                    execution_time=time.time() - start_time
                )
            
            # Generate enhanced SQL
            sql_query, error = self._generate_enhanced_sql(question, relevant_tables, needs_join)
            
            if error:
                return QueryResult(
                    success=False, 
                    error_message=f"sql_generation_failed: {error}",
                    execution_time=time.time() - start_time
                )
            
            # Execute the query
            result_df = self.db_manager.execute_query(sql_query)
            
            # Determine if response should be displayed as table
            display_as_table = self._enhanced_should_display_as_table(question, result_df)
            table_title = self._generate_table_title(question, result_df) if display_as_table else None
            
            return QueryResult(
                success=True,
                data=result_df,
                sql_query=sql_query,
                execution_time=time.time() - start_time,
                display_as_table=display_as_table,
                table_title=table_title
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return QueryResult(
                success=False,
                error_message=f"execution_error: {str(e)}",
                execution_time=execution_time
            )
    
    def _requires_cross_table_data(self, question: str) -> bool:
        """Determine if question requires joining multiple tables."""
        question_lower = question.lower()
        
        # Production + wells data indicators
        cross_table_indicators = [
            'depth', 'formation', 'permit', 'drilled', 'completed',
            'horizontal', 'vertical', 'latitude', 'longitude',
            'status', 'operator', 'spudded', 'location',
            'by formation', 'by depth', 'by status', 'by operator',
            'with their', 'and their', 'along with'
        ]
        
        return any(indicator in question_lower for indicator in cross_table_indicators)
    
    def _find_relevant_tables_enhanced(self, question: str) -> List[str]:
        """Enhanced table finding with multi-table awareness."""
        question_lower = question.lower()
        relevant_tables = set()
        
        # Production data indicators
        production_indicators = [
            'production', 'prod', 'gas', 'oil', 'water', 
            'producing', 'monthly', 'total', 'volume'
        ]
        
        # Wells master data indicators
        wells_indicators = [
            'well', 'depth', 'formation', 'permit', 'drill',
            'location', 'coordinate', 'status', 'company',
            'operator', 'horizontal', 'vertical'
        ]
        
        # Check for production data needs
        if any(indicator in question_lower for indicator in production_indicators):
            relevant_tables.add('production')
        
        # Check for wells data needs
        if any(indicator in question_lower for indicator in wells_indicators):
            relevant_tables.add('wells')
        
        # If we have both types of indicators, we need both tables
        analysis_indicators = ['compare', 'by', 'with', 'and', 'total', 'summary']
        
        if len(relevant_tables) == 1 and any(indicator in question_lower for indicator in analysis_indicators):
            relevant_tables.update(['production', 'wells'])
        
        # Default to both tables if we can't determine specifically
        if not relevant_tables:
            relevant_tables.update(['production', 'wells'])
        
        return list(relevant_tables)
    
    def _generate_enhanced_sql(self, question: str, relevant_tables: List[str], needs_join: bool) -> Tuple[str, str]:
        """Generate SQL with multi-table support."""
        if not self.openai_client:
            return "", "OpenAI API key not configured"
        
        try:
            # Get enhanced schema context
            schema_context = self._build_enhanced_schema_context(relevant_tables)
            sample_data_context = self._get_enhanced_sample_data(relevant_tables)
            relationship_context = self._build_relationship_context(relevant_tables)
            
            prompt = f"""
Generate a SQL query for this oil & gas data question:

QUESTION: {question}

AVAILABLE TABLES AND RELATIONSHIPS:
{schema_context}

SAMPLE DATA:
{sample_data_context}

TABLE RELATIONSHIPS:
{relationship_context}

QUERY GUIDELINES:
- Use proper JOINs when data from multiple tables is needed
- Join production and wells tables on API_WellNo when appropriate  
- For production totals, use SUM aggregation
- For counts, use COUNT
- Include ORDER BY for top/ranking questions
- Use LIMIT for performance (1000 max)
- Always use proper column names from the schema above

Generate only the SQL query that answers the question.
"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert SQL analyst. Generate clean, efficient SQL queries for oil & gas databases."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
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
    
    def _build_enhanced_schema_context(self, table_names: List[str]) -> str:
        """Build enhanced schema context with relationships."""
        context_parts = []
        
        for table_name in table_names:
            table_info = self.db_manager.get_table_info(table_name)
            if table_info:
                columns = [f"{col['name']} ({col['type']})" for col in table_info.columns]
                context_parts.append(f"""
TABLE: {table_name}
Columns: {', '.join(columns)}
Rows: {table_info.row_count or 'Unknown'}
Purpose: {self.table_relationships.get(table_name, {}).get('description', 'Data table')}
""")
        
        return "\n".join(context_parts)
    
    def _get_enhanced_sample_data(self, table_names: List[str]) -> str:
        """Get enhanced sample data showing relationships."""
        sample_context = []
        
        for table_name in table_names[:2]:
            try:
                if table_name == 'production':
                    sample_query = """
                    SELECT API_WellNo, County, Operator, Well_Name, 
                           GasProd, OilProd, WaterProd, Year, Month 
                    FROM production 
                    WHERE GasProd > 0 OR OilProd > 0
                    ORDER BY Year DESC, GasProd DESC 
                    LIMIT 3
                    """
                elif table_name == 'wells':
                    sample_query = """
                    SELECT API_WellNo, Well_Name, Company_name, County, 
                           Well_Status, Well_Type, Producing_formation,
                           True_vertical_depth, Surface_latitude, Surface_longitude
                    FROM wells 
                    WHERE Well_Status != '' 
                    LIMIT 3
                    """
                else:
                    sample_query = f"SELECT * FROM {table_name} LIMIT 3"
                
                sample_df = self.db_manager.execute_query(sample_query)
                
                if not sample_df.empty:
                    sample_rows = sample_df.to_dict('records')
                    sample_context.append(f"Sample from {table_name}:")
                    for i, row in enumerate(sample_rows, 1):
                        row_str = ", ".join([f"{k}: {v}" for k, v in list(row.items())[:6] if v is not None])
                        sample_context.append(f"  Row {i}: {row_str}")
                        
            except Exception as e:
                sample_context.append(f"Could not sample {table_name}: {str(e)}")
        
        return "\n".join(sample_context)
    
    def _build_relationship_context(self, table_names: List[str]) -> str:
        """Build context about table relationships."""
        if len(table_names) < 2:
            return "Single table query - no joins needed"
        
        relationships = []
        if 'production' in table_names and 'wells' in table_names:
            relationships.append("""
JOIN production p and wells w ON p.API_WellNo = w.API_WellNo
- This links monthly production data to well master data
- Allows queries combining production metrics with well attributes
""")
        
        return "\n".join(relationships)
    
    def _enhanced_should_display_as_table(self, question: str, df: pd.DataFrame) -> bool:
        """Enhanced table detection - more aggressive about showing tables."""
        if df is None:
            print("DEBUG: No table - df is None")
            return False
            
        if hasattr(df, 'empty') and df.empty:
            print("DEBUG: No table - df is empty")
            return False
        
        question_lower = question.lower()
        
        # ALWAYS show as table for these explicit requests
        explicit_table_requests = [
            'show me', 'list', 'display', 'table', 'compare',
            'top', 'ranking', 'wells', 'production', 'summary',
            'by county', 'by operator', 'by formation', 'total'
        ]
        
        for indicator in explicit_table_requests:
            if indicator in question_lower:
                print(f"DEBUG: Table requested due to indicator: {indicator}")
                return True
        
        # Show as table if we have multiple rows OR multiple meaningful columns
        if len(df) > 1:
            print(f"DEBUG: Table requested due to multiple rows: {len(df)}")
            return True
        
        if len(df.columns) >= 3:
            print(f"DEBUG: Table requested due to multiple columns: {len(df.columns)}")
            return True
        
        # Default: if we got this far and have data, probably show it
        if len(df) > 0:
            print(f"DEBUG: Table requested as fallback - has data")
            return True
        
        print(f"DEBUG: No table - no conditions met")
        return False
    
    def _should_display_as_table(self, question: str, df: pd.DataFrame) -> bool:
        """Determine if response should be displayed as a structured table."""
        return self._enhanced_should_display_as_table(question, df)
    
    def _generate_table_title(self, question: str, df: pd.DataFrame) -> str:
        """Generate an appropriate title for the table."""
        question_lower = question.lower()
        
        if 'top' in question_lower and 'wells' in question_lower:
            return "Top Performing Wells"
        elif 'production' in question_lower and 'county' in question_lower:
            return "Production by County"
        elif 'compare' in question_lower:
            return "Comparison Results"
        elif 'wells' in question_lower and 'formation' in question_lower:
            return "Wells by Formation"
        elif len(df) > 10:
            return f"Query Results ({len(df)} records)"
        else:
            return "Query Results"
    
    def _generate_enhanced_response(self, question: str, query_result: QueryResult) -> Dict[str, Any]:
        """Generate response with both text and structured data - FIXED VERSION."""
        
        try:
            # Debug logging
            print(f"DEBUG: query_result.success = {query_result.success}")
            print(f"DEBUG: query_result.data type = {type(query_result.data)}")
            if query_result.data is not None:
                print(f"DEBUG: query_result.data shape = {query_result.data.shape}")
                print(f"DEBUG: query_result.data columns = {list(query_result.data.columns)}")
            print(f"DEBUG: query_result.display_as_table = {query_result.display_as_table}")
            print(f"DEBUG: query_result.table_title = {query_result.table_title}")
            
            # Handle failures
            if not query_result.success:
                return {
                    'text': self._handle_query_failure(question, query_result.error_message),
                    'table_data': None,
                    'table_title': None
                }
            
            # Handle no data - be more permissive here
            if query_result.data is None:
                return {
                    'text': self._handle_no_data_found(question),
                    'table_data': None,
                    'table_title': None
                }
            
            # Handle empty DataFrame
            if hasattr(query_result.data, 'empty') and query_result.data.empty:
                return {
                    'text': self._handle_no_data_found(question),
                    'table_data': None,
                    'table_title': None
                }
            
            # Generate text response
            text_response = self._generate_text_response(question, query_result)
            
            # ENHANCED TABLE DETECTION - be more aggressive about showing tables
            should_show_table = self._enhanced_should_display_as_table(question, query_result.data)
            
            # FIXED: Always create DataFrame copy to avoid reference issues
            table_data = None
            table_title = None
            
            if should_show_table and query_result.data is not None:
                # Create a clean copy of the DataFrame
                table_data = query_result.data.copy()
                table_title = query_result.table_title or self._generate_table_title(question, table_data)
                
                print(f"DEBUG: Table will be displayed - shape: {table_data.shape}")
            else:
                print(f"DEBUG: No table display - should_show_table={should_show_table}")
            
            # Return structured response
            response_data = {
                'text': text_response,
                'table_data': table_data,
                'table_title': table_title
            }
            
            print(f"DEBUG: Final response keys: {list(response_data.keys())}")
            print(f"DEBUG: Final table_data type: {type(response_data['table_data'])}")
            
            return response_data
            
        except Exception as e:
            print(f"ERROR in _generate_enhanced_response: {str(e)}")
            print(f"ERROR traceback: {traceback.format_exc()}")
            return {
                'text': f"I found some data but encountered an issue formatting the response. The query executed successfully but I had trouble presenting the results.",
                'table_data': None,
                'table_title': None
            }
    
    def _generate_text_response(self, question: str, query_result: QueryResult) -> str:
        """Generate conversational text response - FIXED to list all results."""
        if not self.openai_client:
            return self._generate_fallback_response(question, query_result)
        
        try:
            # Prepare data summary for the LLM
            data_summary = self._summarize_enhanced_data(query_result.data)
            
            # Check if user asked for a specific list (top X, show me X, etc.)
            question_lower = question.lower()
            is_list_request = any(pattern in question_lower for pattern in [
                'top ', 'show me', 'list', 'display', 'give me'
            ])
            
            # Get actual data to include in response if it's a list request
            actual_results = ""
            if is_list_request and not query_result.data.empty:
                # Include first 15 rows of actual data in the prompt
                results_sample = query_result.data.head(15)
                actual_results = f"\nACTUAL RESULTS TO INCLUDE IN RESPONSE:\n{results_sample.to_dict('records')}"
            
            prompt = f"""
You are an expert oil & gas analyst having a conversation with a colleague about data from both production records and wells master data.

QUESTION: {question}

DATA ANALYSIS:
{data_summary}
{actual_results}

CONTEXT:
- Our database now contains both monthly production data and comprehensive well master data
- We can analyze production performance alongside well characteristics like depth, formation, location, etc.
- This enables much richer insights combining operational and technical data

RESPONSE STYLE:
- Be conversational and expert-level
- If the user asked for a specific list (top X, show X), LIST ALL THE ACTUAL RESULTS - don't use "..." or placeholders
- When listing results, include the actual well names, operators, and numbers from the data
- Highlight key findings from the data
- When appropriate, mention that detailed data is shown in the table below
- Include relevant oil & gas industry context
- Be complete but engaging

CRITICAL: If this is a "top X" or "show me X" request, provide the COMPLETE list with actual well names and numbers. Do NOT use "..." or placeholders - list every single result requested.

Respond naturally as a knowledgeable colleague would.
"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a knowledgeable oil & gas data analyst. When users ask for lists, provide complete actual results, never use '...' placeholders."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,  # Increased to allow for full lists
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return self._generate_fallback_response(question, query_result)
    
    def _summarize_enhanced_data(self, df: pd.DataFrame) -> str:
        """Enhanced data summary for multi-table results."""
        summary_parts = []
        
        # Basic info
        summary_parts.append(f"Dataset: {len(df)} rows, {len(df.columns)} columns")
        
        # Column analysis
        production_cols = [col for col in df.columns if any(prod in col.lower() for prod in ['gas', 'oil', 'water', 'prod'])]
        well_cols = [col for col in df.columns if any(well in col.lower() for well in ['depth', 'formation', 'status', 'type'])]
        
        if production_cols:
            summary_parts.append(f"Production metrics: {', '.join(production_cols)}")
        if well_cols:
            summary_parts.append(f"Well attributes: {', '.join(well_cols)}")
        
        # Sample data
        if len(df) > 0:
            sample_rows = df.head(3).to_dict('records')
            summary_parts.append(f"Sample data: {sample_rows}")
        
        # Key statistics
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols[:3]:
            if col in df.columns:
                total = df[col].sum()
                avg = df[col].mean()
                summary_parts.append(f"{col}: Total={total:,.0f}, Average={avg:,.0f}")
        
        return "\n".join(summary_parts)
    
    def _handle_query_failure(self, question: str, error_message: str) -> str:
        """Handle query failures conversationally."""
        if "relative_timeframe" in error_message:
            return "I need specific dates to query the data. Our database covers production data through 2024. What specific timeframe would you like to analyze?"
        elif "no_tables_found" in error_message:
            return "I'm having trouble identifying the right data for your question. Could you be more specific about what you're looking for? I can help with production data, well information, or combined analysis."
        else:
            return "I ran into a technical issue processing your query. Could you try rephrasing your question? I can help analyze production data, well characteristics, or combined insights."
    
    def _handle_no_data_found(self, question: str) -> str:
        """Handle no data found scenarios."""
        return "I didn't find any data matching those criteria. This could be because the specific wells, operators, time periods, or conditions don't exist in our database. Would you like me to show you what data is available?"
    
    def _generate_fallback_response(self, question: str, query_result: QueryResult) -> str:
        """Fallback response when OpenAI is unavailable."""
        df = query_result.data
        
        response_parts = [
            f"Found {len(df)} records for your question about the data."
        ]
        
        # Add basic insights
        if len(df) > 0:
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                first_numeric = numeric_cols[0]
                total = df[first_numeric].sum()
                response_parts.append(f"Total {first_numeric}: {total:,.0f}")
        
        if query_result.display_as_table:
            response_parts.append("Detailed results are shown in the table below.")
        
        return " ".join(response_parts)


# Backward compatibility
DatabaseAssistant = MultiTableDatabaseAssistant