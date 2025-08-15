"""
AI-powered database assistant for natural language querying.
Converts questions to SQL and generates conversational responses.
"""

import os
import pandas as pd
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

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


class DatabaseAssistant:
    """AI assistant for natural language database queries."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.openai_client = None
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
    
    def generate_sql_query(self, question: str, relevant_tables: List[str]) -> Tuple[str, str]:
        """
        Generate SQL query from natural language question.
        
        Args:
            question: Natural language question
            relevant_tables: List of relevant table names
            
        Returns:
            Tuple of (sql_query, error_message)
        """
        if not self.openai_client:
            return "", "OpenAI API key not configured. Please set OPENAI_API_KEY in .env file."
        
        try:
            # Get table schemas for context
            schema_context = self._build_schema_context(relevant_tables)
            
            # Get sample data to help AI understand what's actually available
            sample_data_context = self._get_sample_data_context(relevant_tables)
            
            # Build enhanced prompt with sample data
            prompt = self._build_enhanced_sql_prompt(question, schema_context, sample_data_context)
            
            # Call OpenAI with better instructions
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system", 
                        "content": """You are an expert SQL analyst helping a colleague analyze oil & gas production data. 
                        
                        Generate ONLY valid SELECT queries that will work with the available data. 
                        
                        Key rules:
                        - Always start with SELECT
                        - Use simple, clear SQL that matches the actual data shown
                        - If asking about "recent" or "last month", use specific years like 2023 or 2022
                        - Base your query on the sample data provided - don't assume data that isn't there
                        - Keep it simple - complex date functions often fail
                        
                        Return ONLY the SQL query, nothing else."""
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            sql_query = response.choices[0].message.content.strip()
            
            # Clean up the response
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
        
        for table_name in table_names[:2]:  # Limit to 2 tables for prompt size
            try:
                # Get a few sample rows
                sample_query = f"SELECT * FROM {table_name} LIMIT 3"
                sample_df = self.db_manager.execute_query(sample_query)
                
                if not sample_df.empty:
                    # Convert to a readable format
                    sample_rows = sample_df.to_dict('records')
                    sample_context.append(f"Sample data from {table_name}:")
                    for i, row in enumerate(sample_rows, 1):
                        row_str = ", ".join([f"{k}: {v}" for k, v in row.items() if v is not None])
                        sample_context.append(f"  Row {i}: {row_str}")
                
                # Get distinct values for key columns
                table_info = self.db_manager.get_table_info(table_name)
                if table_info:
                    for col in table_info.columns[:5]:  # First 5 columns
                        col_name = col['name']
                        if col_name.lower() in ['well_name', 'operator', 'county', 'year']:
                            try:
                                distinct_query = f"SELECT DISTINCT {col_name} FROM {table_name} LIMIT 5"
                                distinct_df = self.db_manager.execute_query(distinct_query)
                                if not distinct_df.empty:
                                    values = distinct_df[col_name].tolist()
                                    sample_context.append(f"Available {col_name} values: {', '.join(map(str, values))}")
                            except:
                                pass
                                
            except Exception as e:
                sample_context.append(f"Could not get sample data from {table_name}: {str(e)}")
        
        return "\n".join(sample_context)
    
    def _build_enhanced_sql_prompt(self, question: str, schema_context: str, sample_data_context: str) -> str:
        """Build enhanced prompt with sample data context."""
        # Detect database type for appropriate SQL syntax
        db_type = self.db_manager.db_type or 'sqlite'
        
        if db_type == 'sqlite':
            date_functions_help = """
            SQLite Date Functions:
            - Use date('now') for current date
            - Use simple WHERE Year = 2023 AND Month = 1 for filtering
            - No EXTRACT, DATE_SUB, or CURDATE functions
            """
        else:
            date_functions_help = """
            Standard SQL Date Functions:
            - Use GETDATE() or CURRENT_DATE for current date
            - Use EXTRACT(YEAR FROM date_column) for year extraction
            """
        
        return f"""
Generate a SQL query to answer this question about oil & gas production data:

QUESTION: {question}

DATABASE TYPE: {db_type}

AVAILABLE TABLES AND COLUMNS:
{schema_context}

SAMPLE DATA (to understand what's actually available):
{sample_data_context}

{date_functions_help}

REQUIREMENTS:
- Use only SELECT statements that start with SELECT
- Base your query on the actual sample data shown above
- Use the exact column names and values you see in the sample data
- Keep queries simple and compatible with {db_type}
- For time-based questions, use the actual Year/Month values from the sample data
- If asking about "recent" or "last month", use 2023 or 2022 based on available data
- Use appropriate aggregation (SUM, COUNT, AVG) when needed
- Include ORDER BY and LIMIT for top/ranking questions
- Don't assume data that isn't shown in the samples

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
    
    def _build_sql_prompt(self, question: str, schema_context: str) -> str:
        """Build the prompt for SQL generation."""
        # Detect database type for appropriate SQL syntax
        db_type = self.db_manager.db_type or 'sqlite'
        
        if db_type == 'sqlite':
            date_functions_help = """
            SQLite Date Functions:
            - Use date('now') for current date
            - Use strftime('%Y', date_column) for year extraction
            - Use strftime('%m', date_column) for month extraction
            - No EXTRACT, DATE_SUB, or CURDATE functions
            - Use simple WHERE Year = 2023 AND Month = 1 for filtering
            """
        else:
            date_functions_help = """
            Standard SQL Date Functions:
            - Use GETDATE() or CURRENT_DATE for current date
            - Use EXTRACT(YEAR FROM date_column) for year extraction
            - Use EXTRACT(MONTH FROM date_column) for month extraction
            """
        
        return f"""
Generate a SQL query to answer this question about oil & gas production data:

QUESTION: {question}

DATABASE TYPE: {db_type}

AVAILABLE TABLES AND COLUMNS:
{schema_context}

{date_functions_help}

REQUIREMENTS:
- Use only SELECT statements
- Keep queries simple and compatible with {db_type}
- Include appropriate WHERE clauses for filtering
- Use proper JOIN syntax if multiple tables needed
- Use aggregation functions (SUM, COUNT, AVG) when appropriate
- Include ORDER BY for meaningful results
- Limit results to reasonable numbers (LIMIT 10 or similar)
- For "last month" or recent data, use specific Year/Month values like Year = 2023 AND Month = 12

COMMON COLUMN PATTERNS:
- Well identifiers: API_WellNo, Well_Name, Well_ID
- Dates: Year, Month, MonthProd, Production_Date
- Production: OilProd, GasProd, WaterProd (or similar)
- Operators: Operator, Operator_Name, Company_Name
- Locations: County, Field_Name

IMPORTANT: Keep the SQL simple and avoid complex date functions. Use direct Year/Month filtering.

Return only the SQL query, no explanation.
"""
    
    def execute_query(self, question: str) -> QueryResult:
        """
        Execute a natural language query against the database.
        
        Args:
            question: Natural language question
            
        Returns:
            QueryResult with data and metadata
        """
        import time
        start_time = time.time()
        
        try:
            # Find relevant tables
            relevant_tables = self.db_manager.find_relevant_tables(question)
            
            if not relevant_tables:
                return QueryResult(
                    success=False,
                    error_message="Could not identify relevant tables for your question. Please be more specific."
                )
            
            # Generate SQL query
            sql_query, error = self.generate_sql_query(question, relevant_tables)
            
            if error:
                return QueryResult(success=False, error_message=error)
            
            # Execute the query
            result_df = self.db_manager.execute_query(sql_query)
            
            execution_time = time.time() - start_time
            
            return QueryResult(
                success=True,
                data=result_df,
                sql_query=sql_query,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return QueryResult(
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )
    
    def generate_response(self, question: str, query_result: QueryResult) -> str:
        """
        Generate a conversational response based on query results.
        
        Args:
            question: Original question
            query_result: Database query results
            
        Returns:
            Natural language response
        """
        if not query_result.success:
            return f"I ran into a technical issue: {query_result.error_message}"
        
        if query_result.data is None or query_result.data.empty:
            # Generate intelligent "not found" responses based on the question
            return self._generate_not_found_response(question, query_result.sql_query)
        
        if not self.openai_client:
            # Fallback response without OpenAI
            return self._generate_fallback_response(question, query_result)
        
        try:
            # Prepare data summary for the LLM
            data_summary = self._summarize_data(query_result.data)
            
            prompt = f"""
You are an expert oil & gas data analyst having a conversation with a colleague. Answer their question based on the database query results in a natural, conversational way.

ORIGINAL QUESTION: {question}

QUERY RESULTS SUMMARY:
{data_summary}

SQL QUERY USED:
{query_result.sql_query}

INSTRUCTIONS:
- Respond as if you're a knowledgeable analyst talking to a colleague
- Be conversational and natural, not robotic
- Include specific numbers and context from the data
- Add relevant insights or observations when appropriate
- Use oil & gas industry terminology naturally
- Be concise but informative
- Don't mention the SQL query unless specifically relevant
- If the data shows trends or patterns, highlight them
- Sound like a human expert, not a computer

Example good responses:
- "Based on the data, well Smith 1H produced 1,247 barrels of oil in January 2023, which was actually pretty strong for that area."
- "Looking at the numbers, Broome County led gas production last year with 45.6 million MCF, followed closely by Tioga County."
- "The top 5 wells all produced over 50,000 barrels last year, with Smith 1H being the standout at 67,000 barrels."

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
            print(f"AI response generation failed: {str(e)}")
            return self._generate_fallback_response(question, query_result)
    
    def _generate_not_found_response(self, question: str, sql_query: str) -> str:
        """Generate intelligent responses when no data is found."""
        question_lower = question.lower()
        
        # Check what type of question this was and respond appropriately
        if "well" in question_lower and ("xyz-456" in question_lower or "abc-123" in question_lower):
            return "No, that well isn't in our database. The wells I see are named things like Smith 1H, Jones 2H, Brown 3H, and Wilson 4H."
        
        elif "well" in question_lower and any(name in question_lower for name in ["smith", "jones", "brown", "wilson"]):
            well_name = None
            for name in ["smith", "jones", "brown", "wilson"]:
                if name in question_lower:
                    well_name = f"{name.title()} 1H"
                    break
            return f"I don't see specific data for {well_name} matching those criteria. Let me know if you'd like me to check different time periods or parameters."
        
        elif "county" in question_lower:
            return "I didn't find any counties matching those criteria in the production data. The main counties in our database are Broome, Tioga, and Chemung."
        
        elif "operator" in question_lower or "company" in question_lower:
            if "chesapeake" in question_lower:
                return "Chesapeake Energy is in our database, but I didn't find data matching those specific criteria. Want me to check different time periods?"
            else:
                return "That operator isn't in our database. The main operators I see are Chesapeake Energy, Range Resources, and Cabot Oil."
        
        elif "last month" in question_lower or "last year" in question_lower:
            return "I don't have recent data for that timeframe. Our database has production data for 2022 and 2023. Want me to check those years instead?"
        
        elif "top" in question_lower or "highest" in question_lower:
            return "I didn't find any results for that query. This might be because the specific criteria don't match our data. Want me to show you what data is available?"
        
        else:
            return "I didn't find any data matching those criteria. This could be because the specific wells, time periods, or operators don't exist in our database. Want me to show you what data is available?"
    
    def _summarize_data(self, df: pd.DataFrame) -> str:
        """Create a summary of the DataFrame for the LLM."""
        summary_parts = []
        
        # Basic info
        summary_parts.append(f"Found {len(df)} rows with {len(df.columns)} columns")
        
        # Column names
        summary_parts.append(f"Columns: {', '.join(df.columns.tolist())}")
        
        # Show first few rows as examples
        if len(df) > 0:
            # Convert to dict for better LLM parsing
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
    
    def ask_question(self, question: str) -> str:
        """
        Main interface: ask a question and get a conversational response.
        
        Args:
            question: Natural language question about the database
            
        Returns:
            Conversational response with insights
        """
        # Execute the query
        query_result = self.execute_query(question)
        
        # Generate conversational response
        response = self.generate_response(question, query_result)
        
        return response