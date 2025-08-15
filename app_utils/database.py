"""
Database connection and schema management for Database Copilot.
Supports multiple database types with intelligent schema discovery.
"""

import os
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import create_engine, text, inspect
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class TableInfo:
    """Information about a database table."""
    name: str
    schema: str
    columns: List[Dict[str, str]]
    row_count: Optional[int] = None
    description: Optional[str] = None


@dataclass
class ColumnInfo:
    """Information about a database column."""
    name: str
    type: str
    nullable: bool
    primary_key: bool = False
    foreign_key: Optional[str] = None


class DatabaseManager:
    """Manages database connections and schema discovery."""
    
    def __init__(self):
        self.engine = None
        self.connection = None
        self.db_type = None
        self.schema_cache = {}
        
    def connect(self) -> bool:
        """
        Establish database connection based on environment variables.
        
        Returns:
            bool: True if connection successful
        """
        try:
            # Check for full connection string first
            conn_string = os.getenv('DATABASE_CONNECTION_STRING')
            
            if conn_string:
                self.engine = create_engine(conn_string)
            else:
                # Build connection string from components
                db_type = os.getenv('DATABASE_TYPE', 'sqlserver').lower()
                host = os.getenv('DATABASE_HOST')
                port = os.getenv('DATABASE_PORT')
                database = os.getenv('DATABASE_NAME')
                username = os.getenv('DATABASE_USER')
                password = os.getenv('DATABASE_PASSWORD')
                trusted = os.getenv('DATABASE_TRUSTED_CONNECTION', '').lower() == 'yes'
                
                if not host or not database:
                    raise ValueError("DATABASE_HOST and DATABASE_NAME are required")
                
                conn_string = self._build_connection_string(
                    db_type, host, port, database, username, password, trusted
                )
                
                self.engine = create_engine(conn_string)
            
            # Test connection
            self.connection = self.engine.connect()
            self.db_type = self.engine.dialect.name
            
            return True
            
        except Exception as e:
            print(f"Database connection error: {str(e)}")
            return False
    
    def _build_connection_string(self, db_type: str, host: str, port: str, 
                               database: str, username: str, password: str, 
                               trusted: bool = False) -> str:
        """Build database connection string based on type."""
        
        if db_type == 'sqlserver':
            if trusted:
                return f"mssql+pyodbc://{host}:{port}/{database}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
            else:
                return f"mssql+pyodbc://{username}:{password}@{host}:{port}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
        
        elif db_type == 'postgresql':
            return f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"
        
        elif db_type == 'mysql':
            return f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
        
        elif db_type == 'oracle':
            return f"oracle+cx_oracle://{username}:{password}@{host}:{port}/{database}"
        
        elif db_type == 'sqlite':
            # For local testing with SQLite
            return f"sqlite:///{database}"
        
        elif db_type == 'duckdb':
            # For local testing with DuckDB
            return f"duckdb:///{database}"
        
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
    
    def discover_schema(self) -> List[TableInfo]:
        """
        Discover database schema - tables, columns, types.
        
        Returns:
            List of TableInfo objects
        """
        if not self.connection:
            raise RuntimeError("Not connected to database")
        
        try:
            inspector = inspect(self.engine)
            tables = []
            
            # Get all tables
            table_names = inspector.get_table_names()
            
            for table_name in table_names:
                try:
                    # Get column information
                    columns_info = inspector.get_columns(table_name)
                    columns = []
                    
                    for col in columns_info:
                        columns.append({
                            'name': col['name'],
                            'type': str(col['type']),
                            'nullable': col['nullable'],
                            'primary_key': col.get('primary_key', False)
                        })
                    
                    # Get row count (sample for large tables)
                    try:
                        count_query = text(f"SELECT COUNT(*) FROM {table_name}")
                        result = self.connection.execute(count_query)
                        row_count = result.scalar()
                    except:
                        row_count = None
                    
                    table_info = TableInfo(
                        name=table_name,
                        schema='dbo',  # Default schema, could be enhanced
                        columns=columns,
                        row_count=row_count
                    )
                    
                    tables.append(table_info)
                    
                except Exception as e:
                    print(f"Error processing table {table_name}: {str(e)}")
                    continue
            
            # Cache the schema
            self.schema_cache = {table.name: table for table in tables}
            return tables
            
        except Exception as e:
            raise RuntimeError(f"Schema discovery failed: {str(e)}")
    
    def get_table_info(self, table_name: str) -> Optional[TableInfo]:
        """Get detailed information about a specific table."""
        return self.schema_cache.get(table_name)
    
    def execute_query(self, sql: str) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame.
        
        Args:
            sql: SQL query to execute
            
        Returns:
            DataFrame with results
        """
        if not self.connection:
            raise RuntimeError("Not connected to database")
        
        try:
            # Add safety limit if not present
            sql_upper = sql.upper().strip()
            if not any(keyword in sql_upper for keyword in ['TOP', 'LIMIT', 'ROWNUM']):
                if sql.rstrip().endswith(';'):
                    sql = sql.rstrip()[:-1]
                
                # Add appropriate limit syntax based on database type
                if self.db_type == 'sqlserver':
                    # Add TOP to SELECT
                    sql = sql.replace('SELECT', 'SELECT TOP 1000', 1)
                elif self.db_type in ['postgresql', 'mysql', 'duckdb']:
                    sql += ' LIMIT 1000'
                elif self.db_type == 'oracle':
                    sql += ' AND ROWNUM <= 1000'
            
            result = pd.read_sql(sql, self.connection)
            return result
            
        except Exception as e:
            raise RuntimeError(f"Query execution failed: {str(e)}")
    
    def is_safe_query(self, sql: str) -> Tuple[bool, str]:
        """
        Check if SQL query is safe (read-only).
        
        Returns:
            Tuple of (is_safe, error_message)
        """
        sql_upper = sql.upper().strip()
        
        # Remove comments
        sql_clean = ' '.join(line.split('--')[0] for line in sql_upper.split('\n'))
        
        # Dangerous keywords
        dangerous_keywords = [
            'INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'CREATE', 
            'TRUNCATE', 'EXEC', 'EXECUTE', 'MERGE', 'BULK', 'BACKUP',
            'RESTORE', 'GRANT', 'REVOKE', 'sp_', 'xp_'
        ]
        
        for keyword in dangerous_keywords:
            if keyword in sql_clean:
                return False, f"Dangerous keyword '{keyword}' detected. Only SELECT queries allowed."
        
        # Must contain SELECT
        if 'SELECT' not in sql_clean:
            return False, "Query must contain SELECT statement"
        
        return True, ""
    
    def find_relevant_tables(self, question: str) -> List[str]:
        """
        Find tables likely relevant to the user's question.
        
        Args:
            question: User's natural language question
            
        Returns:
            List of relevant table names
        """
        question_lower = question.lower()
        relevant_tables = []
        
        # Keywords that suggest specific tables
        table_keywords = {
            'production': ['production', 'prod', 'well_production'],
            'wells': ['wells', 'well', 'wellbore'],
            'operators': ['operators', 'operator', 'company'],
            'facilities': ['facilities', 'facility', 'plant'],
            'reserves': ['reserves', 'reserve'],
            'completion': ['completion', 'frac', 'stimulation']
        }
        
        # Check schema cache for matching tables
        for table_name, table_info in self.schema_cache.items():
            table_lower = table_name.lower()
            
            # Direct table name mention
            if table_lower in question_lower:
                relevant_tables.append(table_name)
                continue
            
            # Keyword matching
            for category, keywords in table_keywords.items():
                if any(keyword in question_lower for keyword in keywords):
                    if any(keyword in table_lower for keyword in keywords):
                        relevant_tables.append(table_name)
                        break
        
        # If no specific matches, include tables with common oil & gas columns
        if not relevant_tables:
            for table_name, table_info in self.schema_cache.items():
                column_names = [col['name'].lower() for col in table_info.columns]
                
                # Look for common oil & gas columns
                og_columns = [
                    'well_id', 'api', 'operator', 'production_date', 
                    'oil', 'gas', 'water', 'county', 'field'
                ]
                
                if any(og_col in ' '.join(column_names) for og_col in og_columns):
                    relevant_tables.append(table_name)
        
        return relevant_tables[:5]  # Limit to top 5 relevant tables
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
        if self.engine:
            self.engine.dispose()


# Global database manager instance
db_manager = DatabaseManager()