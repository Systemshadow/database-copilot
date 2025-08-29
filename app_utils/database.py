"""
Database connection and schema management for Database Copilot.
PRODUCTION-READY: Handles both local .env and Streamlit Cloud secrets.
Supports multiple database types with intelligent schema discovery.
"""

import os
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import create_engine, text, inspect
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
from pathlib import Path

# Try to import streamlit for secrets (will fail in non-Streamlit environments)
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

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
        
    def _get_config_value(self, key: str, default: str = None) -> Optional[str]:
        """
        Get configuration value from Streamlit secrets or environment variables.
        Streamlit Cloud: Uses st.secrets
        Local development: Uses .env file
        """
        # Try Streamlit secrets first (for cloud deployment)
        if STREAMLIT_AVAILABLE:
            try:
                if hasattr(st, 'secrets') and key in st.secrets:
                    value = st.secrets[key]
                    if value:  # Handle empty strings
                        return str(value)
            except Exception as e:
                # Secrets might not be available in local development
                pass
        
        # Fallback to environment variables (for local development)
        value = os.getenv(key, default)
        return value if value else default
    
    def _resolve_database_path(self, db_name: str) -> str:
        """
        Resolve database file path for different environments.
        Handles both local development and Streamlit Cloud deployment.
        """
        if not db_name:
            raise ValueError("Database name is required")
        
        # If it's already an absolute path, use it
        if os.path.isabs(db_name):
            print(f"Using absolute database path: {db_name}")
            return db_name
        
        # For relative paths, try different locations
        current_dir = os.getcwd()
        project_root = Path(__file__).parent.parent
        
        possible_paths = [
            db_name,  # Current directory
            os.path.join(current_dir, db_name),  # Explicit current directory
            os.path.join(project_root, db_name),  # Project root
        ]
        
        # In Streamlit Cloud, try the mount path
        if STREAMLIT_AVAILABLE:
            cloud_paths = [
                f"/mount/src/database-copilot/{db_name}",
                f"/app/{db_name}",
                f"/home/appuser/{db_name}"
            ]
            possible_paths = cloud_paths + possible_paths
        
        print(f"Searching for database file: {db_name}")
        print(f"Current working directory: {current_dir}")
        
        # Find the first path that exists
        for path in possible_paths:
            print(f"Checking path: {path}")
            if os.path.exists(path):
                print(f"✅ Database found at: {path}")
                return path
        
        # If no existing file found, use the most appropriate path for creation
        if STREAMLIT_AVAILABLE:
            # In Streamlit Cloud, prefer the mount path
            final_path = f"/mount/src/database-copilot/{db_name}"
        else:
            # Local development, use project root
            final_path = os.path.join(project_root, db_name)
        
        print(f"⚠️ Database not found. Will use: {final_path}")
        return final_path
        
    def connect(self) -> bool:
        """
        Establish database connection based on configuration.
        Supports both local .env and Streamlit Cloud secrets.
        
        Returns:
            bool: True if connection successful
        """
        try:
            print("🔌 Attempting database connection...")
            print(f"Streamlit available: {STREAMLIT_AVAILABLE}")
            
            # Check for full connection string first
            conn_string = self._get_config_value('DATABASE_CONNECTION_STRING')
            
            if conn_string:
                print(f"Using full connection string")
                self.engine = create_engine(conn_string)
            else:
                # Build connection string from components
                db_type = self._get_config_value('DATABASE_TYPE', 'sqlite').lower()
                host = self._get_config_value('DATABASE_HOST')
                port = self._get_config_value('DATABASE_PORT')
                database = self._get_config_value('DATABASE_NAME')
                username = self._get_config_value('DATABASE_USER')
                password = self._get_config_value('DATABASE_PASSWORD')
                trusted = self._get_config_value('DATABASE_TRUSTED_CONNECTION', '').lower() == 'yes'
                
                print(f"Database type: {db_type}")
                print(f"Database name: {database}")
                print(f"Database host: {host}")
                
                if not database:
                    raise ValueError("DATABASE_NAME is required")
                
                # Handle SQLite path resolution
                if db_type == 'sqlite':
                    resolved_path = self._resolve_database_path(database)
                    database = resolved_path
                    print(f"Resolved SQLite path: {database}")
                elif not host:
                    raise ValueError("DATABASE_HOST is required for non-SQLite databases")
                
                conn_string = self._build_connection_string(
                    db_type, host, port, database, username, password, trusted
                )
                
                print(f"Built connection string for {db_type}")
                self.engine = create_engine(conn_string)
            
            # Test connection
            print("🧪 Testing database connection...")
            self.connection = self.engine.connect()
            self.db_type = self.engine.dialect.name
            
            print(f"✅ Successfully connected to {self.db_type} database")
            return True
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Database connection error: {error_msg}")
            
            # Provide helpful error messages
            if "no such file or directory" in error_msg.lower():
                print("💡 SQLite database file not found. Check the DATABASE_NAME path.")
                print("💡 Make sure the file exists in your GitHub repository.")
            elif "connection refused" in error_msg.lower():
                print("💡 Connection refused. Check if the database server is running.")
            elif "access denied" in error_msg.lower():
                print("💡 Access denied. Check your username and password.")
            elif "driver" in error_msg.lower():
                print("💡 Database driver issue. Check if required drivers are installed.")
            
            return False
    
    def _build_connection_string(self, db_type: str, host: str, port: str, 
                               database: str, username: str, password: str, 
                               trusted: bool = False) -> str:
        """Build database connection string based on type."""
        
        if db_type == 'sqlite':
            return f"sqlite:///{database}"
        
        elif db_type == 'sqlserver':
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
        
        elif db_type == 'duckdb':
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
            print("🔍 Discovering database schema...")
            inspector = inspect(self.engine)
            tables = []
            
            # Get all tables
            table_names = inspector.get_table_names()
            print(f"Found {len(table_names)} tables: {table_names}")
            
            for table_name in table_names:
                try:
                    print(f"Analyzing table: {table_name}")
                    
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
                    
                    # Get row count (with error handling for large tables)
                    try:
                        count_query = text(f"SELECT COUNT(*) FROM {table_name}")
                        result = self.connection.execute(count_query)
                        row_count = result.scalar()
                        print(f"  - {table_name}: {row_count:,} rows, {len(columns)} columns")
                    except Exception as e:
                        print(f"  - {table_name}: Could not count rows ({str(e)})")
                        row_count = None
                    
                    table_info = TableInfo(
                        name=table_name,
                        schema='main',  # Default schema
                        columns=columns,
                        row_count=row_count
                    )
                    
                    tables.append(table_info)
                    
                except Exception as e:
                    print(f"❌ Error processing table {table_name}: {str(e)}")
                    continue
            
            # Cache the schema
            self.schema_cache = {table.name: table for table in tables}
            print(f"✅ Schema discovery complete: {len(tables)} tables cached")
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
                elif self.db_type in ['postgresql', 'mysql', 'sqlite']:
                    sql += ' LIMIT 1000'
                elif self.db_type == 'oracle':
                    sql += ' AND ROWNUM <= 1000'
            
            print(f"🔍 Executing query: {sql[:100]}...")
            result = pd.read_sql(sql, self.connection)
            print(f"✅ Query returned {len(result)} rows")
            return result
            
        except Exception as e:
            print(f"❌ Query execution failed: {str(e)}")
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
    
    def get_debug_info(self) -> Dict[str, str]:
        """Get debugging information about the current configuration."""
        debug_info = {
            'Database Type': self._get_config_value('DATABASE_TYPE', 'Not Set'),
            'Database Name': self._get_config_value('DATABASE_NAME', 'Not Set'),
            'Database Host': self._get_config_value('DATABASE_HOST', 'Not Set'),
            'Connection Status': 'Connected' if self.connection else 'Not Connected',
            'Engine Type': str(self.db_type) if self.db_type else 'None',
            'Tables Cached': str(len(self.schema_cache)),
            'Streamlit Available': str(STREAMLIT_AVAILABLE),
            'Current Working Directory': os.getcwd(),
            'Environment': 'Streamlit Cloud' if STREAMLIT_AVAILABLE and hasattr(st, 'secrets') else 'Local Development'
        }
        
        # Add secrets availability check
        if STREAMLIT_AVAILABLE:
            try:
                has_secrets = hasattr(st, 'secrets') and bool(st.secrets)
                debug_info['Streamlit Secrets'] = 'Available' if has_secrets else 'Not Available'
            except:
                debug_info['Streamlit Secrets'] = 'Error Checking'
        
        return debug_info
    
    def close(self):
        """Close database connection."""
        try:
            if self.connection:
                self.connection.close()
                print("🔌 Database connection closed")
            if self.engine:
                self.engine.dispose()
                print("🔧 Database engine disposed")
        except Exception as e:
            print(f"⚠️ Error closing database connection: {str(e)}")


# Global database manager instance
db_manager = DatabaseManager()