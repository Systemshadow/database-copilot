#!/usr/bin/env python3
"""
Smart Demo Database Creator for Streamlit Cloud Deployment
Creates a focused, high-quality sample from your massive enterprise database.
"""

import sqlite3
import pandas as pd
from pathlib import Path
import shutil

class DemoDatabaseCreator:
    """Create an intelligent sample database perfect for demos."""
    
    def __init__(self, source_db_path, target_db_path):
        self.source_db_path = source_db_path
        self.target_db_path = target_db_path
        self.source_conn = None
        self.target_conn = None
    
    def create_demo_database(self):
        """Main method to create the demo database."""
        print("🎯 CREATING DEMO DATABASE FOR STREAMLIT CLOUD")
        print("=" * 60)
        
        try:
            # Connect to source database
            self._connect_to_source()
            
            # Analyze source database structure
            source_info = self._analyze_source_database()
            
            # Create target database
            self._create_target_database()
            
            # Sample and copy data intelligently
            self._sample_and_copy_data(source_info)
            
            # Verify and report
            self._verify_demo_database()
            
            print("\n🎉 SUCCESS! Demo database created!")
            print(f"📁 Location: {self.target_db_path}")
            print("🚀 Ready for Streamlit Cloud deployment!")
            
        except Exception as e:
            print(f"❌ Error creating demo database: {str(e)}")
            raise
        finally:
            self._close_connections()
    
    def _connect_to_source(self):
        """Connect to the source massive database."""
        if not Path(self.source_db_path).exists():
            raise FileNotFoundError(f"Source database not found: {self.source_db_path}")
        
        self.source_conn = sqlite3.connect(self.source_db_path)
        print(f"✅ Connected to source: {self.source_db_path}")
        
        # Get source database size
        size_mb = Path(self.source_db_path).stat().st_size / (1024 * 1024)
        print(f"📊 Source size: {size_mb:.1f} MB")
    
    def _analyze_source_database(self):
        """Analyze the source database to understand its structure."""
        print(f"\n🔍 Analyzing source database structure...")
        
        # Get table names
        tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
        tables = pd.read_sql(tables_query, self.source_conn)
        
        print(f"📋 Tables found: {', '.join(tables['name'].tolist())}")
        
        # Assume main table is 'production' or first table
        main_table = 'production' if 'production' in tables['name'].values else tables['name'].iloc[0]
        print(f"🎯 Main table: {main_table}")
        
        # Get table info
        schema_query = f"PRAGMA table_info({main_table})"
        columns = pd.read_sql(schema_query, self.source_conn)
        print(f"📊 Columns: {len(columns)} columns")
        
        # Get total records
        count_query = f"SELECT COUNT(*) as total FROM {main_table}"
        total_records = pd.read_sql(count_query, self.source_conn)['total'].iloc[0]
        print(f"📈 Total records: {total_records:,}")
        
        # Get year range
        year_columns = ['year', 'Year', 'production_date', 'Date']
        year_column = None
        for col in year_columns:
            if col in columns['name'].values:
                year_column = col
                break
        
        if year_column:
            year_query = f"SELECT MIN({year_column}) as min_year, MAX({year_column}) as max_year FROM {main_table}"
            year_range = pd.read_sql(year_query, self.source_conn)
            print(f"📅 Year range: {year_range['min_year'].iloc[0]} to {year_range['max_year'].iloc[0]}")
        
        return {
            'main_table': main_table,
            'total_records': total_records,
            'year_column': year_column,
            'columns': columns,
            'all_tables': tables['name'].tolist()
        }
    
    def _create_target_database(self):
        """Create the target demo database with same structure."""
        print(f"\n🏗️  Creating target database...")
        
        # Remove existing target if it exists
        if Path(self.target_db_path).exists():
            Path(self.target_db_path).unlink()
            print(f"🗑️  Removed existing {self.target_db_path}")
        
        # Ensure target directory exists
        Path(self.target_db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create target database connection
        self.target_conn = sqlite3.connect(self.target_db_path)
        print(f"✅ Created target database: {self.target_db_path}")
    
    def _sample_and_copy_data(self, source_info):
        """Intelligently sample and copy data to target database."""
        print(f"\n📊 Sampling data for demo (2020-2024, top wells)...")
        
        main_table = source_info['main_table']
        year_column = source_info['year_column']
        
        # First, copy the table structure
        self._copy_table_structure(main_table)
        
        # Strategy: Get top wells by production, then get their 2020-2024 data
        sampling_query = self._build_smart_sampling_query(main_table, year_column)
        
        print(f"🔄 Executing sampling query...")
        sampled_data = pd.read_sql(sampling_query, self.source_conn)
        
        print(f"📈 Sampled records: {len(sampled_data):,}")
        
        # Copy sampled data to target
        sampled_data.to_sql(main_table, self.target_conn, if_exists='append', index=False)
        
        # Copy other tables if they exist and are small
        for table_name in source_info['all_tables']:
            if table_name != main_table:
                self._copy_small_table_if_relevant(table_name)
        
        print(f"✅ Data sampling complete!")
    
    def _copy_table_structure(self, table_name):
        """Copy table structure from source to target."""
        # Get CREATE TABLE statement
        create_query = f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        create_sql = pd.read_sql(create_query, self.source_conn)
        
        if not create_sql.empty:
            create_statement = create_sql['sql'].iloc[0]
            self.target_conn.execute(create_statement)
            print(f"📋 Copied table structure for {table_name}")
    
    def _build_smart_sampling_query(self, main_table, year_column):
        """Build an intelligent sampling query that gets the best demo data."""
        
        if not year_column:
            # If no year column found, just sample randomly
            return f"""
            SELECT * FROM {main_table} 
            ORDER BY RANDOM() 
            LIMIT 50000
            """
        
        # Smart sampling: Get top wells by total production, then get their recent data
        query = f"""
        WITH top_wells AS (
            -- Get top 1000 wells by total production
            SELECT well_name, operator, county, 
                   SUM(COALESCE(oil_prod, 0) + COALESCE(gas_prod, 0)) as total_production
            FROM {main_table}
            WHERE {year_column} BETWEEN 2015 AND 2024  -- Wider range to identify top wells
            GROUP BY well_name, operator, county
            ORDER BY total_production DESC
            LIMIT 1000
        ),
        recent_data AS (
            -- Get 2020-2024 data for these top wells
            SELECT p.*
            FROM {main_table} p
            INNER JOIN top_wells tw ON p.well_name = tw.well_name
            WHERE p.{year_column} BETWEEN 2020 AND 2024
        ),
        representative_sample AS (
            -- Add some random wells to ensure operator/county diversity
            SELECT * FROM {main_table}
            WHERE {year_column} BETWEEN 2020 AND 2024
            ORDER BY RANDOM()
            LIMIT 10000
        )
        
        -- Combine top wells recent data + representative sample, remove duplicates
        SELECT DISTINCT * FROM (
            SELECT * FROM recent_data
            UNION
            SELECT * FROM representative_sample
        )
        ORDER BY {year_column} DESC, well_name
        LIMIT 75000  -- Target ~75k records for good demo size
        """
        
        return query
    
    def _copy_small_table_if_relevant(self, table_name):
        """Copy small supporting tables if they exist and are relevant."""
        try:
            # Check table size
            count_query = f"SELECT COUNT(*) as count FROM {table_name}"
            count = pd.read_sql(count_query, self.source_conn)['count'].iloc[0]
            
            # Only copy small tables (< 10k records)
            if count < 10000:
                # Copy structure
                self._copy_table_structure(table_name)
                
                # Copy data
                data = pd.read_sql(f"SELECT * FROM {table_name}", self.source_conn)
                data.to_sql(table_name, self.target_conn, if_exists='append', index=False)
                print(f"📋 Copied supporting table: {table_name} ({count:,} records)")
            else:
                print(f"⏭️  Skipped large table: {table_name} ({count:,} records)")
                
        except Exception as e:
            print(f"⚠️  Could not copy table {table_name}: {str(e)}")
    
    def _verify_demo_database(self):
        """Verify the demo database was created successfully."""
        print(f"\n🔍 Verifying demo database...")
        
        # Check file size
        size_mb = Path(self.target_db_path).stat().st_size / (1024 * 1024)
        print(f"📁 Demo database size: {size_mb:.1f} MB")
        
        if size_mb > 50:
            print(f"⚠️  Warning: Database is {size_mb:.1f}MB - may be too large for Streamlit Cloud")
            print("💡 Consider reducing the LIMIT in the sampling query")
        elif size_mb < 5:
            print(f"⚠️  Warning: Database is only {size_mb:.1f}MB - might be too small for good demo")
        else:
            print(f"✅ Perfect size for Streamlit Cloud deployment!")
        
        # Check record counts
        tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
        tables = pd.read_sql(tables_query, self.target_conn)
        
        for table_name in tables['name']:
            count_query = f"SELECT COUNT(*) as count FROM {table_name}"
            count = pd.read_sql(count_query, self.target_conn)['count'].iloc[0]
            print(f"📊 {table_name}: {count:,} records")
    
    def _close_connections(self):
        """Close database connections."""
        if self.source_conn:
            self.source_conn.close()
        if self.target_conn:
            self.target_conn.close()


def create_demo_database():
    """Main function to create the demo database."""
    
    # Configuration
    source_db = "enterprise_data/enterprise_nydec_massive.db"
    target_db = "demo_data/nydec_demo.db"
    
    print("🎯 DATABASE COPILOT DEMO CREATOR")
    print("=" * 50)
    print(f"📂 Source: {source_db}")
    print(f"🎯 Target: {target_db}")
    print(f"📊 Strategy: 2020-2024 data, top wells, <25MB")
    print()
    
    # Check if source exists
    if not Path(source_db).exists():
        print(f"❌ Source database not found: {source_db}")
        print("💡 Available databases:")
        for db_file in Path("enterprise_data").glob("*.db"):
            size_mb = db_file.stat().st_size / (1024 * 1024)
            print(f"   📁 {db_file} ({size_mb:.1f} MB)")
        return False
    
    try:
        # Create the demo database
        creator = DemoDatabaseCreator(source_db, target_db)
        creator.create_demo_database()
        
        print(f"\n🎉 DEMO DATABASE READY!")
        print(f"📁 File: {target_db}")
        print(f"🚀 Next steps:")
        print(f"   1. Update .env to use: DATABASE_NAME={target_db}")
        print(f"   2. Test locally: streamlit run app.py")
        print(f"   3. Deploy to Streamlit Cloud")
        print(f"   4. Send link to your tester!")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create demo database: {str(e)}")
        return False


if __name__ == "__main__":
    success = create_demo_database()