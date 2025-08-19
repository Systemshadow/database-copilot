#!/usr/bin/env python3
"""
Real Database Inspector & Example Generator
Discovers what's actually in your database and creates working example questions.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import sqlite3
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def inspect_current_database():
    """Inspect the database currently configured in .env"""
    print("🔍 INSPECTING YOUR ACTUAL DATABASE")
    print("=" * 50)
    
    # Get database path from .env
    db_type = os.getenv('DATABASE_TYPE', 'sqlite')
    db_name = os.getenv('DATABASE_NAME')
    
    print(f"📁 Database Type: {db_type}")
    print(f"📁 Database Path: {db_name}")
    
    if not db_name or not Path(db_name).exists():
        print(f"❌ Database not found: {db_name}")
        print("\n💡 Available databases:")
        for folder in ['enterprise_data', 'demo_data', 'test_data']:
            if Path(folder).exists():
                for db_file in Path(folder).glob("*.db"):
                    size_mb = db_file.stat().st_size / (1024 * 1024)
                    print(f"   📄 {db_file} ({size_mb:.1f} MB)")
        return None
    
    return db_name

def analyze_real_data(db_path):
    """Analyze the real data in the database."""
    print(f"\n🔍 Analyzing real data in {db_path}...")
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Get tables
        tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
        tables = pd.read_sql(tables_query, conn)
        print(f"📊 Tables: {', '.join(tables['name'].tolist())}")
        
        # Focus on main production table
        main_table = 'production' if 'production' in tables['name'].values else tables['name'].iloc[0]
        print(f"🎯 Analyzing table: {main_table}")
        
        # Get column information
        columns_query = f"PRAGMA table_info({main_table})"
        columns = pd.read_sql(columns_query, conn)
        print(f"📋 Columns: {', '.join(columns['name'].tolist())}")
        
        # Get total record count
        count_query = f"SELECT COUNT(*) as total FROM {main_table}"
        total = pd.read_sql(count_query, conn)['total'].iloc[0]
        print(f"📈 Total records: {total:,}")
        
        # Sample real data
        sample_query = f"SELECT * FROM {main_table} LIMIT 5"
        sample_data = pd.read_sql(sample_query, conn)
        
        print(f"\n📊 Sample of real data:")
        print(sample_data.head())
        
        # Extract real values for examples
        real_data = extract_real_values(conn, main_table, columns)
        
        conn.close()
        return real_data
        
    except Exception as e:
        print(f"❌ Error analyzing database: {str(e)}")
        return None

def extract_real_values(conn, table_name, columns):
    """Extract real well names, operators, counties from the database."""
    print(f"\n🎯 Extracting real values for examples...")
    
    real_data = {}
    
    # Common column name variations
    column_mappings = {
        'well_names': ['Well_Name', 'well_name', 'WellName', 'API_Well_Number', 'well_id'],
        'operators': ['Operator', 'operator', 'Operator_Name', 'operator_name', 'Company', 'company'],
        'counties': ['County', 'county', 'County_Name', 'county_name'],
        'years': ['Year', 'year', 'Production_Year', 'production_year']
    }
    
    available_columns = columns['name'].tolist()
    
    for category, possible_columns in column_mappings.items():
        found_column = None
        for col in possible_columns:
            if col in available_columns:
                found_column = col
                break
        
        if found_column:
            try:
                # Get distinct values, limiting to reasonable number
                query = f"""
                SELECT DISTINCT {found_column} as value, COUNT(*) as count 
                FROM {table_name} 
                WHERE {found_column} IS NOT NULL 
                  AND {found_column} != '' 
                GROUP BY {found_column} 
                ORDER BY count DESC 
                LIMIT 10
                """
                values = pd.read_sql(query, conn)
                real_data[category] = {
                    'column': found_column,
                    'values': values['value'].tolist()
                }
                print(f"✅ Found {category}: {found_column} ({len(values)} unique values)")
                print(f"   Top values: {', '.join(map(str, values['value'][:3]))}")
                
            except Exception as e:
                print(f"❌ Error getting {category} from {found_column}: {str(e)}")
        else:
            print(f"⚠️  No column found for {category}")
    
    return real_data

def generate_real_examples(real_data):
    """Generate example questions using real data from the database."""
    print(f"\n🎯 Generating real example questions...")
    
    examples = []
    
    # Generic questions that always work
    examples.extend([
        "How many wells are in the database?",
        "Show me the top 10 producing wells in 2023",
        "What was the total oil production in 2023?"
    ])
    
    # Well-specific questions
    if 'well_names' in real_data and real_data['well_names']['values']:
        well = real_data['well_names']['values'][0]
        examples.append(f"What was the production for well {well} in 2023?")
        
        if len(real_data['well_names']['values']) > 1:
            well2 = real_data['well_names']['values'][1]
            examples.append(f"Compare production between {well} and {well2}")
    
    # Operator questions
    if 'operators' in real_data and real_data['operators']['values']:
        operator = real_data['operators']['values'][0]
        examples.append(f"What's the total production for {operator} in 2023?")
        examples.append(f"How many wells does {operator} operate?")
    
    # County questions
    if 'counties' in real_data and real_data['counties']['values']:
        county = real_data['counties']['values'][0]
        examples.append(f"Show me all wells in {county} County")
        
        if len(real_data['counties']['values']) > 1:
            county2 = real_data['counties']['values'][1]
            examples.append(f"Compare production between {county} and {county2} counties")
    
    # Limit to 8 examples for clean UI
    return examples[:8]

def generate_app_code(examples):
    """Generate the exact code to replace in app.py"""
    print(f"\n📝 EXACT CODE FOR YOUR APP.PY:")
    print("=" * 50)
    print("Replace the 'examples' list in your chat_interface() function with:")
    print()
    
    print("        examples = [")
    for example in examples:
        print(f'            "{example}",')
    print("        ]")
    
    print(f"\n📋 Copy the above code and replace the examples list in app.py")

def main():
    """Main function to inspect database and generate real examples."""
    
    # Step 1: Find the current database
    db_path = inspect_current_database()
    if not db_path:
        return
    
    # Step 2: Analyze real data
    real_data = analyze_real_data(db_path)
    if not real_data:
        return
    
    # Step 3: Generate real examples
    examples = generate_real_examples(real_data)
    
    # Step 4: Generate code for app.py
    generate_app_code(examples)
    
    print(f"\n🎉 SUCCESS!")
    print(f"📋 Your database has been analyzed")
    print(f"🎯 Real example questions generated")
    print(f"📝 Copy the code above into your app.py")
    print(f"🚀 Test locally before deploying to cloud")

if __name__ == "__main__":
    main()