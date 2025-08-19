#!/usr/bin/env python3
"""
Database Inspector - Check what's actually in your database
"""

import sqlite3
import pandas as pd
from pathlib import Path
import os

def inspect_database():
    """Inspect the actual database contents."""
    print("🔍 Database Inspector")
    print("=" * 50)
    
    # Check what database files exist
    possible_dbs = [
        "test_data/oilgas_test.db",
        "enterprise_data/enterprise_oilgas.db", 
        "enterprise_data/enterprise_nydec.db",
        "enterprise_data/enterprise_nydec_massive.db"
    ]
    
    found_dbs = []
    for db_path in possible_dbs:
        if Path(db_path).exists():
            size_mb = Path(db_path).stat().st_size / (1024 * 1024)
            found_dbs.append((db_path, size_mb))
            print(f"📁 Found: {db_path} ({size_mb:.1f} MB)")
    
    if not found_dbs:
        print("❌ No database files found!")
        return
    
    # Inspect each database
    for db_path, size_mb in found_dbs:
        print(f"\n🔍 Inspecting: {db_path}")
        print("-" * 40)
        
        try:
            conn = sqlite3.connect(db_path)
            
            # Get all tables
            tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
            tables_df = pd.read_sql(tables_query, conn)
            
            if tables_df.empty:
                print("   ❌ No tables found")
                continue
            
            print(f"   📊 Tables found: {len(tables_df)}")
            
            for table_name in tables_df['name']:
                try:
                    # Get row count
                    count_query = f"SELECT COUNT(*) as count FROM {table_name}"
                    count = pd.read_sql(count_query, conn)['count'][0]
                    
                    # Get column info
                    pragma_query = f"PRAGMA table_info({table_name})"
                    columns_df = pd.read_sql(pragma_query, conn)
                    
                    print(f"   📋 {table_name}:")
                    print(f"      • Rows: {count:,}")
                    print(f"      • Columns: {len(columns_df)}")
                    
                    # Show first few column names
                    col_names = columns_df['name'].tolist()[:5]
                    print(f"      • Sample columns: {', '.join(col_names)}")
                    
                    # If it's production table, show date range
                    if table_name == 'production' and count > 0:
                        try:
                            date_query = f"SELECT MIN(year) as min_year, MAX(year) as max_year FROM {table_name}"
                            date_info = pd.read_sql(date_query, conn)
                            min_year = date_info['min_year'][0]
                            max_year = date_info['max_year'][0]
                            if min_year and max_year:
                                print(f"      • Year range: {min_year}-{max_year}")
                        except:
                            pass
                    
                except Exception as e:
                    print(f"   ❌ Error inspecting {table_name}: {str(e)}")
            
            # Check for views
            views_query = "SELECT name FROM sqlite_master WHERE type='view'"
            views_df = pd.read_sql(views_query, conn)
            if not views_df.empty:
                print(f"   📈 Views: {len(views_df)} ({', '.join(views_df['name'])})")
            
            conn.close()
            
        except Exception as e:
            print(f"   ❌ Error accessing database: {str(e)}")
    
    # Check your .env file
    print(f"\n📝 Current .env configuration:")
    if Path('.env').exists():
        with open('.env', 'r') as f:
            for line in f:
                if line.startswith('DATABASE_'):
                    print(f"   {line.strip()}")
    else:
        print("   ❌ No .env file found")
    
    print("\n💡 Recommendations:")
    if found_dbs:
        largest_db = max(found_dbs, key=lambda x: x[1])
        print(f"   • Use the largest database: {largest_db[0]} ({largest_db[1]:.1f} MB)")
        print(f"   • Update your .env file:")
        print(f"     DATABASE_TYPE=sqlite")
        print(f"     DATABASE_NAME={largest_db[0]}")
    
    if any(size > 10 for _, size in found_dbs):
        print("   • You have a substantial database - the Streamlit schema discovery might be timing out")
        print("   • Try refreshing the connection in the app")

if __name__ == "__main__":
    inspect_database()