#!/usr/bin/env python3
"""
Database Cleanup Script
Remove sample/demo data from enterprise database to prevent AI confusion.
"""

import sqlite3
import pandas as pd
from pathlib import Path

def clean_enterprise_database(db_path: str):
    """Remove sample data from enterprise database."""
    print(f"🧹 Cleaning enterprise database: {db_path}")
    
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # List of sample well names to remove
        sample_wells = [
            'Smith 1H', 'Jones 2H', 'Brown 3H', 'Wilson 4H',
            'Smith', 'Jones', 'Brown', 'Wilson'
        ]
        
        # List of sample API numbers to remove
        sample_apis = [
            '31-001-12345', '31-001-12346', '31-001-12347', '31-001-12348'
        ]
        
        print("🔍 Checking for sample data...")
        
        # Check production table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='production'")
        if cursor.fetchone():
            print("  📊 Checking production table...")
            
            # Check for sample wells by name
            for well_name in sample_wells:
                cursor.execute("SELECT COUNT(*) FROM production WHERE Well_Name LIKE ?", (f"%{well_name}%",))
                count = cursor.fetchone()[0]
                if count > 0:
                    print(f"    ⚠️  Found {count} records for {well_name}")
                    cursor.execute("DELETE FROM production WHERE Well_Name LIKE ?", (f"%{well_name}%",))
                    print(f"    ✅ Removed {count} production records for {well_name}")
            
            # Check for sample wells by API
            for api in sample_apis:
                cursor.execute("SELECT COUNT(*) FROM production WHERE API_WellNo = ?", (api,))
                count = cursor.fetchone()[0]
                if count > 0:
                    print(f"    ⚠️  Found {count} records for API {api}")
                    cursor.execute("DELETE FROM production WHERE API_WellNo = ?", (api,))
                    print(f"    ✅ Removed {count} production records for API {api}")
        
        # Check wells table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='wells'")
        if cursor.fetchone():
            print("  🛢️  Checking wells table...")
            
            # Check for sample wells by name
            for well_name in sample_wells:
                cursor.execute("SELECT COUNT(*) FROM wells WHERE Well_Name LIKE ?", (f"%{well_name}%",))
                count = cursor.fetchone()[0]
                if count > 0:
                    print(f"    ⚠️  Found {count} wells named {well_name}")
                    cursor.execute("DELETE FROM wells WHERE Well_Name LIKE ?", (f"%{well_name}%",))
                    print(f"    ✅ Removed {count} wells named {well_name}")
            
            # Check for sample wells by API
            for api in sample_apis:
                cursor.execute("SELECT COUNT(*) FROM wells WHERE API_WellNo = ?", (api,))
                count = cursor.fetchone()[0]
                if count > 0:
                    print(f"    ⚠️  Found {count} wells with API {api}")
                    cursor.execute("DELETE FROM wells WHERE API_WellNo = ?", (api,))
                    print(f"    ✅ Removed {count} wells with API {api}")
        
        # Remove any obvious test operators
        test_operators = ['Test Oil & Gas Company', 'Demo Company', 'Sample Corp']
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='production'")
        if cursor.fetchone():
            for operator in test_operators:
                cursor.execute("SELECT COUNT(*) FROM production WHERE Operator LIKE ?", (f"%{operator}%",))
                count = cursor.fetchone()[0]
                if count > 0:
                    cursor.execute("DELETE FROM production WHERE Operator LIKE ?", (f"%{operator}%",))
                    print(f"    ✅ Removed {count} records for test operator {operator}")
        
        # Commit changes
        conn.commit()
        
        # Get final counts
        cursor.execute("SELECT COUNT(*) FROM production")
        prod_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM wells")
        well_count = cursor.fetchone()[0]
        
        print(f"\n✅ Cleanup complete!")
        print(f"  📊 Production records: {prod_count:,}")
        print(f"  🛢️  Wells: {well_count:,}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Cleanup failed: {str(e)}")
        return False

def verify_cleanup(db_path: str):
    """Verify that sample data has been removed."""
    print(f"\n🔍 Verifying cleanup...")
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Check for any remaining sample data
        sample_checks = [
            "SELECT COUNT(*) FROM wells WHERE Well_Name LIKE '%Smith%'",
            "SELECT COUNT(*) FROM wells WHERE Well_Name LIKE '%Jones%'",
            "SELECT COUNT(*) FROM wells WHERE Well_Name LIKE '%Brown%'",
            "SELECT COUNT(*) FROM wells WHERE Well_Name LIKE '%Wilson%'",
            "SELECT COUNT(*) FROM wells WHERE API_WellNo LIKE '31-001-123%'"
        ]
        
        for check in sample_checks:
            cursor = conn.cursor()
            cursor.execute(check)
            count = cursor.fetchone()[0]
            if count > 0:
                print(f"  ⚠️  Still found sample data: {check} returned {count}")
            else:
                print(f"  ✅ Clean: {check.split('LIKE')[1].strip()}")
        
        # Show some real well examples
        cursor.execute("SELECT Well_Name FROM wells LIMIT 5")
        wells = cursor.fetchall()
        print(f"\n📋 Sample of remaining wells:")
        for well in wells:
            print(f"  - {well[0]}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Verification failed: {str(e)}")

def main():
    """Main cleanup function."""
    print("🧹 Database Copilot - Sample Data Cleanup\n")
    
    # Clean both enterprise databases
    databases = [
        "enterprise_data/enterprise_nydec.db",
        "enterprise_data/enterprise_nydec_massive.db"
    ]
    
    for db_path in databases:
        if Path(db_path).exists():
            print(f"\n{'='*60}")
            clean_enterprise_database(db_path)
            verify_cleanup(db_path)
        else:
            print(f"⚠️  Database not found: {db_path}")
    
    print(f"\n🎉 All cleanup complete!")
    print(f"\nNext: Restart your app to see clean, consistent AI responses.")

if __name__ == "__main__":
    main()