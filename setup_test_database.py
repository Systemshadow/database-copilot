#!/usr/bin/env python3
"""
Setup test database with NY DEC oil & gas data for Database Copilot testing.
Downloads real production data and creates a local SQLite database.
"""

import pandas as pd
import sqlite3
import requests
import zipfile
import os
from pathlib import Path
import sys

def download_ny_dec_data():
    """Download NY DEC oil and gas production data."""
    print("📥 Downloading NY DEC Oil & Gas Data...")
    
    # Create data directory
    data_dir = Path("test_data")
    data_dir.mkdir(exist_ok=True)
    
    # NY DEC data URLs - Using data.ny.gov API endpoints for reliability
    datasets = {
        "well_production": "https://data.ny.gov/api/views/szuv-pyxi/rows.csv?accessType=DOWNLOAD",
        "wells": "https://data.ny.gov/api/views/jklw-kmbc/rows.csv?accessType=DOWNLOAD",
        "permits": "https://data.ny.gov/api/views/ksv8-5h93/rows.csv?accessType=DOWNLOAD"
    }
    
    downloaded_files = {}
    
    for name, url in datasets.items():
        try:
            print(f"  📄 Downloading {name}...")
            
            # Try to download the file
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            file_path = data_dir / f"{name}.csv"
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            downloaded_files[name] = file_path
            print(f"    ✅ Downloaded: {file_path}")
            
        except Exception as e:
            print(f"    ⚠️  Could not download {name}: {str(e)}")
            print(f"    💡 You can manually download from: {url}")
    
    return downloaded_files

def create_sample_data():
    """Create sample oil & gas data if downloads fail."""
    print("🔧 Creating sample test data...")
    
    data_dir = Path("test_data")
    data_dir.mkdir(exist_ok=True)
    
    # Sample well production data
    sample_production = pd.DataFrame({
        'API_WellNo': ['31-001-12345', '31-001-12346', '31-001-12347', '31-001-12348'],
        'Well_Name': ['Smith 1H', 'Jones 2H', 'Brown 3H', 'Wilson 4H'],
        'Operator': ['Chesapeake Energy', 'Range Resources', 'Chesapeake Energy', 'Cabot Oil'],
        'County': ['Broome', 'Tioga', 'Chemung', 'Broome'],
        'Year': [2023, 2023, 2023, 2023],
        'Month': [1, 1, 1, 1],
        'MonthProd': [1, 1, 1, 1],
        'OilProd': [1247.5, 892.3, 1156.8, 723.4],
        'GasProd': [45623.2, 52341.7, 48952.1, 39876.5],
        'WaterProd': [234.6, 156.8, 287.9, 145.2],
        'Production_Date': ['2023-01-01', '2023-01-01', '2023-01-01', '2023-01-01']
    })
    
    # Add more months of data
    all_data = []
    for month in range(1, 13):  # 12 months
        monthly_data = sample_production.copy()
        monthly_data['Month'] = month
        monthly_data['MonthProd'] = month
        monthly_data['Production_Date'] = f'2023-{month:02d}-01'
        
        # Vary production slightly by month
        monthly_data['OilProd'] = monthly_data['OilProd'] * (0.8 + 0.4 * (month / 12))
        monthly_data['GasProd'] = monthly_data['GasProd'] * (0.9 + 0.2 * (month / 12))
        monthly_data['WaterProd'] = monthly_data['WaterProd'] * (1.1 - 0.2 * (month / 12))
        
        all_data.append(monthly_data)
    
    # Add 2022 data
    for month in range(1, 13):
        monthly_data = sample_production.copy()
        monthly_data['Year'] = 2022
        monthly_data['Month'] = month
        monthly_data['MonthProd'] = month
        monthly_data['Production_Date'] = f'2022-{month:02d}-01'
        
        # Different production levels for 2022
        monthly_data['OilProd'] = monthly_data['OilProd'] * (0.7 + 0.3 * (month / 12))
        monthly_data['GasProd'] = monthly_data['GasProd'] * (0.85 + 0.15 * (month / 12))
        monthly_data['WaterProd'] = monthly_data['WaterProd'] * (1.2 - 0.3 * (month / 12))
        
        all_data.append(monthly_data)
    
    production_df = pd.concat(all_data, ignore_index=True)
    
    # Sample well information
    wells_df = pd.DataFrame({
        'API_WellNo': ['31-001-12345', '31-001-12346', '31-001-12347', '31-001-12348'],
        'Well_Name': ['Smith 1H', 'Jones 2H', 'Brown 3H', 'Wilson 4H'],
        'Operator_Name': ['Chesapeake Energy', 'Range Resources', 'Chesapeake Energy', 'Cabot Oil'],
        'County': ['Broome', 'Tioga', 'Chemung', 'Broome'],
        'Well_Type': ['Horizontal', 'Horizontal', 'Horizontal', 'Horizontal'],
        'Status': ['Producing', 'Producing', 'Producing', 'Producing'],
        'Spud_Date': ['2022-03-15', '2022-04-20', '2022-05-10', '2022-06-05'],
        'Total_Depth': [8500, 9200, 8800, 8750],
        'Field_Name': ['Marcellus', 'Marcellus', 'Marcellus', 'Marcellus']
    })
    
    # Save sample data
    production_path = data_dir / "well_production.csv"
    wells_path = data_dir / "wells.csv"
    
    production_df.to_csv(production_path, index=False)
    wells_df.to_csv(wells_path, index=False)
    
    print(f"  ✅ Created sample production data: {production_path}")
    print(f"  ✅ Created sample wells data: {wells_path}")
    
    return {
        'well_production': production_path,
        'wells': wells_path
    }

def create_sqlite_database(data_files):
    """Create SQLite database from CSV files."""
    print("🗄️  Creating SQLite test database...")
    
    db_path = Path("test_data") / "oilgas_test.db"
    
    # Remove existing database
    if db_path.exists():
        os.remove(db_path)
    
    conn = sqlite3.connect(str(db_path))
    
    for table_name, file_path in data_files.items():
        try:
            if file_path and Path(file_path).exists():
                print(f"  📊 Loading {table_name} from {file_path}...")
                
                df = pd.read_csv(file_path)
                
                # Clean column names
                df.columns = df.columns.str.strip()
                
                # Load into SQLite
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                
                row_count = len(df)
                print(f"    ✅ Loaded {row_count:,} rows into {table_name}")
            
        except Exception as e:
            print(f"    ❌ Error loading {table_name}: {str(e)}")
    
    conn.close()
    
    print(f"\n✅ Test database created: {db_path}")
    return db_path

def create_env_config(db_path):
    """Create .env configuration for the test database."""
    print("⚙️  Creating test database configuration...")
    
    env_content = f"""# Test Database Configuration for Database Copilot
OPENAI_API_KEY=your_openai_api_key_here

# SQLite Test Database
DATABASE_TYPE=sqlite
DATABASE_NAME={db_path}
DATABASE_HOST=localhost
DATABASE_PORT=
DATABASE_USER=
DATABASE_PASSWORD=

# Application Settings
COMPANY_NAME=Test Oil & Gas Company
APP_TITLE=Database Copilot - Test Mode
"""
    
    env_path = Path(".env.test")
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    print(f"  ✅ Created test configuration: {env_path}")
    print(f"  💡 Copy this to .env to use the test database")
    
    return env_path

def show_test_queries():
    """Show example queries to test with the database."""
    print("\n🧪 Test Queries to Try:")
    
    queries = [
        "What was the oil production for well Smith 1H in January 2023?",
        "Show me the top producing wells in 2023",
        "What's the total gas production for Chesapeake Energy?",
        "Which county had the highest production last year?",
        "How did production change for the Smith 1H well over 2023?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"  {i}. \"{query}\"")

def main():
    """Main setup function."""
    print("🛢️  Database Copilot - Test Database Setup\n")
    
    try:
        # Try to download real data first
        data_files = download_ny_dec_data()
        
        # If downloads failed, create sample data
        if not data_files:
            print("\n📝 Download failed, creating sample data instead...")
            data_files = create_sample_data()
        
        # Create SQLite database
        db_path = create_sqlite_database(data_files)
        
        # Create configuration
        env_path = create_env_config(db_path)
        
        # Show test queries
        show_test_queries()
        
        print(f"\n🎉 Test database setup complete!")
        print(f"\nNext steps:")
        print(f"1. Copy {env_path} to .env (or update your existing .env)")
        print(f"2. Add your OpenAI API key to .env")
        print(f"3. Run: python test_connection.py")
        print(f"4. Run: streamlit run app.py")
        
    except Exception as e:
        print(f"\n❌ Setup failed: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection for downloads")
        print("2. Ensure you have write permissions in this directory")
        print("3. Check that pandas and requests are installed")

if __name__ == "__main__":
    main()