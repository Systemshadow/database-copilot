#!/usr/bin/env python3
"""
Test script for Database Copilot.
Run this to verify your database connection and configuration.
"""

import sys
from pathlib import Path
import os

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from app_utils.database import db_manager
from app_utils.ai_assistant import DatabaseAssistant


def test_environment():
    """Test environment configuration."""
    print("🔧 Testing Environment Configuration...")
    
    required_vars = ['DATABASE_TYPE', 'DATABASE_HOST', 'DATABASE_NAME']
    optional_vars = ['OPENAI_API_KEY', 'DATABASE_USER', 'DATABASE_PASSWORD']
    
    missing_required = []
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"  ✅ {var}: {value}")
        else:
            print(f"  ❌ {var}: Not set")
            missing_required.append(var)
    
    for var in optional_vars:
        value = os.getenv(var)
        if value and value != 'your_openai_api_key_here':
            print(f"  ✅ {var}: Set")
        else:
            print(f"  ⚠️  {var}: Not set (optional)")
    
    if missing_required:
        print(f"\n❌ Missing required variables: {', '.join(missing_required)}")
        return False
    
    print("\n✅ Environment configuration looks good!")
    return True


def test_database_connection():
    """Test database connection."""
    print("\n🔌 Testing Database Connection...")
    
    try:
        success = db_manager.connect()
        if success:
            print("  ✅ Database connection successful!")
            
            # Test schema discovery
            print("\n📊 Testing Schema Discovery...")
            tables = db_manager.discover_schema()
            print(f"  ✅ Found {len(tables)} tables")
            
            # Show first few tables
            for table in tables[:5]:
                print(f"    - {table.name} ({table.row_count:,} rows)" if table.row_count else f"    - {table.name}")
            
            return True
        else:
            print("  ❌ Database connection failed!")
            return False
            
    except Exception as e:
        print(f"  ❌ Database connection error: {str(e)}")
        return False
    finally:
        db_manager.close()


def test_ai_assistant():
    """Test AI assistant functionality."""
    print("\n🤖 Testing AI Assistant...")
    
    try:
        # Check OpenAI configuration
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or api_key == 'your_openai_api_key_here':
            print("  ⚠️  OpenAI API key not configured - AI features will be limited")
            return False
        
        # Connect to database
        if not db_manager.connect():
            print("  ❌ Cannot test AI assistant without database connection")
            return False
        
        # Initialize assistant
        assistant = DatabaseAssistant(db_manager)
        
        # Test simple functionality
        if assistant.openai_client:
            print("  ✅ AI assistant initialized successfully!")
            print("  ✅ OpenAI client configured")
            return True
        else:
            print("  ⚠️  AI assistant initialized but OpenAI client not available")
            return False
            
    except Exception as e:
        print(f"  ❌ AI assistant error: {str(e)}")
        return False
    finally:
        db_manager.close()


def main():
    """Run all tests."""
    print("🧪 Database Copilot - Configuration Test\n")
    
    # Test environment
    env_ok = test_environment()
    
    if not env_ok:
        print("\n❌ Environment configuration issues found. Please check your .env file.")
        return
    
    # Test database
    db_ok = test_database_connection()
    
    # Test AI assistant
    ai_ok = test_ai_assistant()
    
    print("\n" + "="*50)
    print("📋 SUMMARY:")
    print(f"  Environment: {'✅ OK' if env_ok else '❌ Issues'}")
    print(f"  Database:    {'✅ OK' if db_ok else '❌ Issues'}")
    print(f"  AI Assistant: {'✅ OK' if ai_ok else '⚠️ Limited'}")
    
    if env_ok and db_ok:
        print("\n🚀 Ready to run: streamlit run app.py")
    else:
        print("\n🔧 Please fix configuration issues before running the app.")


if __name__ == "__main__":
    main()