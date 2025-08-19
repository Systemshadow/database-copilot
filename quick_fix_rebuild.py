#!/usr/bin/env python3
"""
Quick fix script to clean up and rebuild the enterprise database.
"""

import os
import shutil
from pathlib import Path

def cleanup_and_rebuild():
    """Clean up failed build and rebuild properly."""
    print("🧹 Cleaning up failed build...")
    
    # Remove any partial database files
    enterprise_dir = Path("enterprise_data")
    if enterprise_dir.exists():
        for db_file in enterprise_dir.glob("*.db"):
            print(f"   🗑️  Removing {db_file}")
            db_file.unlink()
    
    print("✅ Cleanup complete!")
    print("\n🚀 Now run:")
    print("   python build_enterprise_database.py")
    print("   python test_massive_database.py")
    
    # Create a simple .env file for testing
    env_content = """# Database Copilot Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Enterprise Database (will be updated after build)
DATABASE_TYPE=sqlite
DATABASE_NAME=enterprise_data/enterprise_nydec_massive.db
DATABASE_HOST=localhost
DATABASE_PORT=
DATABASE_USER=
DATABASE_PASSWORD=

# Application Settings
COMPANY_NAME=Enterprise Oil & Gas Demo
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("\n📝 Created basic .env file")
    print("💡 Add your OpenAI API key to enable AI features")

if __name__ == "__main__":
    cleanup_and_rebuild()