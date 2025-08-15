#!/usr/bin/env python3
"""
Setup verification script for Database Copilot.
Checks that all required files exist in the correct structure.
"""

from pathlib import Path
import sys

def check_file_structure():
    """Check that all required files exist."""
    print("🔍 Checking file structure...")
    
    project_root = Path(__file__).parent
    
    required_files = [
        "app.py",
        "requirements.txt", 
        ".env",
        "app_utils/__init__.py",
        "app_utils/database.py",
        "app_utils/ai_assistant.py",
        ".streamlit/config.toml"
    ]
    
    missing_files = []
    existing_files = []
    
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            existing_files.append(file_path)
            print(f"  ✅ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"  ❌ {file_path}")
    
    print(f"\n📊 Summary: {len(existing_files)}/{len(required_files)} files found")
    
    if missing_files:
        print(f"\n❌ Missing files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    else:
        print("\n✅ All required files present!")
        return True

def check_imports():
    """Test that imports work correctly."""
    print("\n🐍 Testing imports...")
    
    try:
        # Add current directory to path
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root))
        
        from app_utils.database import db_manager
        print("  ✅ database.py imported successfully")
        
        from app_utils.ai_assistant import DatabaseAssistant  
        print("  ✅ ai_assistant.py imported successfully")
        
        print("\n✅ All imports working!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure all files are saved in the correct locations")
        print("2. Check that app_utils/__init__.py exists and is not empty")
        print("3. Verify no syntax errors in the Python files")
        return False

def create_directory_structure():
    """Create missing directories."""
    print("\n📁 Creating directory structure...")
    
    project_root = Path(__file__).parent
    
    directories = [
        "app_utils",
        ".streamlit"
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✅ Created directory: {directory}")
        else:
            print(f"  ✅ Directory exists: {directory}")

def main():
    """Main setup check."""
    print("🛠️  Database Copilot - Setup Check\n")
    
    # Create directories if needed
    create_directory_structure()
    
    # Check file structure
    files_ok = check_file_structure()
    
    if not files_ok:
        print("\n🔧 Please ensure all files are created and saved in the correct locations.")
        return
    
    # Test imports
    imports_ok = check_imports()
    
    print("\n" + "="*50)
    if files_ok and imports_ok:
        print("🎉 Setup check passed! Ready to run:")
        print("   streamlit run app.py")
    else:
        print("❌ Setup issues found. Please fix the errors above.")

if __name__ == "__main__":
    main()