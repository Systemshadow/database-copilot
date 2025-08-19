#!/usr/bin/env python3
"""
Debug script to identify where old robotic error handling is triggered
and why ExpertDatabaseAssistant methods aren't being called.
"""

import os
import sys
from pathlib import Path

def check_current_ai_assistant():
    """Check the current ai_assistant.py file structure."""
    print("🔍 DEBUGGING ERROR HANDLING PIPELINE")
    print("=" * 50)
    
    ai_file = Path("app_utils/ai_assistant.py")
    if not ai_file.exists():
        print("❌ ai_assistant.py not found!")
        return
    
    # Read the file and analyze
    with open(ai_file, 'r') as f:
        content = f.read()
    
    print(f"📄 Current ai_assistant.py:")
    print(f"   📏 Lines: {len(content.splitlines())}")
    
    # Check for key classes and methods
    classes_found = []
    if "class DatabaseAssistant" in content:
        classes_found.append("DatabaseAssistant")
    if "class ExpertDatabaseAssistant" in content:
        classes_found.append("ExpertDatabaseAssistant")
    
    print(f"   🏗️  Classes found: {classes_found}")
    
    # Check for key methods mentioned in status
    key_methods = [
        "_handle_query_failure",
        "ask_question", 
        "generate_response",
        "_generate_not_found_response"
    ]
    
    methods_found = []
    for method in key_methods:
        if f"def {method}" in content:
            methods_found.append(method)
    
    print(f"   🔧 Key methods found: {methods_found}")
    
    # Look for error handling patterns
    print(f"\n🚨 Error Handling Analysis:")
    
    # Check for robotic error messages
    robotic_patterns = [
        "Hey there, It seems like you're trying to retrieve data",
        "Sorry, I encountered an error",
        "I ran into a technical issue"
    ]
    
    for pattern in robotic_patterns:
        if pattern in content:
            print(f"   ❌ Found robotic pattern: '{pattern}'")
        else:
            print(f"   ✅ No robotic pattern: '{pattern}'")
    
    # Check for exception handling
    exception_count = content.count("except Exception as e:")
    print(f"   📊 Exception handlers found: {exception_count}")
    
    return content

def check_app_error_handling():
    """Check how app.py handles errors."""
    print(f"\n📱 APP.PY ERROR HANDLING:")
    
    app_file = Path("app.py")
    if not app_file.exists():
        print("❌ app.py not found!")
        return
    
    with open(app_file, 'r') as f:
        content = f.read()
    
    # Look for error handling in chat interface
    if "except Exception as e:" in content:
        print("   🔍 Found exception handling in app.py")
        
        # Extract the error handling code
        lines = content.splitlines()
        for i, line in enumerate(lines):
            if "except Exception as e:" in line:
                print(f"   📋 Error handling at line {i+1}:")
                # Show surrounding context
                start = max(0, i-2)
                end = min(len(lines), i+5)
                for j in range(start, end):
                    marker = ">>> " if j == i else "    "
                    print(f"   {marker}{j+1}: {lines[j]}")
                print()

def test_failing_query():
    """Test a query that should fail and trace the error path."""
    print(f"\n🧪 TESTING FAILING QUERY:")
    
    try:
        # Add project root to path
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root))
        
        from app_utils.database import db_manager
        from app_utils.ai_assistant import DatabaseAssistant
        
        print("   📚 Imports successful")
        
        # Try to connect
        if db_manager.connect():
            print("   🔌 Database connected")
            
            # Create assistant
            assistant = DatabaseAssistant(db_manager)
            print("   🤖 Assistant created")
            
            # Test a query that should fail with relative timeframe
            test_question = "Show me the top 10 producing wells last month"
            print(f"   ❓ Testing question: '{test_question}'")
            
            try:
                response = assistant.ask_question(test_question)
                print(f"   📝 Response: {response[:100]}...")
                
                # Check if it's the old robotic response
                if "Hey there, It seems like you're trying" in response:
                    print("   ❌ OLD ROBOTIC RESPONSE DETECTED!")
                elif "Sorry, I encountered an error" in response:
                    print("   ❌ BASIC ERROR RESPONSE DETECTED!")
                else:
                    print("   ✅ New intelligent response")
                    
            except Exception as e:
                print(f"   ❌ Exception in ask_question: {str(e)}")
                
        else:
            print("   ❌ Database connection failed")
            
    except ImportError as e:
        print(f"   ❌ Import error: {str(e)}")
    except Exception as e:
        print(f"   ❌ Unexpected error: {str(e)}")

def check_for_multiple_versions():
    """Check if there are multiple versions of files."""
    print(f"\n📂 CHECKING FOR MULTIPLE VERSIONS:")
    
    # Look for backup files or alternative versions
    patterns = [
        "app_utils/ai_assistant*.py",
        "*ai_assistant*.py", 
        "app_*.py",
        "*backup*",
        "*old*"
    ]
    
    all_files = []
    for pattern in patterns:
        files = list(Path(".").glob(pattern))
        all_files.extend(files)
    
    # Remove duplicates and sort
    unique_files = sorted(set(all_files))
    
    for file_path in unique_files:
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"   📄 {file_path} ({size} bytes)")

def main():
    """Run the debug analysis."""
    content = check_current_ai_assistant()
    check_app_error_handling()
    check_for_multiple_versions()
    test_failing_query()
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. Verify ai_assistant.py has ExpertDatabaseAssistant class")
    print("2. Check if _handle_query_failure method exists")
    print("3. Trace where old error responses are coming from")
    print("4. Test relative timeframe queries specifically")

if __name__ == "__main__":
    main()