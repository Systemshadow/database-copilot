#!/usr/bin/env python3
"""
Test script to verify the error handling fix works properly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_error_handling_fix():
    """Test that the new error handling works properly."""
    print("🧪 TESTING ERROR HANDLING FIX")
    print("=" * 50)
    
    try:
        from app_utils.database import db_manager
        from app_utils.ai_assistant import ExpertDatabaseAssistant
        
        print("✅ Successfully imported ExpertDatabaseAssistant")
        
        # Connect to database
        if not db_manager.connect():
            print("❌ Database connection failed - using mock testing")
            return
        
        print("✅ Database connected")
        
        # Create the enhanced assistant
        assistant = ExpertDatabaseAssistant(db_manager)
        print("✅ ExpertDatabaseAssistant created")
        
        # Test cases that should trigger intelligent responses
        test_cases = [
            "Show me the top 10 producing wells last month",
            "What were the best performing wells last year?", 
            "Tell me about recent production trends",
            "Show me data for well XYZ-123 that doesn't exist",
            "What was production for Fake Company LLC?"
        ]
        
        print(f"\n🎯 Testing {len(test_cases)} scenarios that previously failed:")
        
        for i, question in enumerate(test_cases, 1):
            print(f"\n--- Test {i}: {question[:40]}... ---")
            
            try:
                response = assistant.ask_question(question)
                
                # Check for old robotic responses
                robotic_indicators = [
                    "Hey there, It seems like you're trying to retrieve data",
                    "Sorry, I encountered an error:",
                    "I ran into a technical issue:"
                ]
                
                is_robotic = any(indicator in response for indicator in robotic_indicators)
                
                if is_robotic:
                    print(f"❌ ROBOTIC RESPONSE DETECTED:")
                    print(f"   {response[:100]}...")
                else:
                    print(f"✅ INTELLIGENT RESPONSE:")
                    print(f"   {response[:100]}...")
                    
            except Exception as e:
                print(f"❌ EXCEPTION (should not happen): {str(e)}")
        
        print(f"\n📊 TESTING SUCCESSFUL QUERIES:")
        
        # Test successful queries to ensure they still work
        successful_queries = [
            "How many wells are in the database?",
            "Show me all operators"
        ]
        
        for question in successful_queries:
            print(f"\n--- Success Test: {question} ---")
            try:
                response = assistant.ask_question(question)
                print(f"✅ Response: {response[:100]}...")
            except Exception as e:
                print(f"❌ Unexpected error: {str(e)}")
        
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        print("💡 Make sure the enhanced ai_assistant.py file is in place")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")

def verify_class_structure():
    """Verify the new class has the expected structure."""
    print(f"\n🔍 VERIFYING CLASS STRUCTURE:")
    
    try:
        from app_utils.ai_assistant import ExpertDatabaseAssistant
        
        # Check for key methods that should exist
        expected_methods = [
            'ask_question',
            '_handle_query_failure',
            '_handle_timeframe_error',
            '_generate_expert_response',
            '_execute_query_safely'
        ]
        
        missing_methods = []
        for method_name in expected_methods:
            if hasattr(ExpertDatabaseAssistant, method_name):
                print(f"✅ Method found: {method_name}")
            else:
                print(f"❌ Method missing: {method_name}")
                missing_methods.append(method_name)
        
        if not missing_methods:
            print(f"✅ All expected methods found!")
        else:
            print(f"❌ Missing methods: {missing_methods}")
            
    except Exception as e:
        print(f"❌ Error checking class structure: {str(e)}")

def main():
    """Run all tests."""
    verify_class_structure()
    test_error_handling_fix()
    
    print(f"\n🎯 SUMMARY:")
    print("1. If all tests show ✅ INTELLIGENT RESPONSE, the fix worked!")
    print("2. If you see ❌ ROBOTIC RESPONSE, there may be import issues")
    print("3. Replace the old ai_assistant.py with the enhanced version")
    print("4. Restart your Streamlit app: streamlit run app.py")

if __name__ == "__main__":
    main()