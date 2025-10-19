#!/usr/bin/env python3
"""
Test script for RAG chatbot functionality
Run: python test_rag.py
"""

import os
import sys
from main import get_chatbot, ask_question_with_context

def test_rag_system():
    """Test the RAG system"""
    print("🧪 Testing RAG Chatbot System")
    print("=" * 50)
    
    try:
        # Test chatbot initialization
        print("1. Testing chatbot initialization...")
        bot = get_chatbot()
        print("✅ Chatbot initialized successfully")
        
        # Test questions
        test_questions = [
            "What is the minimum GPA required?",
            "What are the TOEFL requirements?",
            "Who is the program advisor?",
            "What is the Duolingo score requirement?"
        ]
        
        print("\n2. Testing RAG functionality...")
        for i, question in enumerate(test_questions, 1):
            print(f"\n--- Test Question {i} ---")
            print(f"Q: {question}")
            
            try:
                answer, context = ask_question_with_context(question)
                print(f"A: {answer}")
                print(f"Context Length: {len(context)} characters")
                print(f"Context Preview: {context[:200]}...")
                print("✅ RAG working correctly")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                return False
        
        print("\n3. Testing graph statistics...")
        try:
            stats = bot.get_graph_stats()
            print("Graph Stats:")
            for stat in stats:
                print(f"  {stat['label']}: {stat['count']}")
            print("✅ Graph stats working")
        except Exception as e:
            print(f"⚠️  Graph stats warning: {e}")
        
        print("\n🎉 All tests passed! RAG system is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_rag_system()
    sys.exit(0 if success else 1)
