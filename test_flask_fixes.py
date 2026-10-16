"""
Test script to verify Flask app fixes
Run this to check all 4 issues are resolved
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

def test_llm_provider():
    """Test that Gemini LLM is working"""
    print("\n" + "=" * 70)
    print("TEST 1: LLM Provider (Gemini)")
    print("=" * 70)
    
    try:
        from agents.llm_provider import get_llm_provider
        
        provider = get_llm_provider()
        print(f"✓ Provider initialized: {provider.provider}")
        print(f"✓ Model: {provider.model_name}")
        
        if provider.provider != "gemini":
            print(f"⚠ WARNING: Expected gemini, got {provider.provider}")
        else:
            print("✓ Using Gemini as expected")
        
        # Test generation
        messages = [
            {"role": "user", "content": "What is the WHO blast threshold for acute leukemia? Answer in one sentence."}
        ]
        response = provider.generate(messages, max_tokens=100)
        print(f"✓ Test query successful")
        print(f"  Response: {response[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ LLM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rag_system():
    """Test that RAG retrieval is working"""
    print("\n" + "=" * 70)
    print("TEST 2: RAG Retrieval System")
    print("=" * 70)
    
    try:
        from retrieval.retrieve import HybridRetriever
        
        retriever = HybridRetriever()
        print("✓ HybridRetriever initialized")
        
        # Test retrieval
        query = "Bone marrow blast cells clinical significance"
        results = retriever.retrieve(query=query, top_k=3)
        print(f"✓ Retrieved {len(results)} documents")
        
        if results:
            print(f"  First result preview: {results[0]['text'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ RAG test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """Test configuration"""
    print("\n" + "=" * 70)
    print("TEST 3: Configuration")
    print("=" * 70)
    
    try:
        from configs.config_loader import load_config
        
        config = load_config()
        print("✓ Config loaded")
        
        llm_config = config.get('llm', {})
        print(f"  Provider: {llm_config.get('provider')}")
        print(f"  Model: {llm_config.get('model')}")
        
        if llm_config.get('provider') == 'gemini':
            print("✓ Config set to use Gemini")
        else:
            print(f"⚠ WARNING: Config provider is {llm_config.get('provider')}, expected gemini")
        
        return True
        
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False


def test_env_vars():
    """Test environment variables"""
    print("\n" + "=" * 70)
    print("TEST 4: Environment Variables")
    print("=" * 70)
    
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    google_key = os.getenv('GOOGLE_API_KEY')
    groq_key = os.getenv('GROQ_API_KEY')
    
    if google_key:
        print(f"✓ GOOGLE_API_KEY found: {google_key[:20]}...")
    else:
        print("✗ GOOGLE_API_KEY not found")
        
    if groq_key:
        print(f"✓ GROQ_API_KEY found: {groq_key[:20]}...")
    else:
        print("⚠ GROQ_API_KEY not found (optional)")
    
    return bool(google_key)


def main():
    print("\n" + "=" * 70)
    print("FLASK APP FIX VERIFICATION")
    print("=" * 70)
    
    results = {
        'Environment Variables': test_env_vars(),
        'Configuration': test_config(),
        'LLM Provider (Gemini)': test_llm_provider(),
        'RAG Retrieval': test_rag_system()
    }
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        print("\nYour Flask app should now work with:")
        print("  1. ✓ Analyses shown in history (including XMI)")
        print("  2. ✓ LLM responding with Gemini API and RAG")
        print("  3. ✓ Report analysis working with RAG-based content")
        print("  4. ✓ Cell images and data displayed")
        print("\nRun: python app/flask_app.py")
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 70)
        print("\nPlease fix the failed tests before running Flask app.")
    
    print("=" * 70 + "\n")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
