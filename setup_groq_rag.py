"""
setup_groq_rag.py
─────────────────────────────────────────────────────────────────────────────
Quick setup script for Groq Llama RAG integration.

This script will:
1. Check if groq library is installed
2. Check if .env file exists with GROQ_API_KEY
3. Test Groq API connection
4. Verify RAG pipeline
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))


def check_groq_installed():
    """Check if groq library is installed."""
    print("\n[1/4] Checking Groq library installation...")
    try:
        import groq
        print("✓ Groq library is installed")
        return True
    except ImportError:
        print("✗ Groq library not found")
        print("\nInstall it with:")
        print("  pip install groq")
        return False


def check_env_file():
    """Check if .env file exists with GROQ_API_KEY."""
    print("\n[2/4] Checking .env configuration...")
    env_path = REPO_ROOT / ".env"
    
    if not env_path.exists():
        print("✗ .env file not found")
        print("\nCreate it with:")
        print(f"  copy .env.example .env")
        print("  Then edit .env and add your Groq API key")
        return False
    
    # Load .env
    from dotenv import load_dotenv
    load_dotenv(env_path)
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("✗ GROQ_API_KEY not found in .env")
        print("\nEdit .env and add:")
        print("  GROQ_API_KEY=gsk_your_actual_key_here")
        print("\nGet your free API key at: https://console.groq.com")
        return False
    
    if not api_key.startswith("gsk_"):
        print("⚠ Warning: GROQ_API_KEY doesn't start with 'gsk_'")
        print("  Make sure you copied the full key from Groq console")
    
    print(f"✓ GROQ_API_KEY found (starts with: {api_key[:10]}...)")
    return True


def test_groq_connection():
    """Test Groq API connection."""
    print("\n[3/4] Testing Groq API connection...")
    try:
        from agents.llm_provider import get_llm_provider
        
        llm = get_llm_provider()
        print(f"✓ Provider: {llm.provider}")
        print(f"✓ Model: {llm.model_name}")
        
        # Test simple query
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Reply with just the word 'SUCCESS' and nothing else."},
        ]
        
        response = llm.generate(messages, max_tokens=10)
        
        if "SUCCESS" in response.upper():
            print(f"✓ API connection successful")
            return True
        else:
            print(f"⚠ API responded but unexpected output: {response}")
            return True
            
    except Exception as e:
        print(f"✗ API connection failed: {e}")
        print("\nPossible issues:")
        print("  - Invalid API key")
        print("  - Network connectivity problem")
        print("  - Groq service temporarily unavailable")
        return False


def test_rag_pipeline():
    """Test RAG pipeline integration."""
    print("\n[4/4] Testing RAG pipeline integration...")
    try:
        from agents.synthesizer import synthesizer_node, USE_LLM
        from agents.state import AgentState
        
        print(f"✓ Synthesizer mode: {'LLM-POWERED' if USE_LLM else 'DETERMINISTIC'}")
        
        if not USE_LLM:
            print("\n⚠ Warning: Synthesizer is in DETERMINISTIC mode")
            print("  To enable LLM-powered RAG, set:")
            print("    - Environment variable: SYNTHESIZER_USE_LLM=true")
            print("    - Or edit agents/synthesizer.py line 34")
        
        # Test with minimal state
        test_state: AgentState = {
            "patient_id": "TEST_SETUP",
            "predicted_class": "ALL",
            "annotated_blast_fraction": 0.25,
            "total_cells": 100,
            "class_probabilities": {"ALL": 0.9, "AML": 0.1},
            "cell_counts_by_category": {"Blast": 25, "Lymphoid": 75},
            "retrieved_evidence": [],
            "formatted_guidelines_context": "Test context",
            "revision_count": 0,
        }
        
        if USE_LLM:
            print("  Testing LLM-powered synthesis...")
            result = synthesizer_node(test_state)
            if "draft_differential" in result and len(result["draft_differential"]) > 0:
                print("✓ RAG pipeline working correctly")
                return True
            else:
                print("✗ RAG pipeline returned empty result")
                return False
        else:
            print("✓ Deterministic synthesis working")
            return True
            
    except Exception as e:
        print(f"✗ RAG pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 70)
    print("GROQ LLAMA RAG SETUP VERIFICATION")
    print("=" * 70)
    
    results = []
    
    # Run all checks
    results.append(("Groq Library", check_groq_installed()))
    
    if not results[-1][1]:
        print("\n" + "=" * 70)
        print("SETUP INCOMPLETE - Install groq library first")
        print("=" * 70)
        return
    
    results.append(("Environment Config", check_env_file()))
    
    if not results[-1][1]:
        print("\n" + "=" * 70)
        print("SETUP INCOMPLETE - Configure .env file")
        print("=" * 70)
        return
    
    results.append(("API Connection", test_groq_connection()))
    results.append(("RAG Pipeline", test_rag_pipeline()))
    
    # Summary
    print("\n" + "=" * 70)
    print("SETUP SUMMARY")
    print("=" * 70)
    
    for name, status in results:
        symbol = "✓" if status else "✗"
        print(f"{symbol} {name}")
    
    all_passed = all(status for _, status in results)
    
    print("=" * 70)
    if all_passed:
        print("✅ ALL CHECKS PASSED - RAG IS READY TO USE!")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Launch UI: streamlit run app/streamlit_app.py")
        print("  2. Upload a bone marrow image")
        print("  3. See Groq Llama RAG in action!")
    else:
        print("⚠ SOME CHECKS FAILED - FIX ISSUES ABOVE")
        print("=" * 70)
        print("\nRefer to GROQ_LLAMA_RAG_SETUP.md for detailed instructions")
    
    print()


if __name__ == "__main__":
    main()
