"""
Quick test script to verify Grok LLM integration

Run this after setting up your Grok API key.
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.grok_service import grok_service
from app.core.config import settings


def test_grok_availability():
    """Test if Grok service is properly configured"""
    print("=" * 60)
    print("GROK LLM INTEGRATION TEST")
    print("=" * 60)
    
    print("\n1. Configuration Status:")
    print(f"   API Key Set: {'✓ Yes' if settings.grok_api_key and settings.grok_api_key != 'your_grok_api_key_here' else '✗ No (update .env file)'}")
    print(f"   Model: {settings.grok_model}")
    print(f"   Temperature: {settings.grok_temperature}")
    print(f"   Max Tokens: {settings.grok_max_tokens}")
    print(f"   LLM Enabled: {'✓ Yes' if settings.use_llm else '✗ No'}")
    
    print("\n2. Service Status:")
    if grok_service.is_available():
        print("   ✓ Grok service is AVAILABLE")
        print("   ✓ System will use Grok LLM for answer generation")
    else:
        print("   ✗ Grok service is NOT available")
        print("   ! System will use extractive fallback mode")
        if not settings.grok_api_key or settings.grok_api_key == "your_grok_api_key_here":
            print("   → Action required: Add your Grok API key to backend/.env file")
        elif not settings.use_llm:
            print("   → LLM is disabled in settings (USE_LLM=false)")
    
    print("\n3. Quick Test:")
    if grok_service.is_available():
        try:
            print("   Testing Grok API connection...")
            # Create a simple test context
            test_chunks = [
                {
                    "text": "Photosynthesis is the process by which green plants use sunlight to synthesize food from carbon dioxide and water.",
                    "metadata": {
                        "class": "10",
                        "subject": "Science",
                        "chapter": "Life Processes",
                        "page": "95"
                    },
                    "primary_page": "95",
                    "score": 0.95
                }
            ]
            
            answer, citations, score = grok_service.generate_answer(
                question="What is photosynthesis?",
                retrieved_chunks=test_chunks,
                class_num=10,
                subject="Science",
                language="en"
            )
            
            print("   ✓ API connection successful!")
            print(f"   ✓ Generated {len(answer)} characters")
            print(f"   ✓ {len(citations)} citations extracted")
            print(f"   ✓ Grounding score: {score:.2f}")
            print("\n   Sample answer preview:")
            print(f"   {answer[:150]}...")
            
        except Exception as e:
            print(f"   ✗ API test failed: {str(e)}")
            print("   → Check your API key and internet connection")
    else:
        print("   Skipped (Grok service not available)")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if grok_service.is_available():
        print("✓ Grok LLM integration is working correctly!")
        print("✓ Your system is ready to generate intelligent answers.")
        print("\nNext steps:")
        print("  1. Start the backend server: python main.py")
        print("  2. Open the frontend and test with real queries")
        print("  3. Monitor logs/app_*.log for detailed information")
    else:
        print("✗ Grok LLM is not yet configured.")
        print("\nTo enable Grok LLM:")
        print("  1. Get API key from https://console.x.ai")
        print("  2. Edit backend/.env file")
        print("  3. Set GROK_API_KEY=xai-your-key-here")
        print("  4. Restart the server")
        print("\nNote: System will still work with extractive fallback mode.")
    
    print("=" * 60)


if __name__ == "__main__":
    test_grok_availability()
