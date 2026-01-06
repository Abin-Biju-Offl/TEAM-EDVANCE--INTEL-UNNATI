"""
Groq LLM Specific Testing Suite
================================
Tests Groq API integration, token usage, and rate limiting.

Usage:
    python test_llm.py                    # Run LLM tests
    python test_llm.py --direct           # Test Groq API directly
    python test_llm.py --rate-limit       # Test rate limiting behavior

Author: Intel Unnati Project Team
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.groq_service import groq_service
from app.services.faiss_service import faiss_service
from app.services.embedding_service import embedding_service
from loguru import logger


class LLMTester:
    """Groq LLM-specific testing"""
    
    def __init__(self):
        self.test_queries = [
            {
                'class': 5,
                'subject': 'English',
                'language': 'en',
                'query': 'What is the importance of reading?',
                'expected_keywords': ['reading', 'learn', 'skill']
            },
            {
                'class': 10,
                'subject': 'English',
                'language': 'en',
                'query': 'Explain literary devices in poetry',
                'expected_keywords': ['metaphor', 'simile', 'poem']
            }
        ]
    
    def test_groq_availability(self) -> bool:
        """Test if Groq API is available"""
        print("\n" + "="*70)
        print("GROQ API AVAILABILITY TEST")
        print("="*70)
        
        try:
            is_available = groq_service.is_available()
            
            if is_available:
                print("✓ Groq API: AVAILABLE")
                print(f"  Model: {groq_service.model}")
                return True
            else:
                print("✗ Groq API: UNAVAILABLE")
                print("  System will fall back to extractive mode")
                return False
                
        except Exception as e:
            print(f"✗ Error checking Groq: {e}")
            return False
    
    def test_direct_generation(self) -> bool:
        """Test direct LLM generation"""
        print("\n" + "="*70)
        print("DIRECT LLM GENERATION TEST")
        print("="*70)
        
        try:
            # Load a test index
            print("\nLoading test index (Class 5 English)...")
            success = faiss_service.load_index_for_class("5", "english", "en")
            
            if not success:
                print("✗ Failed to load index")
                return False
            
            print(f"✓ Index loaded: {faiss_service.index.ntotal} vectors")
            
            # Test query
            query = "What is reading?"
            print(f"\nQuery: {query}")
            
            # Generate embedding and search
            query_embedding = embedding_service.embed_query(query)
            results = faiss_service.search(query_embedding, k=5)
            
            if not results:
                print("✗ No search results")
                return False
            
            print(f"✓ Retrieved {len(results)} chunks")
            print(f"  Top score: {results[0]['score']:.4f}")
            
            # Test LLM generation
            print("\nTesting LLM generation...")
            start_time = time.time()
            
            answer = groq_service.generate_answer(
                query=query,
                context_chunks=results,
                language='en'
            )
            
            elapsed_time = time.time() - start_time
            
            if answer and 'answer' in answer:
                print(f"✓ LLM generation successful")
                print(f"  Response time: {elapsed_time:.2f}s")
                print(f"  Answer length: {len(answer['answer'])} chars")
                print(f"\nAnswer preview:")
                print("-" * 70)
                preview = answer['answer'][:300] + "..." if len(answer['answer']) > 300 else answer['answer']
                print(preview)
                print("-" * 70)
                return True
            else:
                print("✗ LLM generation failed")
                return False
                
        except Exception as e:
            print(f"✗ Error in direct generation: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_rate_limiting(self) -> Dict:
        """Test rate limiting behavior"""
        print("\n" + "="*70)
        print("RATE LIMITING TEST")
        print("="*70)
        
        results = {
            'total_requests': 0,
            'successful': 0,
            'rate_limited': 0,
            'errors': 0
        }
        
        try:
            # Load index
            print("\nLoading test index...")
            faiss_service.load_index_for_class("5", "english", "en")
            
            # Prepare test queries
            test_queries = [
                "What is reading?",
                "Explain learning",
                "What are stories?",
                "Describe books"
            ]
            
            print(f"\nSending {len(test_queries)} rapid requests...\n")
            
            for i, query in enumerate(test_queries, 1):
                results['total_requests'] += 1
                print(f"[{i}/{len(test_queries)}] Query: {query[:40]}...")
                
                try:
                    # Search
                    query_embedding = embedding_service.embed_query(query)
                    chunks = faiss_service.search(query_embedding, k=5)
                    
                    # Generate answer
                    start_time = time.time()
                    answer = groq_service.generate_answer(query, chunks, 'en')
                    elapsed_time = time.time() - start_time
                    
                    if answer and 'answer' in answer:
                        results['successful'] += 1
                        print(f"  ✓ Success ({elapsed_time:.2f}s)")
                    else:
                        # Check if it's rate limiting
                        if 'rate_limit_exceeded' in str(answer).lower():
                            results['rate_limited'] += 1
                            print(f"  ⚠ Rate limited")
                        else:
                            results['errors'] += 1
                            print(f"  ✗ Failed")
                    
                    # Small delay between requests
                    time.sleep(0.5)
                    
                except Exception as e:
                    if '429' in str(e) or 'rate' in str(e).lower():
                        results['rate_limited'] += 1
                        print(f"  ⚠ Rate limited")
                    else:
                        results['errors'] += 1
                        print(f"  ✗ Error: {str(e)[:50]}")
            
            # Print summary
            print("\n" + "-"*70)
            print("RATE LIMITING SUMMARY")
            print("-"*70)
            print(f"Total Requests: {results['total_requests']}")
            print(f"Successful: {results['successful']}")
            print(f"Rate Limited: {results['rate_limited']}")
            print(f"Errors: {results['errors']}")
            
            if results['rate_limited'] > 0:
                print("\n⚠ Rate limiting detected!")
                print("  The system will automatically fall back to extractive mode.")
            
            return results
            
        except Exception as e:
            print(f"✗ Error in rate limiting test: {e}")
            return results
    
    def run_all_tests(self, test_direct: bool = True, 
                     test_rate: bool = False):
        """Run all LLM tests"""
        print("\n" + "="*70)
        print("GROQ LLM TESTING SUITE")
        print("="*70)
        print(f"Model: {groq_service.model}")
        print(f"Max Tokens: {groq_service.max_tokens}")
        print(f"Temperature: {groq_service.temperature}")
        
        results = {}
        
        # Test 1: Availability
        results['availability'] = self.test_groq_availability()
        
        if not results['availability']:
            print("\n⚠ Groq API is not available. Skipping other tests.")
            return results
        
        # Test 2: Direct generation
        if test_direct:
            results['direct_generation'] = self.test_direct_generation()
        
        # Test 3: Rate limiting
        if test_rate:
            results['rate_limiting'] = self.test_rate_limiting()
        
        # Final summary
        print("\n" + "="*70)
        print("FINAL SUMMARY")
        print("="*70)
        
        passed = sum(1 for v in results.values() if v == True)
        total = len([v for v in results.values() if isinstance(v, bool)])
        
        print(f"\nTests Passed: {passed}/{total}")
        
        for test_name, result in results.items():
            if isinstance(result, bool):
                status = "✓ PASS" if result else "✗ FAIL"
                print(f"  {status}: {test_name.replace('_', ' ').title()}")
        
        print("\n" + "="*70)
        
        return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Groq LLM Specific Testing Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--direct', action='store_true',
                       help='Test direct LLM generation')
    parser.add_argument('--rate-limit', action='store_true',
                       help='Test rate limiting behavior')
    parser.add_argument('--all', action='store_true',
                       help='Run all tests')
    
    args = parser.parse_args()
    
    # Default to all tests if no specific test selected
    if not any([args.direct, args.rate_limit]):
        args.all = True
    
    tester = LLMTester()
    tester.run_all_tests(
        test_direct=args.direct or args.all,
        test_rate=args.rate_limit or args.all
    )


if __name__ == "__main__":
    main()
