"""
RAG System Comprehensive Testing Suite
=======================================
Tests extractive fallback, LLM integration, and system performance.

Usage:
    python test_system.py                    # Run all tests
    python test_system.py --mode llm         # Test only LLM mode
    python test_system.py --mode fallback    # Test only extractive fallback
    python test_system.py --class 5          # Test specific class

Author: Intel Unnati Project Team
"""

import requests
import argparse
import time
from typing import Dict, List, Optional
from datetime import datetime


class SystemTester:
    """Comprehensive RAG system testing"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.results = {
            'success': [],
            'rejected': [],
            'error': []
        }
        
        # Comprehensive test cases
        self.test_cases = [
            # Class 5 English
            {
                'name': 'Class 5 English - Reading Comprehension',
                'class': 5,
                'subject': 'English',
                'language': 'en',
                'question': 'What is the importance of reading and how does it help students learn?'
            },
            {
                'name': 'Class 5 English - Story Analysis',
                'class': 5,
                'subject': 'English',
                'language': 'en',
                'question': 'Explain the story of Papa and his spectacles from the textbook'
            },
            
            # Class 5 Hindi
            {
                'name': 'Class 5 Hindi - Poetry',
                'class': 5,
                'subject': 'Hindi',
                'language': 'hi',
                'question': 'हिंदी पाठ्यपुस्तक में कविताओं के महत्व के बारे में बताइए'
            },
            {
                'name': 'Class 5 Hindi - Language Skills',
                'class': 5,
                'subject': 'Hindi',
                'language': 'hi',
                'question': 'भाषा कौशल विकास के लिए क्या क्या गतिविधियां महत्वपूर्ण हैं'
            },
            
            # Class 5 Physical Education
            {
                'name': 'Class 5 Physical Education - Sports',
                'class': 5,
                'subject': 'Physical Education',
                'language': 'en',
                'question': 'What are the benefits of physical education and sports activities for children?'
            },
            
            # Class 10 English
            {
                'name': 'Class 10 English - Literary Devices',
                'class': 10,
                'subject': 'English',
                'language': 'en',
                'question': 'Explain the literary devices used in poems and their importance in understanding literature'
            },
            {
                'name': 'Class 10 English - Prose Themes',
                'class': 10,
                'subject': 'English',
                'language': 'en',
                'question': 'What themes are explored in the prose lessons about courage and determination?'
            },
            
            # Class 10 Hindi - Science
            {
                'name': 'Class 10 Hindi - Water Resources',
                'class': 10,
                'subject': 'Hindi',
                'language': 'hi',
                'question': 'जल संसाधन और जल संरक्षण के महत्व के बारे में विस्तार से समझाइए'
            },
            {
                'name': 'Class 10 Hindi - Democracy',
                'class': 10,
                'subject': 'Hindi',
                'language': 'hi',
                'question': 'भारत में लोकतंत्र और नागरिकता के अधिकार के बारे में विस्तार से बताइए'
            }
        ]
    
    def check_health(self) -> bool:
        """Check if backend is running"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def run_test(self, test_case: Dict) -> Dict:
        """Run a single test case"""
        try:
            payload = {
                'question': test_case['question'],
                'class': test_case['class'],
                'subject': test_case['subject'],
                'language': test_case['language']
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.api_url}/api/query",
                json=payload,
                timeout=60
            )
            elapsed_time = time.time() - start_time
            
            result = response.json()
            result['response_time'] = elapsed_time
            result['test_name'] = test_case['name']
            
            return result
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'test_name': test_case['name']
            }
    
    def print_test_result(self, result: Dict, verbose: bool = True):
        """Print formatted test result"""
        name = result.get('test_name', 'Unknown')
        status = result.get('status', 'unknown')
        
        if status == 'success':
            mode = result.get('metadata', {}).get('mode', 'N/A')
            chunks = result.get('metadata', {}).get('retrieved_chunks', 0)
            avg_score = result.get('metadata', {}).get('avg_retrieval_score', 0)
            response_time = result.get('response_time', 0)
            answer_len = len(result.get('answer', ''))
            
            print(f"✓ {name}")
            if verbose:
                print(f"  Mode: {mode}")
                print(f"  Chunks: {chunks}, Avg Score: {avg_score:.4f}")
                print(f"  Answer Length: {answer_len} chars")
                print(f"  Response Time: {response_time:.2f}s")
            
            self.results['success'].append(result)
            
        elif status == 'rejected':
            reason = result.get('reason', 'Unknown')
            rejection_type = result.get('rejection_type', 'Unknown')
            top_score = result.get('metadata', {}).get('top_score', 0)
            
            print(f"✗ {name}")
            if verbose:
                print(f"  Reason: {reason}")
                print(f"  Type: {rejection_type}")
                print(f"  Top Score: {top_score:.4f}")
            
            self.results['rejected'].append(result)
            
        else:
            error = result.get('error', 'Unknown error')
            print(f"✗ {name}")
            if verbose:
                print(f"  Error: {error}")
            
            self.results['error'].append(result)
    
    def run_all_tests(self, filter_class: Optional[int] = None, 
                     filter_mode: Optional[str] = None, verbose: bool = True):
        """Run all test cases"""
        
        # Filter tests
        tests = self.test_cases
        if filter_class:
            tests = [t for t in tests if t['class'] == filter_class]
        
        print("\n" + "="*80)
        print("RAG SYSTEM COMPREHENSIVE TEST SUITE")
        print("="*80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Tests: {len(tests)}")
        print(f"API Endpoint: {self.api_url}")
        
        # Check backend health
        print("\nChecking backend health...", end=" ")
        if not self.check_health():
            print("✗ FAILED")
            print("\nError: Backend is not responding!")
            print("Please start the backend: uvicorn main:app --reload --host 0.0.0.0 --port 8000")
            return
        print("✓ OK")
        
        print("\n" + "="*80)
        print("RUNNING TESTS")
        print("="*80 + "\n")
        
        # Run tests
        for i, test in enumerate(tests, 1):
            print(f"[{i}/{len(tests)}] ", end="")
            result = self.run_test(test)
            self.print_test_result(result, verbose)
            print()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print comprehensive summary"""
        total = len(self.results['success']) + len(self.results['rejected']) + len(self.results['error'])
        
        print("="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        # Overall statistics
        print(f"\nTotal Tests: {total}")
        print(f"✓ Success: {len(self.results['success'])} ({100*len(self.results['success'])/total:.1f}%)")
        print(f"✗ Rejected: {len(self.results['rejected'])} ({100*len(self.results['rejected'])/total:.1f}%)")
        print(f"? Errors: {len(self.results['error'])} ({100*len(self.results['error'])/total:.1f}%)")
        
        # Mode distribution
        if self.results['success']:
            print("\n" + "-"*80)
            print("MODE DISTRIBUTION")
            print("-"*80)
            mode_counts = {}
            for result in self.results['success']:
                mode = result.get('metadata', {}).get('mode', 'unknown')
                mode_counts[mode] = mode_counts.get(mode, 0) + 1
            
            for mode, count in mode_counts.items():
                print(f"  {mode}: {count} queries")
        
        # Performance metrics
        if self.results['success']:
            response_times = [r.get('response_time', 0) for r in self.results['success']]
            avg_time = sum(response_times) / len(response_times)
            max_time = max(response_times)
            min_time = min(response_times)
            
            print("\n" + "-"*80)
            print("PERFORMANCE METRICS")
            print("-"*80)
            print(f"  Avg Response Time: {avg_time:.2f}s")
            print(f"  Min Response Time: {min_time:.2f}s")
            print(f"  Max Response Time: {max_time:.2f}s")
        
        # Class-wise breakdown
        print("\n" + "-"*80)
        print("CLASS-WISE BREAKDOWN")
        print("-"*80)
        
        all_results = self.results['success'] + self.results['rejected'] + self.results['error']
        class_stats = {}
        
        for result in all_results:
            name = result.get('test_name', '')
            if 'Class 5' in name:
                class_key = 'Class 5'
            elif 'Class 10' in name:
                class_key = 'Class 10'
            else:
                continue
            
            if class_key not in class_stats:
                class_stats[class_key] = {'success': 0, 'rejected': 0, 'error': 0}
            
            status = result.get('status', 'error')
            if status == 'success':
                class_stats[class_key]['success'] += 1
            elif status == 'rejected':
                class_stats[class_key]['rejected'] += 1
            else:
                class_stats[class_key]['error'] += 1
        
        for class_name, stats in sorted(class_stats.items()):
            total_class = sum(stats.values())
            success_pct = 100 * stats['success'] / total_class if total_class > 0 else 0
            print(f"\n{class_name}:")
            print(f"  Success: {stats['success']}/{total_class} ({success_pct:.1f}%)")
            if stats['rejected'] > 0:
                print(f"  Rejected: {stats['rejected']}")
            if stats['error'] > 0:
                print(f"  Errors: {stats['error']}")
        
        print("\n" + "="*80)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='RAG System Comprehensive Testing Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--api', type=str, default='http://localhost:8000',
                       help='API endpoint URL')
    parser.add_argument('--class', type=int, dest='class_filter',
                       help='Filter tests by class number')
    parser.add_argument('--mode', type=str, choices=['llm', 'fallback'],
                       help='Filter tests by mode')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output')
    
    args = parser.parse_args()
    
    tester = SystemTester(api_url=args.api)
    tester.run_all_tests(
        filter_class=args.class_filter,
        filter_mode=args.mode,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
