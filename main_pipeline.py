"""
LLM-Free Main Pipeline - Pure API/HTTP Validation + Rule-Based Method Selection
No LLM dependencies - uses direct API validation and simple rules
"""

import logging
import json
import time
from datetime import datetime
from typing import List, Dict, Any

# Import components (NO LLM dependencies)
from api_first_validator import EnhancedAPIValidator as APIFirstValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LLMFreePipeline:
    """
    LLM-Free pipeline: Pure API validation + Rule-based method selection
    No OpenAI, CrewAI, or any LLM dependencies
    """
    
    def __init__(self):
        """Initialize the LLM-free pipeline"""
        self.execution_log = []
        self.api_validator = APIFirstValidator()
        self.rule_selector = self._create_rule_based_selector()
        
        logger.info("LLM-Free pipeline initialized - Pure API validation + Rule-based methods")
    
    def _create_rule_based_selector(self):
        """Create simple rule-based method selector (NO LLM)"""
        
        class RuleBasedMethodSelector:
            """Simple rules - NO LLM needed"""
            
            def select_method(self, source: Dict[str, Any], validation_result: Dict[str, Any] = None) -> str:
                """Select method using simple rules only"""
                
                source_type = source.get('source_type', '').lower()
                api_url = source.get('api_url')
                data_type = source.get('data_type', '').lower()
                
                # Rule 1: If source_type is 'api' OR has api_url -> api_extraction
                if source_type == 'api' or (api_url and api_url.strip()):
                    return 'api_extraction'
                
                # Rule 2: Stock indices use yfinance (which is API-based)
                if data_type == 'stock_index':
                    return 'api_extraction'
                
                # Rule 3: Everything else -> basic_web_scraping
                return 'basic_web_scraping'
            
            def determine_status(self, validation_result: Dict[str, Any]) -> str:
                """Determine status from validation result"""
                
                if not validation_result:
                    return 'not_working'
                
                accessible = validation_result.get('accessible', False)
                status_code = validation_result.get('status_code')
                
                # API-friendly logic: even 400s might be valid API responses
                if accessible and status_code:
                    return 'working' if status_code < 500 else 'not_working'
                else:
                    return 'not_working'
        
        return RuleBasedMethodSelector()
    
    def process_single_source(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """Process single source with API validation + rule-based method selection (NO LLM)"""
        
        source_name = source.get('name', 'Unknown')
        source_url = source.get('url', '')
        api_url = source.get('api_url', '')
        
        try:
            print(f"📊 Processing: {source_name}")
            print(f"   Main URL: {source_url}")
            if api_url:
                print(f"   API URL: {api_url}")
            print(f"   Type: {source.get('source_type', 'unknown')}")
            
            # Step 1: API-first validation using pure HTTP/API calls
            validation_result = self.api_validator.validate_source_api_first(source)
            
            accessible = validation_result.get('accessible', False)
            response_time = validation_result.get('response_time_ms', 0)
            status_code = validation_result.get('status_code', 'Unknown')
            strategy = validation_result.get('validation_strategy', 'Unknown')
            tested_url = validation_result.get('tested_url', 'None')
            
            print(f"   Validation Strategy: {strategy}")
            print(f"   Tested URL: {tested_url}")
            print(f"   Accessible: {accessible}")
            print(f"   Response time: {response_time}ms")
            print(f"   Status code: {status_code}")
            
            # Step 2: Rule-based method selection (NO LLM)
            method = self.rule_selector.select_method(source, validation_result)
            status = self.rule_selector.determine_status(validation_result)
            
            print(f"   Method: {method} (rule-based)")
            print(f"   Status: {status}")
            
            # Step 3: Format result consistently
            formatted_output = f"""Source: {source_name}
Method: {method}
Status: {status}
URL: {source_url}"""
            
            print(f"   ✅ Result:")
            for line in formatted_output.split('\n'):
                print(f"      {line}")
            
            return {
                'status': 'completed',
                'result': {
                    'source': source_name,
                    'method': method,
                    'status': status,
                    'url': source_url,
                    'validation_details': validation_result,
                    'timestamp': datetime.now().isoformat()
                },
                'formatted_output': formatted_output
            }
            
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            print(f"   ❌ {error_msg}")
            
            return {
                'status': 'failed',
                'error': error_msg,
                'source': source_name
            }
    
    def run_source_validation(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run validation for all sources (NO LLM)"""
        
        start_time = datetime.now()
        
        print(f"🚀 LLM-FREE MAIN PIPELINE")
        print(f"Pure API/HTTP validation + Rule-based method selection")
        print(f"Processing {len(sources)} sources")
        print("=" * 60)
        
        all_formatted_outputs = []
        successful_sources = 0
        
        for i, source in enumerate(sources, 1):
            print(f"\n📋 Source {i}/{len(sources)}")
            
            # Process single source
            result = self.process_single_source(source)
            
            if result['status'] == 'completed':
                all_formatted_outputs.append(result['formatted_output'])
                
                # Check if actually working
                if 'Status: working' in result['formatted_output']:
                    successful_sources += 1
            
            # Small delay between sources
            time.sleep(0.5)
        
        # Final summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        total_sources = len(sources)
        success_rate = (successful_sources / total_sources * 100) if total_sources > 0 else 0
        
        print(f"\n🎯 PIPELINE SUMMARY:")
        print(f"   Total Sources: {total_sources}")
        print(f"   Working Sources: {successful_sources}")
        print(f"   Success Rate: {success_rate:.1f}%")
        print(f"   Duration: {duration:.1f} seconds")
        print(f"   🚀 LLM Usage: NONE (Pure API/HTTP)")
        print(f"   🚀 Method Selection: Rule-based only")
        
        execution_result = {
            'status': 'completed',
            'pipeline_type': 'llm_free_api_validation_rule_methods',
            'sources_processed': total_sources,
            'successful_sources': successful_sources,
            'success_rate': success_rate,
            'duration_seconds': duration,
            'all_formatted_outputs': all_formatted_outputs,
            'llm_usage': 'NONE',
            'validation_method': 'Pure API/HTTP',
            'method_selection': 'Rule-based',
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat()
        }
        
        # Log execution
        self._log_execution('llm_free_pipeline_validation', execution_result)
        
        return execution_result
    
    def validate_single_source(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """Validate single source (for compatibility)"""
        
        result = self.process_single_source(source)
        
        if result['status'] == 'completed':
            return {
                'validation': result['result']['validation_details'],
                'method_selection': {
                    'recommended_method': result['result']['method'],
                    'status': result['result']['status']
                },
                'formatted_output': result['formatted_output']
            }
        else:
            return {
                'error': result['error'],
                'source': result['source']
            }
    
    def test_pipeline_components(self) -> Dict[str, Any]:
        """Test pipeline components (NO LLM)"""
        
        test_results = {
            'api_validator': False,
            'rule_selector': False,
            'overall_status': False
        }
        
        try:
            # Test API validator
            try:
                test_source = {
                    'name': 'Test Source',
                    'url': 'https://www.google.com',
                    'source_type': 'web_scraping',
                    'api_url': None
                }
                validation_result = self.api_validator.validate_source_api_first(test_source)
                test_results['api_validator'] = validation_result is not None
            except Exception as e:
                logger.error(f"API validator test failed: {e}")
            
            # Test rule selector
            try:
                test_source = {'source_type': 'api', 'api_url': 'test'}
                method = self.rule_selector.select_method(test_source)
                test_results['rule_selector'] = method == 'api_extraction'
            except Exception as e:
                logger.error(f"Rule selector test failed: {e}")
            
            # Overall status (NO LLM checks)
            test_results['overall_status'] = all([
                test_results['api_validator'],
                test_results['rule_selector']
            ])
            
            return test_results
            
        except Exception as e:
            logger.error(f"Component testing failed: {e}")
            test_results['error'] = str(e)
            return test_results
    
    def _log_execution(self, execution_type: str, result: Dict[str, Any]):
        """Log pipeline execution"""
        
        log_entry = {
            'type': execution_type,
            'timestamp': datetime.now().isoformat(),
            'result_status': result.get('status', 'unknown'),
            'duration': result.get('duration_seconds', 0),
            'sources_processed': result.get('sources_processed', 0),
            'success_rate': result.get('success_rate', 0)
        }
        
        self.execution_log.append(log_entry)
        
        # Keep only last 10 executions
        if len(self.execution_log) > 10:
            self.execution_log = self.execution_log[-10:]

# =============================================================================
# UTILITY FUNCTIONS (NO LLM)
# =============================================================================

def create_llm_free_pipeline() -> LLMFreePipeline:
    """Factory function to create LLM-free pipeline instance"""
    return LLMFreePipeline()

def run_quick_validation(sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Quick validation using LLM-free pipeline"""
    
    try:
        pipeline = create_llm_free_pipeline()
        return pipeline.run_source_validation(sources)
    except Exception as e:
        logger.error(f"Quick validation failed: {e}")
        return {
            'status': 'failed',
            'error': str(e),
            'sources_processed': len(sources)
        }

def test_with_sample_sources() -> Dict[str, Any]:
    """Test pipeline with sample sources (NO LLM)"""
    
    # Hardcoded sample sources (no dependencies)
    sample_sources = [
        {
            'name': 'Singapore GDP Data',
            'url': 'https://data.gov.sg/datasets/d_87a41037716428a042305df9b948ece2/view',
            'source_type': 'api',
            'data_type': 'GDP',
            'api_url': 'https://data.gov.sg/api/action/datastore_search?resource_id=d_87a41037716428a042305df9b948ece2'
        },
        {
            'name': 'S&P 500 Index',
            'url': 'https://finance.yahoo.com/quote/%5EGSPC/',
            'source_type': 'api',
            'data_type': 'stock_index',
            'api_url': '^GSPC'
        }
    ]
    
    logger.info(f"Testing LLM-free pipeline with {len(sample_sources)} sources")
    
    pipeline = create_llm_free_pipeline()
    return pipeline.run_source_validation(sample_sources)

# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("🚀 LLM-FREE MAIN PIPELINE")
    print("Pure API/HTTP Validation + Rule-Based Method Selection")
    print("✅ No LLM, OpenAI, or CrewAI dependencies!")
    print("=" * 60)
    
    try:
        # Test components
        print("🧪 Testing Pipeline Components...")
        pipeline = create_llm_free_pipeline()
        component_test = pipeline.test_pipeline_components()
        
        for component, status in component_test.items():
            if component != 'overall_status':
                icon = "✅" if status else "❌"
                print(f"   {icon} {component.replace('_', ' ').title()}: {status}")
        
        overall_ok = component_test.get('overall_status', False)
        print(f"\n🎯 Overall Status: {'✅ Ready' if overall_ok else '❌ Issues detected'}")
        
        if overall_ok:
            # Run sample test
            print("\n📊 Running Sample Test...")
            result = test_with_sample_sources()
            
            if result['status'] == 'completed':
                print(f"✅ Pipeline test completed!")
                print(f"   Duration: {result.get('duration_seconds', 0):.1f} seconds")
                print(f"   Sources processed: {result.get('sources_processed', 0)}")
                print(f"   Success rate: {result.get('success_rate', 0):.1f}%")
                print(f"   🚀 LLM usage: {result.get('llm_usage', 'NONE')}")
                print(f"   🚀 Validation method: {result.get('validation_method', 'Pure API/HTTP')}")
                print(f"   🚀 Method selection: {result.get('method_selection', 'Rule-based')}")
            else:
                print(f"❌ Pipeline test failed: {result.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        print("💡 Please check your configuration")
    
    print(f"\n🎯 LLM-free pipeline ready!")
    print("Features:")
    print("  ✅ Pure API/HTTP validation")
    print("  ✅ Rule-based method selection")
    print("  ✅ No LLM dependencies")
    print("  ✅ Works immediately without API keys")
