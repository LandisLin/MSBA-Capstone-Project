"""
Enhanced API Validator with yfinance Support
Handles market indices using yfinance library for accurate validation
"""

import requests
import json
import logging
import time
from typing import Dict, Any
from datetime import datetime
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class EnhancedAPIValidator:
    """Enhanced validator with yfinance support for market indices"""
    
    def __init__(self):
        self.session = requests.Session()
        
        # Set up session with API-friendly headers
        self.session.headers.update({
            'User-Agent': 'MacroDataBot/1.0 (Data Collection)',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive'
        })
        
        self.session.max_redirects = 3
        
        # Try to import yfinance
        self.yfinance_available = self._check_yfinance_availability()
    
    def _check_yfinance_availability(self) -> bool:
        """Check if yfinance is available"""
        try:
            import yfinance as yf
            return True
        except ImportError:
            logger.warning("yfinance not available - market index validation will be limited")
            return False
    
    def validate_source_api_first(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced validation with yfinance support for market indices"""
        
        source_name = source.get('name', 'Unknown')
        main_url = source.get('url', '')
        api_url = source.get('api_url')
        source_type = source.get('source_type', 'unknown')
        data_type = source.get('data_type', 'unknown')
        
        result = {
            'source_name': source_name,
            'main_url': main_url,
            'api_url': api_url,
            'source_type': source_type,
            'data_type': data_type,
            'timestamp': datetime.now().isoformat(),
            'accessible': False,
            'response_time_ms': 0,
            'status_code': None,
            'content_type': None,
            'api_endpoint': False,
            'error_message': None,
            'validation_strategy': None,
            'tested_url': None
        }
        
        try:
            # Strategy 1: If it's a Stock_Index, try yfinance validation first
            if data_type.lower() == 'stock_index' and api_url and self.yfinance_available:
                print(f"   📈 Testing yfinance symbol: {api_url}")
                yfinance_result = self._test_yfinance_symbol(api_url)
                
                if yfinance_result['success']:
                    result.update({
                        'accessible': True,
                        'response_time_ms': yfinance_result['response_time_ms'],
                        'status_code': 200,  # Assume 200 if yfinance works
                        'content_type': 'financial_data',
                        'api_endpoint': True,
                        'validation_strategy': 'yfinance Symbol',
                        'tested_url': api_url,
                        'yfinance_data_sample': yfinance_result.get('data_sample', ''),
                        'yfinance_info': yfinance_result.get('info', {})
                    })
                    return result
                else:
                    print(f"   ❌ yfinance symbol failed: {yfinance_result.get('error', 'Unknown')}")
            
            # Strategy 2: If has API URL and not a yfinance symbol, test as regular API
            if api_url and api_url.strip() and not self._is_yfinance_symbol(api_url):
                print(f"   🔗 Testing API URL: {api_url}")
                api_result = self._test_api_url(api_url)
                
                if api_result['success']:
                    result.update({
                        'accessible': True,
                        'response_time_ms': api_result['response_time_ms'],
                        'status_code': api_result['status_code'],
                        'content_type': api_result.get('content_type', 'unknown'),
                        'api_endpoint': True,
                        'validation_strategy': 'API URL',
                        'tested_url': api_url,
                        'api_response_sample': api_result.get('response_sample', '')
                    })
                    return result
                else:
                    print(f"   ❌ API URL failed: {api_result.get('error', 'Unknown')}")
            
            # Strategy 3: Try to construct API URL if source_type is 'api'
            if source_type.lower() == 'api' and not api_url:
                constructed_api = self._try_construct_api_url(main_url)
                if constructed_api:
                    print(f"   🔗 Testing constructed API URL: {constructed_api}")
                    api_result = self._test_api_url(constructed_api)
                    
                    if api_result['success']:
                        result.update({
                            'accessible': True,
                            'response_time_ms': api_result['response_time_ms'],
                            'status_code': api_result['status_code'],
                            'content_type': api_result.get('content_type', 'unknown'),
                            'api_endpoint': True,
                            'validation_strategy': 'Constructed API URL',
                            'tested_url': constructed_api,
                            'api_response_sample': api_result.get('response_sample', '')
                        })
                        return result
                    else:
                        print(f"   ❌ Constructed API URL failed: {api_result.get('error', 'Unknown')}")
            
            # Strategy 4: Fallback to main URL
            if main_url:
                print(f"   🔗 Fallback - testing main URL: {main_url}")
                main_result = self._test_main_url(main_url)
                
                if main_result['success']:
                    result.update({
                        'accessible': True,
                        'response_time_ms': main_result['response_time_ms'],
                        'status_code': main_result['status_code'],
                        'content_type': main_result.get('content_type', 'unknown'),
                        'api_endpoint': main_result.get('looks_like_api', False),
                        'validation_strategy': 'Main URL',
                        'tested_url': main_url
                    })
                    return result
                else:
                    print(f"   ❌ Main URL failed: {main_result.get('error', 'Unknown')}")
            
            # All strategies failed
            result.update({
                'error_message': 'All validation strategies failed',
                'validation_strategy': 'All failed',
                'tested_url': api_url or main_url
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced API validation error: {e}")
            result['error_message'] = f"Validation error: {str(e)}"
            return result
    
    def _test_yfinance_symbol(self, symbol: str) -> Dict[str, Any]:
        """Test yfinance symbol for market indices"""
        
        start_time = time.time()
        
        try:
            import yfinance as yf
            
            # Clean the symbol (remove any extra characters)
            clean_symbol = symbol.strip()
            
            print(f"       Fetching data for symbol: {clean_symbol}")
            
            # Create ticker object
            ticker = yf.Ticker(clean_symbol)
            
            # Try to get recent data (last 5 days)
            hist_data = ticker.history(period="5d")
            
            end_time = time.time()
            response_time = int((end_time - start_time) * 1000)
            
            if not hist_data.empty:
                # Success - we got data
                latest_close = hist_data['Close'].iloc[-1] if len(hist_data) > 0 else 0
                latest_date = hist_data.index[-1].strftime('%Y-%m-%d') if len(hist_data) > 0 else 'Unknown'
                
                print(f"       ✅ Got data: Latest close = {latest_close:.2f} on {latest_date}")
                
                # Try to get additional info
                try:
                    info = ticker.info
                    long_name = info.get('longName', 'Unknown')
                    currency = info.get('currency', 'Unknown')
                    print(f"       📊 Info: {long_name} ({currency})")
                except:
                    info = {}
                
                return {
                    'success': True,
                    'response_time_ms': response_time,
                    'data_sample': f"Latest: {latest_close:.2f} on {latest_date}",
                    'info': {
                        'latest_close': latest_close,
                        'latest_date': latest_date,
                        'data_points': len(hist_data),
                        'symbol': clean_symbol
                    }
                }
            else:
                # No data returned
                return {
                    'success': False,
                    'error': f'No data available for symbol {clean_symbol}',
                    'response_time_ms': response_time
                }
                
        except ImportError:
            return {
                'success': False,
                'error': 'yfinance library not available',
                'response_time_ms': 0
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'yfinance error: {str(e)}',
                'response_time_ms': int((time.time() - start_time) * 1000)
            }
    
    def _is_yfinance_symbol(self, api_url: str) -> bool:
        """Check if api_url looks like a yfinance symbol"""
        
        if not api_url:
            return False
        
        # yfinance symbols are typically short strings without HTTP schemes
        api_url = api_url.strip()
        
        # If it starts with http, it's not a yfinance symbol
        if api_url.startswith(('http://', 'https://')):
            return False
        
        # If it's short and contains typical market symbols, likely yfinance
        if len(api_url) <= 20 and any(char in api_url for char in ['^', '.', '-']):
            return True
        
        # Common patterns: ^GSPC, AAPL, 000001.SS, etc.
        if len(api_url) <= 10:
            return True
        
        return False
    
    def _test_api_url(self, api_url: str) -> Dict[str, Any]:
        """Test regular API URL"""
        
        start_time = time.time()
        
        try:
            # Special handling for different API types
            if 'datastore_search' in api_url:
                # Singapore data.gov.sg API
                response = self.session.get(api_url, timeout=15)
            elif 'fred.stlouisfed.org' in api_url:
                # FRED API
                response = self.session.get(api_url, timeout=15)
            else:
                # Generic API test
                response = self.session.get(api_url, timeout=15)
            
            end_time = time.time()
            response_time = int((end_time - start_time) * 1000)
            
            # Consider API successful if status < 500
            success = response.status_code < 500
            
            result = {
                'success': success,
                'response_time_ms': response_time,
                'status_code': response.status_code,
                'content_type': response.headers.get('content-type', 'unknown')
            }
            
            # Try to get a sample of the response
            if success:
                try:
                    if 'json' in response.headers.get('content-type', ''):
                        sample = response.text[:500]
                        result['response_sample'] = sample
                except:
                    pass
            
            if not success:
                result['error'] = f"HTTP {response.status_code}"
            
            return result
            
        except requests.exceptions.Timeout:
            return {
                'success': False,
                'error': 'API request timeout',
                'response_time_ms': int((time.time() - start_time) * 1000)
            }
        except requests.exceptions.ConnectionError:
            return {
                'success': False,
                'error': 'API connection failed',
                'response_time_ms': int((time.time() - start_time) * 1000)
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'API test error: {str(e)}',
                'response_time_ms': int((time.time() - start_time) * 1000)
            }
    
    def _try_construct_api_url(self, main_url: str) -> str:
        """Try to construct API URL from main URL"""
        
        try:
            # Singapore data.gov.sg pattern
            if 'data.gov.sg/datasets/' in main_url:
                import re
                match = re.search(r'/datasets/([^/]+)', main_url)
                if match:
                    dataset_id = match.group(1)
                    return f"https://data.gov.sg/api/action/datastore_search?resource_id={dataset_id}"
            
            # FRED pattern
            if 'fred.stlouisfed.org/series/' in main_url:
                import re
                series_match = re.search(r'/series/([^/?]+)', main_url)
                if series_match:
                    series_id = series_match.group(1)
                    return f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key=TEST&file_type=json"
            
            return None
            
        except Exception as e:
            logger.debug(f"API construction failed: {e}")
            return None
    
    def _test_main_url(self, main_url: str) -> Dict[str, Any]:
        """Test main URL as fallback"""
        
        start_time = time.time()
        
        try:
            # Use HEAD request first (faster)
            response = self.session.head(main_url, timeout=10, allow_redirects=True)
            end_time = time.time()
            
            response_time = int((end_time - start_time) * 1000)
            success = response.status_code < 400
            
            result = {
                'success': success,
                'response_time_ms': response_time,
                'status_code': response.status_code,
                'content_type': response.headers.get('content-type', 'unknown'),
                'looks_like_api': 'json' in response.headers.get('content-type', '').lower()
            }
            
            if not success:
                result['error'] = f"HTTP {response.status_code}"
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Main URL error: {str(e)}',
                'response_time_ms': int((time.time() - start_time) * 1000)
            }

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def test_yfinance_symbols():
    """Test yfinance with common market symbols"""
    
    print("🧪 TESTING YFINANCE SYMBOLS")
    print("=" * 50)
    
    test_symbols = [
        ("S&P 500", "^GSPC"),
        ("NASDAQ", "^IXIC"),
        ("Hang Seng", "^HSI"),
        ("Nikkei 225", "^N225"),
        ("Shanghai Composite", "000001.SS"),
        ("Singapore STI", "^STI"),
        ("VIX", "^VIX"),
        ("EURO STOXX 50", "^STOXX50E")
    ]
    
    validator = EnhancedAPIValidator()
    
    if not validator.yfinance_available:
        print("❌ yfinance not available - install with: pip install yfinance")
        return
    
    working_symbols = []
    
    for name, symbol in test_symbols:
        print(f"\n📊 Testing {name} ({symbol}):")
        
        result = validator._test_yfinance_symbol(symbol)
        
        if result['success']:
            working_symbols.append((name, symbol))
            print(f"   ✅ Working: {result.get('data_sample', 'No sample')}")
        else:
            print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
    
    print(f"\n🎯 SUMMARY:")
    print(f"   Working symbols: {len(working_symbols)}/{len(test_symbols)}")
    print(f"   Success rate: {len(working_symbols)/len(test_symbols)*100:.1f}%")
    
    if working_symbols:
        print(f"\n✅ Working symbols:")
        for name, symbol in working_symbols:
            print(f"   • {name}: {symbol}")

if __name__ == "__main__":
    test_yfinance_symbols()