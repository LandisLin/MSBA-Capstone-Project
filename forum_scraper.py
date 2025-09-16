"""
Forum Scraper for HardwareZone Property Discussions
Based on existing news scraper architecture with crawl4ai
"""

import asyncio
import json
import os
import time
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse
import re
from typing import List, Dict, Optional

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Configuration
load_dotenv()
TESTING_MODE = os.getenv('TESTING_MODE', 'false').lower() == 'true'
MAX_POSTS_FOR_TESTING = int(os.getenv('MAX_POSTS_FOR_TESTING', '200'))  # Increased for better testing

# Forum configuration
FORUM_CONFIG = {
    "base_url": "https://forums.hardwarezone.com.sg",
    "rate_limit_delay": 3,  # Seconds between requests
    "start_date": "2025-01-01",  # Easily adjustable
    "max_pages": 500,  # Increased for large threads
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

class ForumPost:
    """Data class for forum posts"""
    def __init__(self, post_id: str, username: str, post_date: datetime, content: str, post_url: str):
        self.post_id = post_id
        self.username = username
        self.post_date = post_date
        self.content = content
        self.post_url = post_url
        self.week = self.get_iso_week()
    
    def get_iso_week(self) -> str:
        """Get ISO week format: 2025-W01"""
        year, week, _ = self.post_date.isocalendar()
        return f"{year}-W{week:02d}"

class ForumThread:
    """Data class for forum threads"""
    def __init__(self, thread_title: str, thread_url: str, thread_id: str):
        self.thread_title = thread_title
        self.thread_url = thread_url
        self.thread_id = thread_id
        self.posts: List[ForumPost] = []
        self.total_pages = 0
        self.scraped_pages = 0

def extract_page_from_url(url: str) -> int:
    """Extract page number from URL, default to 1 if not found"""
    match = re.search(r'/page-(\d+)', url)
    return int(match.group(1)) if match else 1

def get_base_thread_url(url: str) -> str:
    """Remove page info to get base thread URL"""
    return re.sub(r'/page-\d+', '', url)

async def extract_thread_info(crawler, thread_url: str, starting_page: int) -> Dict:
    """Extract basic thread information from first page"""
    print(f"🔍 Extracting thread info from: {thread_url}")
    
    try:
        result = await crawler.arun(
            url=thread_url,
            config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
        )
        
        soup = BeautifulSoup(result.html, 'html.parser')
        
        # Extract thread title
        title_element = soup.find('h1', class_='p-title-value')
        thread_title = title_element.get_text(strip=True) if title_element else "Unknown Title"
        
        # Extract thread ID from URL
        thread_id_match = re.search(r'threads/.*\.(\d+)/?', thread_url)
        thread_id = thread_id_match.group(1) if thread_id_match else "unknown"
        
        # Find pagination info to determine total pages  
        total_pages = 1

        # Look for pagination nav with the specific structure you mentioned
        pagination = soup.find('ul', class_='pageNav-main')
        if pagination:
            print(f"🔍 Found pageNav-main pagination")
            
            # Method 1: Find max page from all pageNav-page items
            page_items = pagination.find_all('li', class_='pageNav-page')
            if page_items:
                max_page = 0
                for item in page_items:
                    page_link = item.find('a')
                    if page_link:
                        # Try data-page attribute first
                        page_num_str = page_link.get('data-page')
                        if not page_num_str:
                            # Fallback to link text
                            page_num_str = page_link.get_text(strip=True)
                        
                        try:
                            page_num = int(page_num_str)
                            max_page = max(max_page, page_num)
                        except (ValueError, TypeError):
                            continue
                
                if max_page > 0:
                    total_pages = max_page
                    print(f"📄 Found max page from pageNav-page items: {total_pages}")

        else:
            print("⚠️ No pageNav-main found")
            # If no pagination found but we have a starting page > 1, use that as minimum
            if starting_page > 1:
                total_pages = starting_page + 100  # Estimate
                print(f"📄 No pagination found, estimating {total_pages} pages based on starting page")
        
        print(f"📋 Thread: {thread_title}")
        print(f"🆔 Thread ID: {thread_id}")
        print(f"📄 Total pages: {total_pages}")
        
        return {
            'title': thread_title,
            'thread_id': thread_id,
            'total_pages': total_pages,
            'url': thread_url
        }
        
    except Exception as e:
        print(f"❌ Error extracting thread info: {e}")
        return None

def parse_post_date(date_element) -> Optional[datetime]:
    """Parse post date from time element"""
    if not date_element:
        return None
    
    try:
        # Try to get datetime attribute first
        datetime_attr = date_element.get('datetime', '')
        if datetime_attr:
            # Remove timezone info and parse
            datetime_clean = datetime_attr.replace('Z', '').split('+')[0].split('-')[0]
            if 'T' in datetime_clean:
                return datetime.fromisoformat(datetime_clean)
        
        # Fallback to data-time attribute (timestamp)
        timestamp = date_element.get('data-time', '')
        if timestamp and timestamp.isdigit():
            return datetime.fromtimestamp(int(timestamp))
        
        # Fallback to text content parsing
        date_text = date_element.get_text(strip=True)
        if date_text:
            # Handle various date formats that might appear
            # This might need adjustment based on actual forum date formats
            return datetime.strptime(date_text, "%b %d, %Y")
            
    except Exception as e:
        print(f"⚠️ Date parsing error: {e}")
        return None
    
    return None

def should_include_post(post_date: datetime, start_date_str: str) -> bool:
    """Check if post should be included based on start date"""
    if not post_date:
        return False
    
    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        return post_date >= start_date
    except Exception:
        return True  # Include if we can't parse start date

async def extract_posts_from_page(crawler, page_url: str, start_date: str) -> List[ForumPost]:
    """Extract posts from a single forum page"""
    print(f"📄 Scraping page: {page_url}")
    
    try:
        # Add delay for rate limiting
        await asyncio.sleep(FORUM_CONFIG["rate_limit_delay"])
        
        result = await crawler.arun(
            url=page_url,
            config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
        )
        
        soup = BeautifulSoup(result.html, 'html.parser')
        posts = []
        
        # Find all post containers
        post_containers = soup.find_all('div', class_='message-cell message-cell--main')
        
        for container in post_containers:
            try:
                # Extract post ID from data attributes or nearby elements
                post_id = "unknown"
                parent_article = container.find_parent('article')
                if parent_article:
                    post_id = parent_article.get('data-content', 'unknown')
                
                # Extract username
                username = "unknown"
                user_element = container.find('a', class_='username')
                if user_element:
                    username = user_element.get_text(strip=True)
                
                # Extract post date
                time_element = container.find('time', class_='u-dt')
                post_date = parse_post_date(time_element)
                
                # Skip if date parsing failed or post is too old
                if not post_date or not should_include_post(post_date, start_date):
                    continue
                
                # Extract post content
                content_element = container.find('div', class_='bbWrapper')
                content = ""
                if content_element:
                    # Remove quoted messages (blockquotes) and other noise
                    for quote in content_element.find_all('blockquote'):
                        quote.decompose()
                    for quote in content_element.find_all('div', class_='bbCodeBlock'):
                        quote.decompose()
                    # Remove other common noise elements
                    for element in content_element.find_all(['div', 'span'], class_=['expandSignature', 'js-selectToQuoteEnd']):
                        element.decompose()
                    content = content_element.get_text(strip=True, separator=' ')
                
                # Skip very short posts (likely just reactions)
                if len(content.strip()) < 10:
                    continue
                
                # Create post object (removed username as it's not needed)
                post = ForumPost(
                    post_id=post_id,
                    username="",  # Not needed for analysis
                    post_date=post_date,
                    content=content,
                    post_url=page_url + f"#post-{post_id}"
                )
                
                posts.append(post)
                
                print(f"  ✅ Post on {post_date.strftime('%Y-%m-%d')} - {len(content)} chars")
                
            except Exception as e:
                print(f"  ⚠️ Error processing post: {e}")
                continue
        
        print(f"📊 Extracted {len(posts)} valid posts from page")
        return posts
        
    except Exception as e:
        print(f"❌ Error scraping page {page_url}: {e}")
        return []

async def scrape_forum_thread(thread_url: str, start_date: str = None, manual_total_pages: int = None) -> ForumThread:
    """Main function to scrape a complete forum thread"""
    
    if start_date is None:
        start_date = FORUM_CONFIG["start_date"]
    
    print(f"🚀 Starting forum thread scrape")
    print(f"📅 Start date filter: {start_date}")
    print(f"🔗 Thread URL: {thread_url}")
    
    # Extract starting page from URL
    starting_page = extract_page_from_url(thread_url)
    base_url = get_base_thread_url(thread_url)
    
    print(f"📍 Starting from page {starting_page} (extracted from URL)")
    print(f"🔗 Base thread URL: {base_url}")
    
    async with AsyncWebCrawler(verbose=True) as crawler:
        # Extract thread information using base URL, pass starting page for estimation
        thread_info = await extract_thread_info(crawler, base_url, starting_page)
        if not thread_info:
            print("❌ Failed to extract thread information")
            return None
        
        # Create thread object
        forum_thread = ForumThread(
            thread_title=thread_info['title'],
            thread_url=base_url,
            thread_id=thread_info['thread_id']
        )
        
        # Use manual total pages if provided, otherwise use detected
        if manual_total_pages:
            forum_thread.total_pages = manual_total_pages
            print(f"📄 Using manual total pages: {manual_total_pages}")
        else:
            forum_thread.total_pages = thread_info['total_pages']
        
        # Scrape pages starting from extracted page
        posts_found = 0
        
        # Use detected total pages or manual override
        max_pages = manual_total_pages if manual_total_pages else forum_thread.total_pages
        for page_num in range(starting_page, max_pages + 1):
            # Construct page URL
            if page_num == 1:
                page_url = base_url
            else:
                page_url = f"{base_url}/page-{page_num}"
            
            # Extract posts from page
            page_posts = await extract_posts_from_page(crawler, page_url, start_date)
            forum_thread.posts.extend(page_posts)
            forum_thread.scraped_pages += 1
            posts_found += len(page_posts)
            
            print(f"📈 Progress: Page {page_num}/{thread_info['total_pages']}, Total posts: {posts_found}")
            
            # Early termination for testing
            if TESTING_MODE and posts_found >= MAX_POSTS_FOR_TESTING:
                print(f"🧪 Testing mode: Stopping at {posts_found} posts")
                break
            
            # Safety limit
            if page_num - starting_page > FORUM_CONFIG["max_pages"]:
                print(f"⚠️ Reached maximum page limit ({FORUM_CONFIG['max_pages']})")
                break
        
        print(f"✅ Scraping completed!")
        print(f"📊 Total posts collected: {len(forum_thread.posts)}")
        print(f"📄 Pages scraped: {forum_thread.scraped_pages}")
        
        return forum_thread

def group_posts_by_week(posts: List[ForumPost]) -> Dict[str, List[ForumPost]]:
    """Group posts by ISO week"""
    weekly_groups = {}
    
    for post in posts:
        week = post.week
        if week not in weekly_groups:
            weekly_groups[week] = []
        weekly_groups[week].append(post)
    
    return weekly_groups

def save_raw_data(forum_thread: ForumThread, output_dir: str = "forum_data"):
    """Save raw scraped data to JSON for backup/debugging"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for JSON serialization
    thread_data = {
        'thread_title': forum_thread.thread_title,
        'thread_url': forum_thread.thread_url,
        'thread_id': forum_thread.thread_id,
        'total_pages': forum_thread.total_pages,
        'scraped_pages': forum_thread.scraped_pages,
        'scrape_timestamp': datetime.now().isoformat(),
        'posts': []
    }
    
    for post in forum_thread.posts:
        post_data = {
            'post_id': post.post_id,
            'username': post.username,
            'post_date': post.post_date.isoformat(),
            'content': post.content,
            'post_url': post.post_url,
            'week': post.week
        }
        thread_data['posts'].append(post_data)
    
    # Group posts by week for summary
    weekly_groups = group_posts_by_week(forum_thread.posts)
    thread_data['weekly_summary'] = {
        week: len(posts) for week, posts in weekly_groups.items()
    }
    
    # Save to file
    filename = f"thread_{forum_thread.thread_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(thread_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Raw data saved to: {filepath}")
    return filepath

# Test function
async def test_forum_scraper():
    """Test function for the forum scraper"""
    # Use the URL with starting page you provided
    test_url = "https://forums.hardwarezone.com.sg/threads/which-direction-will-property-prices-go.6382009/page-1228"
    # test_url = "https://forums.hardwarezone.com.sg/threads/free-advice-discussion-on-bank-mortgage-loan.3639010/page-599"
    
    start_date = "2025-01-01"
    manual_total_pages = None  # Since we know the thread has 1337 pages
    
    forum_thread = await scrape_forum_thread(test_url, start_date, manual_total_pages)
    
    if forum_thread:
        # Save raw data
        save_raw_data(forum_thread)
        
        # Print summary
        weekly_groups = group_posts_by_week(forum_thread.posts)
        print(f"\n📊 SUMMARY:")
        print(f"Thread: {forum_thread.thread_title}")
        print(f"Total Posts: {len(forum_thread.posts)}")
        print(f"Weeks with Activity: {len(weekly_groups)}")
        
        for week in sorted(weekly_groups.keys()):
            posts_count = len(weekly_groups[week])
            print(f"  {week}: {posts_count} posts")
    
    return forum_thread

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_forum_scraper())