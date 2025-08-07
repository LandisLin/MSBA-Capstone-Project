import asyncio
import json
import os
import time
from datetime import datetime
from urllib.parse import urljoin

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, VirtualScrollConfig
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy

from openai import AsyncOpenAI
from bs4 import BeautifulSoup
from dotenv import load_dotenv


# Configuration
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Testing configuration - import from environment
TESTING_MODE = os.getenv('TESTING_MODE', 'false').lower() == 'true'
MAX_ARTICLES_TO_CHECK = int(os.getenv('MAX_ARTICLES_FOR_TESTING', '3'))

# Universal news source configurations
NEWS_SOURCES = {
    "bbc": {
        "name": "BBC Business",
        "base_url": "https://www.bbc.com/business",
        "website_base": "https://www.bbc.com",
        "extraction_strategy": {
            "name": "BBC Business Index",
            "baseSelector": 'div[data-testid="anchor-inner-wrapper"]',
            "fields": [
                {"name": "Title", "selector": 'h2[data-testid="card-headline"]', "type": "text"},
                {"name": "Description", "selector": 'p[data-testid="card-description"]', "type": "text"},
                {"name": "Section", "selector": 'span[data-testid="card-metadata-tag"]', "type": "text"},
                {"name": "Link", "selector": 'a[data-testid="internal-link"]', "type": "attribute", "attribute": "href"}
            ]
        },
        "content_extraction": {
            "text_selector": 'div[data-component="text-block"]',
            "paragraph_selector": "p",
            "time_selector": "time",
            "datetime_attribute": "datetime"
        },
        "skip_links": ["/audio", "/video", "/reel"],
        "anti_bot_delay": 10
    },
    "yahoo_economic": {
        "name": "Yahoo Finance-Economic",
        "base_url": "https://finance.yahoo.com/topic/economic-news/",
        "website_base": "https://finance.yahoo.com",
        "extraction_strategy": {
            "name": "Yahoo Economic News",
            "baseSelector": 'div.topic-stream li',
            "fields": [
                {"name": "Title", "selector": 'a.subtle-link', "type": "text"},
                {"name": "Link", "selector": 'a.subtle-link', "type": "attribute", "attribute": "href"},
                {"name": "AlternativeTitle", "selector": 'h3', "type": "text"},
                {"name": "AlternativeLink", "selector": 'a[href*="/news/"]', "type": "attribute", "attribute": "href"}
            ]
        },
        "content_extraction": {
            "text_selector": 'div.body',
            "paragraph_selector": "p",
            "time_selector": "time",
            "datetime_attribute": "datetime"
        },
        "skip_links": [],
        "anti_bot_delay": 10
    },
    "yahoo_market": {
        "name": "Yahoo Finance-Stock Market",
        "base_url": "https://finance.yahoo.com/topic/stock-market-news/",
        "website_base": "https://finance.yahoo.com",
        "extraction_strategy": {
            "name": "Yahoo Stock Market News",
            "baseSelector": 'div.topic-stream li',
            "fields": [
                {"name": "Title", "selector": 'a.subtle-link', "type": "text"},
                {"name": "Link", "selector": 'a.subtle-link', "type": "attribute", "attribute": "href"},
                {"name": "AlternativeTitle", "selector": 'h3', "type": "text"},
                {"name": "AlternativeLink", "selector": 'a[href*="/news/"]', "type": "attribute", "attribute": "href"}
            ]
        },
        "content_extraction": {
            "text_selector": 'div.body',
            "paragraph_selector": "p",
            "time_selector": "time",
            "datetime_attribute": "datetime"
        },
        "skip_links": [],
        "anti_bot_delay": 10
    },

    "business_times_sg": {
        "name": "Business Times Singapore",
        "base_url": "https://www.businesstimes.com.sg/singapore?ref=header",
        "website_base": "https://www.businesstimes.com.sg",
        "extraction_strategy": {
            "name": "Business Times Singapore",
            "baseSelector": 'div.stories div[data-testid="basic-card"]',
            "fields": [
                {"name": "Title", "selector": 'h3[data-testid*="title_link"]', "type": "text"},
                {"name": "Link", "selector": 'h3[data-testid*="title_link"] a', "type": "attribute", "attribute": "href"},
                {"name": "Section", "selector": '.section, .category, .topic', "type": "text"}
            ]
        },
        "content_extraction": {
            "text_selector": '.story-content, .article-content, .content, .body-content',
            "paragraph_selector": "p",
            "time_selector": "time, .publish-date, .date",
            "datetime_attribute": "datetime"
        },
        "skip_links": ["/video", "/audio", "/gallery"],
        "anti_bot_delay": 8,
        "selenium_scroll": {
            "times": 15,
            "wait": 2.0
        }
    },
}

# ——————————————
# Selenium Scroller
# ——————————————
def get_scrolled_html_with_selenium(url: str, scroll_times: int = 15, wait: float = 5.0) -> str:
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=options)
    try:
        driver.get(url)
        for i in range(scroll_times):
            driver.execute_script("window.scrollBy(0, 1000);")
            print(f"🌀 Scrolled {i + 1}/{scroll_times}")
            time.sleep(wait)
        html = driver.page_source
        print("✅ Completed scrolling with Selenium")
        return html
    finally:
        driver.quit()

# ——————————————
# Relevance & Content Helpers
# ——————————————

async def is_article_relevant_to_economy(title: str, text: str) -> bool:
    """Universal relevance checker for all news sources"""
    prompt = (
        "You are a financial news classifier. Given a news article, determine whether it is relevant to or impactful on the economy "
        "or financial markets of any of the following regions: Singapore, United States (US), Euro Area, United Kingdom (UK), China, "
        "Japan, India, Indonesia, Malaysia, Thailand, or Vietnam.\n\n"
        "Relevance includes direct economic topics like GDP, inflation, interest rates, employment, or central bank policy, "
        "as well as indirect but impactful events like major political decisions, wars, trade tensions, major regulations, "
        "or large-scale corporate changes that could affect investor sentiment or economic stability in the listed regions.\n\n"
        "Reply **only** with 'Yes' if the article is **explicitly or implicitly** relevant to the economy or financial markets of **any** of those regions. "
        "Otherwise, reply 'No'.\n\n"
        f"Title: {title.strip()}\n\n"
        f"Content:\n{text[:1000]}\n\n"
        "Is this news article relevant to or impactful on the listed regions' economy or financial markets?"
    )

    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0
        )
        answer = response.choices[0].message.content.strip().lower()
        return answer.startswith("yes")
    except Exception as e:
        print(f"⚠️ LLM relevance check failed: {e}")
        return False

def format_datetime(datetime_str: str) -> str:
    """Convert ISO datetime string to readable format"""
    if not datetime_str:
        return ""
    
    try:
        if 'T' in datetime_str:
            dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
        else:
            dt = datetime.fromisoformat(datetime_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return datetime_str

async def fetch_article_content_bbc(crawler, relative_url: str, source_config: dict):
    """Extract content specifically for BBC articles"""
    website_base = source_config["website_base"]
    full_url = website_base + relative_url
    
    try:
        result = await crawler.arun(
            url=full_url,
            config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
        )
        
        soup = BeautifulSoup(result.html, 'html.parser')
        
        # BBC-specific content extraction
        text_blocks = soup.find_all('div', {'data-component': 'text-block'})
        
        all_paragraphs = []
        for text_block in text_blocks:
            paragraphs = text_block.find_all('p')
            for p in paragraphs:
                text = p.get_text(strip=True)
                if text:
                    all_paragraphs.append(text)
        
        # Get published date
        published = ""
        time_element = soup.find('time')
        if time_element:
            datetime_attr = time_element.get('datetime', '')
            published = format_datetime(datetime_attr)
        
        # Remove duplicates while preserving order
        unique_paragraphs = []
        seen = set()
        for p in all_paragraphs:
            if p not in seen:
                unique_paragraphs.append(p)
                seen.add(p)
        
        print(f"📝 Extracted {len(unique_paragraphs)} paragraphs")
        print(f"📅 Published: {published}")
        
        return {"Paragraphs": unique_paragraphs, "Published": published}

    except Exception as e:
        print(f"❌ Error fetching {full_url}: {e}")
        return {"Paragraphs": [], "Published": ""}

async def fetch_article_content_yahoo(crawler, article_url: str, source_config: dict):
    """Extract content specifically for Yahoo Finance articles"""
    try:
        # Yahoo URLs are typically complete, but ensure we have the full URL
        if not article_url.startswith('http'):
            full_url = urljoin(source_config["website_base"], article_url)
        else:
            full_url = article_url
            
        result = await crawler.arun(
            url=full_url,
            config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
        )
        
        soup = BeautifulSoup(result.html, 'html.parser')
        
        # Yahoo-specific content extraction
        # Try multiple selectors for Yahoo content
        content_selectors = [
            'div.body',
            'div.caas-body',
            'div[data-module="ArticleBody"]',
            'div.content'
        ]
        
        all_paragraphs = []
        for selector in content_selectors:
            content_div = soup.select_one(selector)
            if content_div:
                paragraphs = content_div.find_all('p')
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if text and len(text) > 20:  # Filter out very short paragraphs
                        all_paragraphs.append(text)
                break
        
        # Get published date
        published = ""
        time_selectors = ['time', '[datetime]', '.publish-time']
        for selector in time_selectors:
            time_element = soup.select_one(selector)
            if time_element:
                datetime_attr = time_element.get('datetime', '') or time_element.get_text(strip=True)
                published = format_datetime(datetime_attr) if datetime_attr else ""
                break
        
        # Remove duplicates while preserving order
        unique_paragraphs = []
        seen = set()
        for p in all_paragraphs:
            if p not in seen:
                unique_paragraphs.append(p)
                seen.add(p)
        
        print(f"📝 Extracted {len(unique_paragraphs)} paragraphs")
        print(f"📅 Published: {published}")
        
        return {"Paragraphs": unique_paragraphs, "Published": published}

    except Exception as e:
        print(f"❌ Error fetching {article_url}: {e}")
        return {"Paragraphs": [], "Published": ""}

async def fetch_article_content_business_times(crawler, article_url: str, source_config: dict):
    """Extract content specifically for Business Times Singapore articles with dynamic loading"""
    try:
        # Ensure we have the full URL
        if not article_url.startswith('http'):
            full_url = urljoin(source_config["website_base"], article_url)
        else:
            full_url = article_url
            
        # Dynamic content loading
        result = await crawler.arun(
            url=full_url,
            config=CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                delay_before_return_html=4,  # Reduced wait time
                js_code=[
                    "await new Promise(resolve => setTimeout(resolve, 1500));",
                    "window.scrollTo(0, document.body.scrollHeight);",
                    "await new Promise(resolve => setTimeout(resolve, 1000));",
                    "window.scrollTo(0, 0);"
                ]
            )
        )
        
        soup = BeautifulSoup(result.html, 'html.parser')
        all_paragraphs = []
        
        # DEBUG: Check what we got
        print(f"🔍 DEBUG: HTML length: {len(result.html)}")
        paragraph_component_count = result.html.count('paragraph-component')
        print(f"🔍 DEBUG: paragraph-component count: {paragraph_component_count}")
        
        # METHOD 1: Primary method - data-testid="paragraph-component"
        article_main = soup.find("div", attrs={"data-testid": "article-paragraphs-component"})
        if not article_main:
            print("⚠️ Main article content not found")
            return []

        paragraph_components = article_main.find_all(attrs={"data-testid": "paragraph-component"})
        print(f"📍 Found {len(paragraph_components)} paragraph-component elements")
        
        for i, element in enumerate(paragraph_components):
            text = element.get_text(strip=True)
            if text and len(text) > 10:
                all_paragraphs.append(text)
                print(f"📄 Added paragraph {i+1}: {text[:100]}...")
        
        # # METHOD 2: Fallback to data-story-element
        # if len(all_paragraphs) < 5:
        #     print(f"⚠️ Only found {len(all_paragraphs)} from paragraph-component, trying data-story-element...")
        #     story_elements = soup.find_all(attrs={"data-story-element": "paragraph"})
            
        #     for element in story_elements:
        #         text = element.get_text(strip=True)
        #         if text and len(text) > 10 and text not in all_paragraphs:
        #             all_paragraphs.append(text)
        #             print(f"📄 Added from story-element: {text[:100]}...")
        
        # QUALITY CHECK: Return None if insufficient content
        if len(all_paragraphs) < 5:
            print(f"❌ Only extracted {len(all_paragraphs)} paragraphs - insufficient content, skipping article")
            return None  # Signal to skip this article
        
        # Get published date
        published = ""
        script_tags = soup.find_all('script')
        for script in script_tags:
            if script.string and 'pubdate' in script.string:
                script_content = script.string
                import re
                pubdate_match = re.search(r'"pubdate"\s*:\s*"([^"]+)"', script_content)
                if pubdate_match:
                    published = pubdate_match.group(1)
                    break
        
        # Get section
        section = ""
        section_element = soup.find(attrs={"data-testid": "button-link-component"})
        if section_element:
            section = section_element.get_text(strip=True)
        
        # Clean up and deduplicate
        unique_paragraphs = []
        seen = set()
        
        for p in all_paragraphs:
            clean_text = p.strip()
            if (clean_text and clean_text not in seen and len(clean_text) > 25):
                unique_paragraphs.append(clean_text)
                seen.add(clean_text)
        
        print(f"📝 Final result: {len(unique_paragraphs)} paragraphs extracted ✅")
        
        return {"Paragraphs": unique_paragraphs, "Published": published, "Section": section}

    except Exception as e:
        print(f"❌ Error fetching Business Times content from {article_url}: {e}")
        return None

async def fetch_article_content(crawler, link: str, source: str, source_config: dict):
    """Universal content fetcher that routes to appropriate handler"""
    if source == "bbc":
        return await fetch_article_content_bbc(crawler, link, source_config)
    elif source == "business_times_sg":
        return await fetch_article_content_business_times(crawler, link, source_config)
    elif source.startswith("yahoo"):
        return await fetch_article_content_yahoo(crawler, link, source_config)
    else:
        # Default generic extractor
        return await fetch_article_content_yahoo(crawler, link, source_config)

# ——————————————
# Main Scraper Function
# ——————————————

async def scrape_news_source(source: str = "bbc"):
    if source not in NEWS_SOURCES:
        raise ValueError(f"Unsupported source: {source}")

    config = NEWS_SOURCES[source]
    print(f"🚀 Starting scrape for {config['name']}")

    async with AsyncWebCrawler() as crawler:
        strategy = JsonCssExtractionStrategy(config["extraction_strategy"])
        crawler_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            extraction_strategy=strategy
        )

        # Use Selenium scroll if configured for business_times_sg
        if source == "business_times_sg":
            scroll_cfg = config.get("selenium_scroll", {})
            print(f"🔄 Using Selenium to scroll {scroll_cfg.get('times', 10)} times...")
            html = get_scrolled_html_with_selenium(
                config["base_url"],
                scroll_times=scroll_cfg.get("times", 10),
                wait=scroll_cfg.get("wait", 2.0)
            )
            result = await crawler.arun(
                # url=f"raw:{html}",
                url=f"raw:{html}",
                config=crawler_config
            )
        else:
            # Falls back to default crawling behavior
            if "virtual_scroll_config" in config:
                scroll = config["virtual_scroll_config"]
                crawler_config.virtual_scroll_config = VirtualScrollConfig(
                    container_selector=scroll.get("container_selector", "body"),
                    scroll_count=scroll.get("scroll_count", 5),
                    scroll_by=scroll.get("scroll_by", "container_height"),
                    wait_after_scroll=scroll.get("wait_after_scroll", 2.0)
                )
            result = await crawler.arun(url=config["base_url"], config=crawler_config)

        articles = json.loads(result.extracted_content)
        print(f"📰 Found {len(articles)} article listings")
        
        if source == "business_times_sg" and TESTING_MODE:
            print(f"\n🔍 DEBUGGING: First 25 articles found:")
            for i, article in enumerate(articles[:25]):
                title = article.get("Title", "") or article.get("AlternativeTitle", "")
                link = article.get("Link", "") or article.get("AlternativeLink", "")
                print(f"  {i+1:2d}. {title[:70]}...")
            print(f"{'='*50}\n")

        seen_links = set()
        relevant_articles = []
        article_count = 0
        checked_count = 0

        for article in articles:
            # Handle different link structures
            link = article.get("Link", "") or article.get("AlternativeLink", "")
            title = article.get("Title", "") or article.get("AlternativeTitle", "")
            
            # Skip invalid articles
            if not link or not title:
                print(f"⏭️ Skipping invalid article (missing link or title)")
                continue

            # FIXED: Properly normalize links (handle relative URLs)
            if link.startswith('http'):
                # Already a full URL
                normalized_link = link
            else:
                # Relative URL - convert to full URL first
                normalized_link = urljoin(config["website_base"], link)
            
            # Remove query parameters, fragments, and trailing slashes for comparison
            import urllib.parse
            parsed_link = urllib.parse.urlparse(normalized_link)
            final_normalized_link = f"{parsed_link.scheme}://{parsed_link.netloc}{parsed_link.path.rstrip('/')}"
            
            # Check for duplicates using normalized link
            if final_normalized_link in seen_links:
                print(f"⏭️ Skipping duplicate link (normalized): {title[:50]}...")
                continue
                
            # Skip specified link types
            if any(link.startswith(prefix) for prefix in config.get("skip_links", [])):
                print(f"⏭️ Skipping excluded link type: {link}")
                continue
            
            # TESTING MODE: Stop checking after MAX_ARTICLES_TO_CHECK
            if TESTING_MODE and checked_count >= MAX_ARTICLES_TO_CHECK:
                print(f"🧪 TESTING MODE: Reached limit of {MAX_ARTICLES_TO_CHECK} articles to check, stopping")
                break
                
            checked_count += 1
            print(f"\n🔍 [{checked_count}] Checking article: {title}")

            print(f"🔗 Raw link found: '{link}'")
            # print(f"📰 Title: '{title}'")
            print(f"🔧 Normalized link: '{final_normalized_link}'")

            seen_links.add(final_normalized_link)  # Store normalized link
            article_count += 1
            
            print(f"✅ Added to processing queue. Total unique links: {len(seen_links)}")
            
            # Fetch full article content
            article_data = await fetch_article_content(crawler, link, source, config)

            # ADDED: Skip articles with insufficient content
            if article_data is None:
                print("⏭️ Skipping article due to insufficient content or extraction error")
                continue
            
            full_text = " ".join(article_data["Paragraphs"])

            # Check relevance using LLM
            if await is_article_relevant_to_economy(title, full_text):
                print("✅ Article is relevant to economy or financial markets.")
                
                # Build full URL for the link
                if source == "bbc":
                    full_link = config["website_base"] + link
                elif source == "business_times_sg":
                    # Handle Business Times Singapore URLs
                    if link.startswith('http'):
                        full_link = link
                    else:
                        full_link = urljoin(config["website_base"], link)
                elif link.startswith('http'):
                    full_link = link
                else:
                    full_link = urljoin(config["website_base"], link)
                
                article_section = article_data.get("Section", "") or article.get("Section", "") or config["name"]
                article_entry = {
                    "Title": title,
                    "Description": article.get("Description", ""),
                    "Section": article_section,
                    "Published": article_data["Published"],
                    "URL": full_link,
                    "Content": article_data["Paragraphs"],
                    "Source": config["name"]
                }
                relevant_articles.append(article_entry)
            else:
                print("⛔ Not relevant. Skipping.")
            
            # Anti-bot delay
            await asyncio.sleep(config.get("anti_bot_delay", 5))

        # Show results
        if TESTING_MODE:
            print(f"\n📊 TESTING MODE: Checked {checked_count} articles, found {len(relevant_articles)} relevant articles.")
        else:
            print(f"\n📊 Found {len(relevant_articles)} relevant articles out of {checked_count} checked.")
        
        # Save results
        os.makedirs("./news_data", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"./news_data/{source}_news_{timestamp}.json"
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(relevant_articles, f, ensure_ascii=False, indent=2)
        
        print(f"🗂 Saved {len(relevant_articles)} articles to {output_path}")
        return output_path
    
async def main(source: str = "bbc"):
    """Main function - can be called with different sources"""
    try:
        return await scrape_news_source(source)
    except Exception as e:
        print(f"❌ Error in main scraping function: {e}")
        raise

# Convenience functions for specific sources
async def scrape_bbc_news():
    """Scrape BBC Business news"""
    return await main("bbc")

async def scrape_yahoo_economic_news():
    """Scrape Yahoo Finance Economic news"""
    return await main("yahoo_economic")

async def scrape_yahoo_market_news():
    """Scrape Yahoo Finance Stock Market news"""
    return await main("yahoo_market")

async def scrape_business_times_singapore():
    """Scrape Business Times Singapore news with infinite scroll support"""
    return await main("business_times_sg")

async def scrape_all_sources():
    """Scrape all configured news sources"""
    results = {}
    for source in NEWS_SOURCES.keys():
        try:
            print(f"\n{'='*50}")
            print(f"🔄 Processing {NEWS_SOURCES[source]['name']}")
            print(f"{'='*50}")
            result = await main(source)
            results[source] = result
            print(f"✅ {NEWS_SOURCES[source]['name']} completed successfully!")
        except Exception as e:
            print(f"❌ Failed to process {NEWS_SOURCES[source]['name']}: {e}")
            results[source] = None
    
    print(f"\n🎉 Scraping completed for all sources!")
    return results

if __name__ == "__main__":
    # Example usage:
    # asyncio.run(main("bbc"))                    # Scrape BBC only
    # asyncio.run(main("yahoo_economic"))         # Scrape Yahoo Economic only
    # asyncio.run(main("yahoo_market"))           # Scrape Yahoo Market only
    asyncio.run(main("business_times_sg"))       # Scrape Business Times Singapore only
    
    # Test all sources:
    # asyncio.run(scrape_all_sources())            # Scrape all sources