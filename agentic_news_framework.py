"""
OPTIMIZED Agentic News Analysis Framework using CrewAI
Cleaned version: Scraping → Sequential Analysis → Reporting
"""

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from typing import List, Dict, Any, Union
import pandas as pd
import asyncio
import json
import os
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import your existing modules
from news_scraper_selenium import main as scrape_news
from news_analyzer import (
    analyze_sentiment_ensemble, 
    analyze_topics, 
    analyze_news_impact, 
    analyze_country_region,
    summarize_article,
    create_excel_with_sheets,
    get_latest_news_file,
    _remove_duplicates_optimized,
)

# Configuration from .env file
class AgenticConfig:
    """Configuration management for agentic framework"""
    
    # API Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    FRED_API_KEY = os.getenv('FRED_API_KEY')
    
    # CrewAI Configuration
    CREWAI_MODEL_NAME = os.getenv('CREWAI_MODEL_NAME', 'gpt-4o-mini')
    CREWAI_DISABLE_TELEMETRY = os.getenv('CREWAI_DISABLE_TELEMETRY', 'true').lower() == 'true'
    
    # News Analysis Configuration
    NEWS_DATA_FOLDER = os.getenv('NEWS_DATA_FOLDER', './news_data')
    NEWS_ANALYSIS_OUTPUT_FOLDER = os.getenv('NEWS_ANALYSIS_OUTPUT_FOLDER', './news_analysis_output')
    NEWS_SCRAPING_DELAY = int(os.getenv('NEWS_SCRAPING_DELAY', '10'))
    
    # Framework Configuration
    AGENT_MEMORY_ENABLED = os.getenv('AGENT_MEMORY_ENABLED', 'false').lower() == 'true'
    AGENT_MAX_ITERATIONS = int(os.getenv('AGENT_MAX_ITERATIONS', '2'))
    
    # Testing Configuration - CHANGE TESTING_MODE TO 'false' TO DISABLE 3 ARTICLE LIMIT
    TESTING_MODE = os.getenv('TESTING_MODE', 'false').lower() == 'true'  # SET TO 'false' TO PROCESS ALL ARTICLES
    MAX_ARTICLES_FOR_TESTING = int(os.getenv('MAX_ARTICLES_FOR_TESTING', '3'))
    
    @classmethod
    def validate_config(cls):
        """Validate required configuration"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required in .env file")
        
        # Ensure folder paths are not None and have fallback defaults
        if not cls.NEWS_DATA_FOLDER:
            cls.NEWS_DATA_FOLDER = './news_data'
            
        if not cls.NEWS_ANALYSIS_OUTPUT_FOLDER:
            cls.NEWS_ANALYSIS_OUTPUT_FOLDER = './news_analysis_output'

        # Create required directories
        Path(cls.NEWS_DATA_FOLDER).mkdir(exist_ok=True)
        Path(cls.NEWS_ANALYSIS_OUTPUT_FOLDER).mkdir(exist_ok=True)
        
        # Log the paths being used
        print(f"📁 News data folder: {cls.NEWS_DATA_FOLDER}")
        print(f"📁 Analysis output folder: {cls.NEWS_ANALYSIS_OUTPUT_FOLDER}")

        return True

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Batch Analysis Helper
class BatchAnalysisHelper:
    """Helper class to handle batch analysis of articles from news files"""
    
    # @staticmethod
    # def get_articles_from_latest_file():
    #     """Get articles from the most recent news file - FIXED to handle all sources"""
    #     try:
    #         # Try to find the latest file from any source
    #         latest_file = None
    #         latest_time = 0
            
    #         # Check all possible sources
    #         possible_sources = ["bbc", "yahoo_economic", "yahoo_market"]
            
    #         for source in possible_sources:
    #             try:
    #                 file_path = get_latest_news_file(source)
    #                 file_time = os.path.getmtime(file_path)
    #                 if file_time > latest_time:
    #                     latest_time = file_time
    #                     latest_file = file_path
    #                     logger.info(f"📁 Found latest file: {latest_file}")
    #             except FileNotFoundError:
    #                 logger.info(f"No files found for source: {source}")
    #                 continue
            
    #         if not latest_file:
    #             logger.error("❌ No news files found from any source")
    #             return []
            
    #         # Load news articles
    #         with open(latest_file, 'r', encoding='utf-8') as f:
    #             articles = json.load(f)
            
    #         logger.info(f"🔍 Loaded {len(articles)} articles from {latest_file}")
            
    #         # Apply testing limit if enabled
    #         if AgenticConfig.TESTING_MODE:
    #             original_count = len(articles)
    #             articles = articles[:AgenticConfig.MAX_ARTICLES_FOR_TESTING]
    #             logger.info(f"🧪 TESTING MODE: Limited articles from {original_count} to {len(articles)} for testing")
            
    #         return articles
    #     except Exception as e:
    #         logger.error(f"Error loading articles: {e}")
    #         return []
    
    @staticmethod
    def get_all_articles_from_all_sources():
        """Get NEW articles only - reads from JSON, not master Excel"""
        try:
            # IMPORTANT: Only read from JSON files (newly scraped articles)
            # The master Excel file is handled by create_excel_with_sheets()
            
            all_articles = []
            possible_sources = ["bbc", "yahoo_economic", "yahoo_market", "business_times_sg"]

            for source in possible_sources:
                try:
                    file_path = get_latest_news_file(source)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        articles = json.load(f)

                        # TESTING: Limit each source to MAX_ARTICLES_FOR_TESTING
                        if AgenticConfig.TESTING_MODE:
                            original_count = len(articles)
                            articles = articles[:AgenticConfig.MAX_ARTICLES_FOR_TESTING]
                            logger.info(f"🧪 TESTING MODE: {source} limited from {original_count} to {len(articles)} articles")

                        logger.info(f"Loaded {len(articles)} NEW articles from {source}")
                        all_articles.extend(articles)
                except FileNotFoundError:
                    logger.info(f"No files found for source: {source}")
                    continue
                except Exception as e:
                    logger.error(f"Error reading {source}: {e}")
                    continue
            
            logger.info(f"📊 Combined total: {len(all_articles)} NEW articles from JSON files")
            
            return all_articles
            
        except Exception as e:
            logger.error(f"Error loading articles from all sources: {e}")
            return []

    @staticmethod
    async def analyze_all_articles_sentiment():
        """Analyze sentiment for all articles in the latest file"""
        articles = BatchAnalysisHelper.get_all_articles_from_all_sources()
        results = []
        
        for article in articles:
            try:
                title = article.get("Title", "")
                content = article.get("Content", [])
                score, label = await analyze_sentiment_ensemble(title, content)
                results.append({
                    "title": title,
                    "sentiment_score": score,
                    "sentiment_label": label
                })
            except Exception as e:
                logger.error(f"Error analyzing sentiment for article: {e}")
                continue
        
        return json.dumps(results)
    
    @staticmethod
    async def analyze_all_articles_topics():
        """Analyze topics for all articles in the latest file"""
        articles = BatchAnalysisHelper.get_all_articles_from_all_sources()
        results = []
        
        for article in articles:
            try:
                title = article.get("Title", "")
                content = article.get("Content", [])
                topics = await analyze_topics(title, content)
                results.append({
                    "title": title,
                    "topics": topics
                })
            except Exception as e:
                logger.error(f"Error analyzing topics for article: {e}")
                continue
        
        return json.dumps(results)
    
    @staticmethod
    async def analyze_all_articles_impact():
        """Analyze impact for all articles in the latest file"""
        articles = BatchAnalysisHelper.get_all_articles_from_all_sources()
        results = []
        
        for article in articles:
            try:
                title = article.get("Title", "")
                content = article.get("Content", [])
                impact = await analyze_news_impact(title, content)
                results.append({
                    "title": title,
                    "impact": impact
                })
            except Exception as e:
                logger.error(f"Error analyzing impact for article: {e}")
                continue
        
        return json.dumps(results)
    
    @staticmethod
    async def analyze_all_articles_regions():
        """Analyze regions for all articles in the latest file"""
        articles = BatchAnalysisHelper.get_all_articles_from_all_sources()
        results = []
        
        for article in articles:
            try:
                title = article.get("Title", "")
                content = article.get("Content", [])
                region = await analyze_country_region(title, content)
                results.append({
                    "title": title,
                    "region": region
                })
            except Exception as e:
                logger.error(f"Error analyzing region for article: {e}")
                continue
        
        return json.dumps(results)
    
    @staticmethod
    async def analyze_all_articles_summaries():
        """Summarize all articles in the latest file"""
        articles = BatchAnalysisHelper.get_all_articles_from_all_sources()
        results = []
        
        for article in articles:
            try:
                title = article.get("Title", "")
                content = article.get("Content", [])
                summary = await summarize_article(title, content)
                results.append({
                    "title": title,
                    "summary": summary
                })
            except Exception as e:
                logger.error(f"Error summarizing article: {e}")
                continue
        
        return json.dumps(results)
            
# Custom Tools
class NewsScrapingTool(BaseTool):
    name: str = "news_scraper"
    description: str = "Scrapes news articles from multiple sources (BBC, Yahoo Finance) with built-in relevance filtering"
    
    def _remove_duplicates_immediately(self) -> int:
        """Remove duplicates immediately after scraping, before any analysis"""
        try:
            # Get the master file path
            output_folder = AgenticConfig.NEWS_ANALYSIS_OUTPUT_FOLDER or "./news_analysis_output"
            Path(output_folder).mkdir(exist_ok=True)
            master_file_path = os.path.join(output_folder, "master_news_analysis.xlsx")
            
            if not os.path.exists(master_file_path):
                print("📝 No existing master file found - all articles are new")
                return 0
            
            print(f"🔍 Checking for duplicates against: {master_file_path}")
            
            # Process each source file individually and track results
            possible_sources = ["bbc", "yahoo_economic", "yahoo_market", "business_times_sg"]
            source_results = {}  # Track results by source
            total_removed = 0
            
            for source in possible_sources:
                try:
                    # Get the latest file for this source
                    file_path = get_latest_news_file(source)
                    
                    # Load articles from the JSON file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        articles = json.load(f)
                    
                    original_count = len(articles)
                    print(f"📊 {source}: {original_count} articles scraped")
                    
                    # Remove duplicates against master file
                    unique_articles = _remove_duplicates_optimized(articles, master_file_path)
                    duplicates_removed = original_count - len(unique_articles)
                    unique_count = len(unique_articles)
                    
                    # Store results for this source
                    source_results[source] = {
                        'original': original_count,
                        'unique': unique_count,
                        'duplicates': duplicates_removed
                    }
                    
                    total_removed += duplicates_removed
                    
                    if duplicates_removed > 0:
                        print(f"🔄 {source}: Removed {duplicates_removed} duplicates, {unique_count} unique")
                        
                        # Save the filtered articles back to the JSON file
                        with open(file_path, 'w', encoding='utf-8') as f:
                            json.dump(unique_articles, f, indent=2, ensure_ascii=False)
                    else:
                        print(f"✅ {source}: All articles are unique")
                        
                except FileNotFoundError:
                    print(f"⚠️ No file found for source: {source}")
                    source_results[source] = {
                        'original': 0,
                        'unique': 0,
                        'duplicates': 0
                    }
                    continue
                except Exception as e:
                    print(f"❌ Error processing {source}: {e}")
                    source_results[source] = {
                        'original': 0,
                        'unique': 0,
                        'duplicates': 0
                    }
                    continue
            
            # Display comprehensive summary
            print(f"\n🎯 DUPLICATE CHECK SUMMARY:")
            print(f"   Total duplicates removed: {total_removed}")
            
            # Map source keys to display names
            source_display_names = {
                'bbc': 'BBC',
                'yahoo_economic': 'Yahoo Economic', 
                'yahoo_market': 'Yahoo Market'
            }
            
            # Show distribution of unique articles by source
            print(f"   Unique articles by source:")
            total_unique = 0
            for source, results in source_results.items():
                display_name = source_display_names.get(source, source)
                unique_count = results['unique']
                total_unique += unique_count
                print(f"     {display_name}: {unique_count} unique articles")
            
            print(f"   Total unique articles ready for analysis: {total_unique}")
            
            return total_removed
            
        except Exception as e:
            print(f"❌ Error in duplicate checking: {e}")
            return 0

    def _run(self, source: str = "all") -> str:
        try:
            if source == "all":
                # Scrape all sources
                logger.info("Starting news scraping from all sources")

                sources_to_scrape = ["bbc", "yahoo_economic", "yahoo_market", "business_times_sg"]
                results = []
                total_articles = 0
                
                for src in sources_to_scrape:
                    try:
                        logger.info(f"Scraping {src}...")
                        result_path = asyncio.run(scrape_news(src))
                        
                        # Count articles in the result file
                        try:
                            with open(result_path, 'r', encoding='utf-8') as f:
                                articles = json.load(f)
                                article_count = len(articles)
                                total_articles += article_count
                                results.append(f"{src}: {article_count} articles")
                        except:
                            results.append(f"{src}: completed")
                        
                        # Add delay between sources
                        if AgenticConfig.NEWS_SCRAPING_DELAY > 0:
                            import time
                            time.sleep(AgenticConfig.NEWS_SCRAPING_DELAY)
                            
                    except Exception as e:
                        logger.error(f"Error scraping {src}: {str(e)}")
                        results.append(f"{src}: Error - {str(e)}")
                
                # CRITICAL: Perform duplicate check IMMEDIATELY after scraping
                print(f"\n🔍 PERFORMING IMMEDIATE DUPLICATE CHECK...")
                duplicates_removed = self._remove_duplicates_immediately()

                # Update the results to reflect the actual unique articles
                remaining_articles = total_articles - duplicates_removed

                combined_result = "\n".join(results)
                if duplicates_removed > 0:
                    final_message = f"Multi-source scraping completed:\n{combined_result}\n🔄 Removed {duplicates_removed} duplicates\n✅ {remaining_articles} unique articles ready for analysis"
                else:
                    final_message = f"Multi-source scraping completed:\n{combined_result}\n✅ All {total_articles} articles are unique and ready for analysis"
                logger.info(f"Successfully completed multi-source scraping with duplicate check")
                return final_message
                
            else:
                # Scrape single source
                logger.info(f"Starting news scraping from {source}")
                result = asyncio.run(scrape_news(source))
                
                # Add delay based on configuration
                if AgenticConfig.NEWS_SCRAPING_DELAY > 0:
                    import time
                    time.sleep(AgenticConfig.NEWS_SCRAPING_DELAY)

                # For single source, also check duplicates
                duplicates_removed = self._remove_duplicates_immediately()
                
                if duplicates_removed > 0:
                    final_message = f"Successfully scraped and filtered relevant economic news articles from {source}. Removed {duplicates_removed} duplicates."
                else:
                    final_message = f"Successfully scraped and filtered relevant economic news articles from {source}. All articles are unique."
                
                logger.info(f"Successfully scraped and filtered relevant news articles from {source}")
                return final_message
                
        except Exception as e:
            logger.error(f"Error scraping news from {source}: {str(e)}")
            return f"Error scraping news from {source}: {str(e)}"

class SentimentAnalysisTool(BaseTool):
    name: str = "sentiment_analyzer"
    description: str = "Analyzes sentiment of news articles from the latest file"
    
    def _run(self) -> str:
        try:
            logger.info("Analyzing sentiment for articles")
            result = asyncio.run(BatchAnalysisHelper.analyze_all_articles_sentiment())
            return result
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return f"Error analyzing sentiment: {str(e)}"

class TopicAnalysisTool(BaseTool):
    name: str = "topic_analyzer"
    description: str = "Identifies topics and economic indicators in articles from the latest file"
    
    def _run(self) -> str:
        try:
            logger.info("Analyzing topics and indicators for articles")
            result = asyncio.run(BatchAnalysisHelper.analyze_all_articles_topics())
            return result
        except Exception as e:
            logger.error(f"Error analyzing topics: {str(e)}")
            return f"Error analyzing topics: {str(e)}"

class ImpactAnalysisTool(BaseTool):
    name: str = "impact_analyzer"
    description: str = "Analyzes economic impact on various countries and markets from the latest file"
    
    def _run(self) -> str:
        try:
            logger.info("Analyzing economic impact for articles")
            result = asyncio.run(BatchAnalysisHelper.analyze_all_articles_impact())
            return result
        except Exception as e:
            logger.error(f"Error analyzing impact: {str(e)}")
            return f"Error analyzing impact: {str(e)}"

class CountryRegionAnalysisTool(BaseTool):
    name: str = "country_region_analyzer"
    description: str = "Identifies the primary country/region articles are about from the latest file"
    
    def _run(self) -> str:
        try:
            logger.info("Analyzing country/region for articles")
            result = asyncio.run(BatchAnalysisHelper.analyze_all_articles_regions())
            return result
        except Exception as e:
            logger.error(f"Error analyzing country/region: {str(e)}")
            return f"Error analyzing country/region: {str(e)}"

class ArticleSummarizerTool(BaseTool):
    name: str = "article_summarizer"
    description: str = "Creates concise summaries of news articles from the latest file"
    
    def _run(self) -> str:
        try:
            logger.info("Summarizing articles")
            result = asyncio.run(BatchAnalysisHelper.analyze_all_articles_summaries())
            return result
        except Exception as e:
            logger.error(f"Error summarizing articles: {str(e)}")
            return f"Error summarizing articles: {str(e)}"

class ReportGeneratorTool(BaseTool):
    name: str = "report_generator"
    description: str = "Generates Excel reports with analysis results"
    
    def _run(self) -> str:
        print("🔍 DEBUG: ReportGeneratorTool._run() called!")
        try:
            logger.info("Generating Excel report")
            print(f"🔍 DEBUG: Output folder: {AgenticConfig.NEWS_ANALYSIS_OUTPUT_FOLDER}")
            # Option A: Use latest file (current behavior)
            # articles = BatchAnalysisHelper.get_articles_from_latest_file()
            
            output_folder = AgenticConfig.NEWS_ANALYSIS_OUTPUT_FOLDER or "./news_analysis_output"
            Path(output_folder).mkdir(exist_ok=True)
            print(f"🔍 DEBUG: Output folder: {output_folder}")

            # Get articles from all sources (duplicates already removed by NewsScrapingTool)
            articles = BatchAnalysisHelper.get_all_articles_from_all_sources()
            print(f"🔍 DEBUG: Found {len(articles)} articles")

            if not articles:
                print("❌ DEBUG: No articles found for report generation")
                return "Error: No articles found for report generation"        

            # Get all analysis results
            sentiment_results = json.loads(asyncio.run(BatchAnalysisHelper.analyze_all_articles_sentiment()))
            topic_results = json.loads(asyncio.run(BatchAnalysisHelper.analyze_all_articles_topics()))
            impact_results = json.loads(asyncio.run(BatchAnalysisHelper.analyze_all_articles_impact()))
            region_results = json.loads(asyncio.run(BatchAnalysisHelper.analyze_all_articles_regions()))
            summary_results = json.loads(asyncio.run(BatchAnalysisHelper.analyze_all_articles_summaries()))
            
            # Combine all analysis results with original articles
            comprehensive_data = []
            for i, article in enumerate(articles):
                combined_article = {
                    **article,  # Original article data
                    "Summary": summary_results[i]["summary"] if i < len(summary_results) else "",
                    "Region": region_results[i]["region"] if i < len(region_results) else "",
                    "Impact_Analysis": impact_results[i]["impact"] if i < len(impact_results) else "",
                    "Topic_Analysis": topic_results[i]["topics"] if i < len(topic_results) else "",
                    "Sentiment_Score": sentiment_results[i]["sentiment_score"] if i < len(sentiment_results) else 0,
                    "Sentiment_Label": sentiment_results[i]["sentiment_label"] if i < len(sentiment_results) else "Neutral"
                }
                comprehensive_data.append(combined_article)
            
            # Generate timestamp for output file
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # output_path = f"{AgenticConfig.NEWS_ANALYSIS_OUTPUT_FOLDER}/agentic_analysis_{timestamp}.xlsx"
            
            # Ensure output directory exists
            # Path(AgenticConfig.NEWS_ANALYSIS_OUTPUT_FOLDER).mkdir(exist_ok=True)
            
            # Group articles by source for multi-sheet Excel
            articles_by_source = {}
            for article in comprehensive_data:
                source = article.get("Source", "Unknown")
                if source == "BBC Business":
                    source_key = "BBC Business"
                elif source == "Yahoo Finance-Economic":
                    source_key = "Yahoo Finance-Economic" 
                elif source == "Yahoo Finance-Stock Market":
                    source_key = "Yahoo Finance-Stock Market"
                elif source == "Business Times Singapore":
                    source_key = "Business Times Singapore"
                else:
                    source_key = "Mixed Sources"
                
                if source_key not in articles_by_source:
                    articles_by_source[source_key] = []
                articles_by_source[source_key].append(article)
            
            # Call your existing Excel generation function
            output_path = None
            create_excel_with_sheets(articles_by_source, output_path)
            
            logger.info("Excel report generated: master_news_analysis.xlsx")
            return "Excel report generated successfully: master_news_analysis.xlsx"
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return f"Error generating report: {str(e)}"

# Define Agents
def create_news_agents():
    """Create optimized agents for news analysis workflow"""
    
    logger.info("Creating news analysis agents")
    
    # News Collector
    news_collector = Agent(
        role="News Data Collector",
        goal="Collect relevant economic news articles with built-in filtering",
        backstory="You are a specialized data collection agent that scrapes economic news "
                 "from all news sources and automatically filters for relevance to target regions.",
        tools=[NewsScrapingTool()],
        verbose=True,
        allow_delegation=False,
        max_iter=AgenticConfig.AGENT_MAX_ITERATIONS,
        llm=AgenticConfig.CREWAI_MODEL_NAME,
        memory=False
    )
    
    # Analysis Agents
    sentiment_agent = Agent(
        role="Sentiment Analysis Specialist",
        goal="Analyze sentiment of all collected articles",
        backstory="You are an expert at analyzing market sentiment from economic news articles.",
        tools=[SentimentAnalysisTool()],
        verbose=True,
        allow_delegation=False,
        max_iter=AgenticConfig.AGENT_MAX_ITERATIONS,
        llm=AgenticConfig.CREWAI_MODEL_NAME,
        memory=False
    )
    
    topic_agent = Agent(
        role="Topic Analysis Specialist", 
        goal="Identify economic topics and indicators",
        backstory="You specialize in extracting economic themes and indicators from news content.",
        tools=[TopicAnalysisTool()],
        verbose=True,
        allow_delegation=False,
        max_iter=AgenticConfig.AGENT_MAX_ITERATIONS,
        llm=AgenticConfig.CREWAI_MODEL_NAME,
        memory=False
    )
    
    impact_agent = Agent(
        role="Economic Impact Specialist",
        goal="Analyze economic impact on countries and markets",
        backstory="You are an expert at assessing economic impacts across different regions and markets.",
        tools=[ImpactAnalysisTool()],
        verbose=True,
        allow_delegation=False,
        max_iter=AgenticConfig.AGENT_MAX_ITERATIONS,
        llm=AgenticConfig.CREWAI_MODEL_NAME,
        memory=False
    )
    
    region_agent = Agent(
        role="Regional Analysis Specialist",
        goal="Identify primary country/region focus",
        backstory="You specialize in determining geographical relevance of economic news.",
        tools=[CountryRegionAnalysisTool()],
        verbose=True,
        allow_delegation=False,
        max_iter=AgenticConfig.AGENT_MAX_ITERATIONS,
        llm=AgenticConfig.CREWAI_MODEL_NAME,
        memory=False
    )
    
    summary_agent = Agent(
        role="Article Summarization Specialist",
        goal="Create informative summaries",
        backstory="You are an expert at creating concise, informative summaries of economic news.",
        tools=[ArticleSummarizerTool()],
        verbose=True,
        allow_delegation=False,
        max_iter=AgenticConfig.AGENT_MAX_ITERATIONS,
        llm=AgenticConfig.CREWAI_MODEL_NAME,
        memory=False
    )
    
    # Report Generator
    report_generator = Agent(
        role="Report Generator",
        goal="Generate comprehensive Excel reports",
        backstory="You create well-structured Excel reports with comprehensive analysis results.",
        tools=[ReportGeneratorTool()],
        verbose=False,
        allow_delegation=False,
        max_iter=AgenticConfig.AGENT_MAX_ITERATIONS,
        llm=AgenticConfig.CREWAI_MODEL_NAME,
        memory=False
    )
    
    logger.info("All agents created successfully")
    return (news_collector, sentiment_agent, topic_agent, impact_agent, 
            region_agent, summary_agent, report_generator)

# Define Tasks
def create_news_tasks(agents):
    """Create tasks for the news analysis workflow"""
    
    (news_collector, sentiment_agent, topic_agent, impact_agent, 
     region_agent, summary_agent, report_generator) = agents
    
    # Task 1: News Collection - UPDATED to scrape all sources
    collect_news_task = Task(
        description="Collect and filter relevant economic news articles from ALL sources "
                   "(BBC Business, Yahoo Finance Economic News, Yahoo Finance Stock Market News, Business Times Singapore). "
                   "Use the news scraper with source='all' to scrape from all configured sources. "
                   "The scraping tool includes relevance filtering for economic impact "
                   "on Singapore, US, Euro Area, UK, China, Japan, India, Indonesia, Malaysia, "
                   "Thailand, and Vietnam. Make sure to collect from all four sources.",
        expected_output="Confirmation that relevant economic news articles have been "
                       "successfully scraped from ALL sources (BBC, Yahoo Economic, Yahoo Market, Business Times Singapore), "
                       "filtered, and saved with total article counts from each source",
        agent=news_collector
    )
    
    # Analysis Tasks
    sentiment_analysis_task = Task(
        description="Analyze sentiment of all collected articles. "
                   "Use the sentiment analyzer tool to process all articles automatically.",
        expected_output="Sentiment analysis results for all articles with scores and labels in JSON format",
        agent=sentiment_agent,
        context=[collect_news_task]
    )
    
    topic_analysis_task = Task(
        description="Identify topics and economic indicators in all collected articles. "
                   "Use the topic analyzer tool to process all articles automatically.",
        expected_output="Topic analysis results with identified themes and indicators in JSON format",
        agent=topic_agent,
        context=[collect_news_task]
    )
    
    impact_analysis_task = Task(
        description="Assess economic impact of all collected articles on various countries. "
                   "Use the impact analyzer tool to process all articles automatically.",
        expected_output="Economic impact analysis for different regions and markets in JSON format",
        agent=impact_agent,
        context=[collect_news_task]
    )
    
    region_analysis_task = Task(
        description="Identify primary country/region focus of all collected articles. "
                   "Use the country/region analyzer tool to process all articles automatically.",
        expected_output="Regional classification results for all articles in JSON format",
        agent=region_agent,
        context=[collect_news_task]
    )
    
    summary_task = Task(
        description="Create concise summaries of all collected articles. "
                   "Use the article summarizer tool to process all articles automatically.",
        expected_output="Article summaries highlighting key economic information in JSON format",
        agent=summary_agent,
        context=[collect_news_task]
    )
    
    # Final Report Generation
    generate_report_task = Task(
        description="Generate comprehensive Excel report combining all analysis results. "
                   "Compile sentiment, topic, impact, region, and summary analyses into "
                   "a well-formatted Excel file with multiple sheets and navigation.",
        expected_output="Path to generated Excel file with comprehensive analysis results",
        agent=report_generator,
        context=[sentiment_analysis_task, topic_analysis_task, impact_analysis_task, 
                region_analysis_task, summary_task]
    )
    
    return [
        collect_news_task,
        sentiment_analysis_task,
        topic_analysis_task, 
        impact_analysis_task,
        region_analysis_task,
        summary_task,
        generate_report_task
    ]

# Crew Configuration
class NewsAnalysisCrew:
    """Crew orchestrating the news analysis workflow"""
    
    def __init__(self):
        AgenticConfig.validate_config()
        logger.info("Initializing NewsAnalysisCrew")
        self.agents = create_news_agents()
        self.tasks = create_news_tasks(self.agents)
        
    def create_crew(self):
        """Create and configure the crew"""
        logger.info("Creating crew with sequential process")
        
        crew_config = {
            "agents": self.agents,
            "tasks": self.tasks,
            "process": Process.sequential,
            "verbose": True, 
            "memory": False
        }
        
        return Crew(**crew_config)
    
    def run_analysis(self):
        """Execute the news analysis workflow"""
        crew = self.create_crew()
        
        logger.info("🚀 Starting Agentic News Analysis...")
        print("🚀 Starting Agentic News Analysis...")
        print("=" * 50)
        print(f"📋 Configuration:")
        print(f"   • Model: {AgenticConfig.CREWAI_MODEL_NAME}")
        print(f"   • Execution Mode: Sequential")
        print(f"   • Memory Enabled: {AgenticConfig.AGENT_MEMORY_ENABLED}")
        print(f"   • Max Iterations: {AgenticConfig.AGENT_MAX_ITERATIONS}")
        print(f"   • Output Folder: {AgenticConfig.NEWS_ANALYSIS_OUTPUT_FOLDER}")
        print(f"   • Scraping Delay: {AgenticConfig.NEWS_SCRAPING_DELAY}s")
        if AgenticConfig.TESTING_MODE:
            print(f"   🧪 TESTING MODE: Limited to {AgenticConfig.MAX_ARTICLES_FOR_TESTING} articles")
        else:
            print(f"   🚀 PRODUCTION MODE: Processing ALL scraped articles")
        print("=" * 50)
        
        try:
            import time
            start_time = time.time()
            
            # Execute the crew
            result = crew.kickoff()
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            logger.info(f"✅ Analysis completed in {execution_time:.1f} seconds!")
            print(f"\n✅ Analysis completed in {execution_time:.1f} seconds!")
            print(f"📊 Result: {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in analysis: {e}")
            print(f"❌ Error in analysis: {e}")
            raise

# Usage Example
if __name__ == "__main__":
    try:
        AgenticConfig.validate_config()
        print("✅ Configuration validated successfully")
        print(f"📋 Using model: {AgenticConfig.CREWAI_MODEL_NAME}")
        print(f"📁 Output folder: {AgenticConfig.NEWS_ANALYSIS_OUTPUT_FOLDER}")
        
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("Please check your .env file and ensure all required variables are set.")
        exit(1)
    
    try:
        crew = NewsAnalysisCrew()
        result = crew.run_analysis()
        print("\n🎉 Agentic news analysis completed!")
        
    except Exception as e:
        logger.error(f"Failed to run news analysis: {e}")
        print(f"❌ Failed to run news analysis: {e}")
        raise