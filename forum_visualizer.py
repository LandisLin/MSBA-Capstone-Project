"""
Forum Analysis Visualizer
Interactive visualizations for forum discussion analysis with word clouds, timeline, and expandable cards
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import re
from collections import Counter
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

class ForumVisualizer:
    """Forum analysis visualization class"""
    
    def __init__(self):
        self.forum_data = {}
        self.load_forum_data()
    
    def load_forum_data(self):
        """Load forum analysis data from Excel file"""
        try:
            # Try to load master forum analysis file from forum_analysis_output folder
            file_path = Path("forum_analysis_output/master_forum_analysis.xlsx")
            if not file_path.exists():
                # Fallback: try root directory
                file_path = Path("master_forum_analysis.xlsx")
                if not file_path.exists():
                    print("❌ Forum analysis file not found in forum_analysis_output/ or root directory")
                    return
            
            # Load all sheets from the Excel file
            xl_file = pd.ExcelFile(file_path)
            
            # Load Contents sheet for thread information
            if 'Contents' in xl_file.sheet_names:
                contents_df = pd.read_excel(file_path, sheet_name='Contents')
                print(f"📋 Found {len(contents_df)} threads in Contents")
            
            for sheet_name in xl_file.sheet_names:
                if sheet_name.startswith('Thread_') and sheet_name.endswith('_Weekly'):
                    # Extract thread info from sheet name (e.g., Thread_6382009_Weekly -> 6382009)
                    thread_id = sheet_name.replace('Thread_', '').replace('_Weekly', '')
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    
                    # Ensure required columns exist
                    required_cols = ['Week', 'Post_Count', 'Summary', 'Topics', 'Sentiment_Score', 'Sentiment_Label']
                    if all(col in df.columns for col in required_cols):
                        # Clean and process the topics column - but don't store as list in DataFrame
                        # Keep original Topics column and add processed version as needed
                        self.forum_data[thread_id] = df.copy()
                        print(f"✅ Loaded {len(df)} weeks of data for Thread {thread_id}")
                    else:
                        print(f"⚠️  Missing required columns in {sheet_name}")
                        print(f"Available columns: {list(df.columns)}")
            
            print(f"📊 Total threads loaded: {len(self.forum_data)}")
            
        except Exception as e:
            print(f"❌ Error loading forum data: {e}")
    
    def _process_topics_string(self, topics_str):
        """Process topics string from [Topic1, Topic2] format to clean list"""
        if pd.isna(topics_str) or not topics_str:
            return []
        
        try:
            # Convert string representation of array to actual list
            topics_str = str(topics_str).strip()
            
            if topics_str.startswith('[') and topics_str.endswith(']'):
                # Remove brackets and split by comma
                topics_str = topics_str[1:-1]
                # Split by comma and clean each topic
                topics = []
                for topic in topics_str.split(','):
                    topic = topic.strip().strip('"').strip("'").strip()
                    if topic:  # Only add non-empty topics
                        topics.append(topic)
                return topics
            else:
                # Fallback: split by comma
                topics = []
                for topic in topics_str.split(','):
                    topic = topic.strip().strip('"').strip("'").strip()
                    if topic:  # Only add non-empty topics
                        topics.append(topic)
                return topics
        except Exception as e:
            print(f"Error processing topics: {topics_str} -> {e}")
            return [str(topics_str)] if topics_str else []
    
    def get_available_threads(self) -> List[str]:
        """Get list of available forum threads"""
        return list(self.forum_data.keys())
    
    def get_thread_data(self, thread_name: str) -> pd.DataFrame:
        """Get data for specific thread"""
        return self.forum_data.get(thread_name, pd.DataFrame())
    
    def get_combined_data(self) -> pd.DataFrame:
        """Combine all thread data for cross-thread analysis"""
        all_data = []
        for thread_name, df in self.forum_data.items():
            df_copy = df.copy()
            df_copy['Thread'] = thread_name
            df_copy['Thread_Title'] = f"Thread {thread_name}"  # You can enhance this with actual titles
            all_data.append(df_copy)
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()
    
    def extract_topics_for_wordcloud(self, thread_name: str = None, date_range: tuple = None) -> Dict[str, int]:
        """Extract topics for word cloud generation"""
        if thread_name:
            df = self.get_thread_data(thread_name)
        else:
            df = self.get_combined_data()
        
        if df.empty:
            return {}
        
        # Filter by date range if provided
        if date_range and 'Week' in df.columns:
            start_week, end_week = date_range
            df = df[(df['Week'] >= start_week) & (df['Week'] <= end_week)]
        
        # Extract all topics by processing the Topics column on-the-fly
        all_topics = []
        for topics_str in df['Topics'].fillna(''):
            if topics_str:
                processed_topics = self._process_topics_string(topics_str)
                all_topics.extend(processed_topics)
        
        # Count topic frequencies
        topic_counts = Counter(all_topics)
        # Remove empty strings and clean up
        topic_counts = {k.strip(): v for k, v in topic_counts.items() if k.strip()}
        
        return topic_counts
    
    def create_sentiment_timeline(self, thread_name: str = None) -> go.Figure:
        """Create interactive sentiment timeline with week selection"""
        if thread_name:
            df = self.get_thread_data(thread_name)
            title = f"📈 Sentiment Timeline - {thread_name}"
        else:
            df = self.get_combined_data()
            title = "📈 Combined Sentiment Timeline - All Threads"
        
        if df.empty:
            return None
        
        # Ensure numeric sentiment scores
        df['Sentiment_Score'] = pd.to_numeric(df['Sentiment_Score'], errors='coerce')
        
        # Create figure
        fig = go.Figure()
        
        if thread_name:
            # Single thread timeline
            fig.add_trace(go.Scatter(
                x=df['Week'],
                y=df['Sentiment_Score'],
                mode='lines+markers',
                name=thread_name,
                line=dict(width=3, color='#1f77b4'),
                marker=dict(size=8, color=df['Sentiment_Score'], 
                          colorscale='RdYlGn', showscale=True,
                          colorbar=dict(title="Sentiment Score")),
                hovertemplate=(
                    "<b>Week:</b> %{x}<br>"
                    "<b>Sentiment:</b> %{y:.3f}<br>"
                    "<b>Posts:</b> %{customdata[0]}<br>"
                    "<b>Topics:</b> %{customdata[1]}<br>"
                    "<extra></extra>"
                ),
                customdata=df[['Post_Count', 'Topics']].values
            ))
        else:
            # Multi-thread timeline
            for thread in df['Thread'].unique():
                thread_df = df[df['Thread'] == thread]
                fig.add_trace(go.Scatter(
                    x=thread_df['Week'],
                    y=thread_df['Sentiment_Score'],
                    mode='lines+markers',
                    name=thread,
                    line=dict(width=2),
                    marker=dict(size=6),
                    hovertemplate=(
                        f"<b>{thread}</b><br>"
                        "<b>Week:</b> %{x}<br>"
                        "<b>Sentiment:</b> %{y:.3f}<br>"
                        "<extra></extra>"
                    )
                ))
        
        # Add horizontal reference lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_hline(y=0.2, line_dash="dot", line_color="green", opacity=0.3)
        fig.add_hline(y=-0.2, line_dash="dot", line_color="red", opacity=0.3)
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Week",
            yaxis_title="Sentiment Score",
            height=500,
            showlegend=True,
            hovermode='x unified',
            yaxis=dict(range=[-1, 1]),
            template='plotly_white'
        )
        
        return fig
    
    def create_post_volume_chart(self, thread_name: str = None) -> go.Figure:
        """Create post volume chart with sentiment overlay"""
        if thread_name:
            df = self.get_thread_data(thread_name)
            title = f"📊 Post Volume & Sentiment - {thread_name}"
        else:
            df = self.get_combined_data()
            if not df.empty:
                # Aggregate by week for combined view
                df = df.groupby('Week').agg({
                    'Post_Count': 'sum',
                    'Sentiment_Score': 'mean'
                }).reset_index()
            title = "📊 Combined Post Volume & Sentiment"
        
        if df.empty:
            return None
        
        # Create subplot with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add post volume bars
        fig.add_trace(
            go.Bar(
                x=df['Week'],
                y=df['Post_Count'],
                name="Post Volume",
                marker_color='lightblue',
                opacity=0.7,
                hovertemplate=(
                    "<b>Week:</b> %{x}<br>"
                    "<b>Posts:</b> %{y}<br>"
                    "<extra></extra>"
                )
            ),
            secondary_y=False,
        )
        
        # Add sentiment line
        fig.add_trace(
            go.Scatter(
                x=df['Week'],
                y=df['Sentiment_Score'],
                mode='lines+markers',
                name="Sentiment",
                line=dict(color='red', width=3),
                marker=dict(size=8),
                hovertemplate=(
                    "<b>Week:</b> %{x}<br>"
                    "<b>Sentiment:</b> %{y:.3f}<br>"
                    "<extra></extra>"
                )
            ),
            secondary_y=True,
        )
        
        # Update layout
        fig.update_xaxes(title_text="Week")
        fig.update_yaxes(title_text="Number of Posts", secondary_y=False)
        fig.update_yaxes(title_text="Sentiment Score", secondary_y=True, range=[-1, 1])
        
        fig.update_layout(
            title=title,
            height=450,
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig
    
    def create_topic_distribution_chart(self, thread_name: str = None, top_n: int = 10) -> go.Figure:
        """Create topic distribution chart"""
        topic_counts = self.extract_topics_for_wordcloud(thread_name)
        
        if not topic_counts:
            return None
        
        print(f"Debug: Total unique topics found: {len(topic_counts)}")
        print(f"Debug: Top topics: {dict(list(topic_counts.items())[:5])}")
        
        # Get top N topics, but ensure we don't exceed available topics
        available_topics = len(topic_counts)
        actual_top_n = min(top_n, available_topics)
        
        top_topics = dict(sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:actual_top_n])
        
        print(f"Debug: Requested {top_n}, available {available_topics}, showing {actual_top_n}")
        print(f"Debug: Final topics for chart: {top_topics}")
        
        if not top_topics:
            return None
        
        # Create horizontal bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=list(top_topics.values()),
                y=list(top_topics.keys()),
                orientation='h',
                marker=dict(
                    color=list(range(len(top_topics))),  # Use indices for consistent coloring
                    colorscale='viridis',
                    showscale=True,
                    colorbar=dict(title="Topic Rank")
                ),
                hovertemplate=(
                    "<b>Topic:</b> %{y}<br>"
                    "<b>Mentions:</b> %{x}<br>"
                    "<extra></extra>"
                )
            )
        ])
        
        title_suffix = f" - Thread {thread_name}" if thread_name else " - All Threads"
        fig.update_layout(
            title=f"🏷️ Top {actual_top_n} Discussion Topics{title_suffix}",
            xaxis_title="Number of Mentions",
            yaxis_title="Topics",
            height=max(400, actual_top_n * 40),  # Dynamic height based on number of topics
            template='plotly_white',
            showlegend=False
        )
        
        # Ensure y-axis shows all topics
        fig.update_yaxes(categoryorder="total ascending")
        
        return fig
    
    def create_weekly_cards_component(self, thread_name: str) -> None:
        """Create expandable cards for each week's analysis"""
        df = self.get_thread_data(thread_name)
        
        if df.empty:
            st.warning("No data available for selected thread")
            return
        
        st.subheader(f"📅 Weekly Analysis - Thread {thread_name}")
        
        # Sort by week in descending order (most recent first)
        df_sorted = df.sort_values('Week', ascending=False)
        
        for _, row in df_sorted.iterrows():
            week = row['Week']
            post_count = row['Post_Count']
            sentiment_score = row['Sentiment_Score']
            sentiment_label = row['Sentiment_Label']
            topics_str = row['Topics']  # Use original Topics column
            summary = row['Summary']
            date_range = row.get('Date_Range', 'N/A')
            
            # Process topics on-the-fly
            topics_processed = self._process_topics_string(topics_str)
            
            # Determine sentiment emoji and color
            if sentiment_score > 0.2:
                sentiment_emoji = "😊"
                card_color = "#d4edda"
            elif sentiment_score < -0.2:
                sentiment_emoji = "😟"
                card_color = "#f8d7da"
            else:
                sentiment_emoji = "😐"
                card_color = "#fff3cd"
            
            # Create expander for each week
            with st.expander(f"{sentiment_emoji} {week} | {post_count} posts | Sentiment: {sentiment_label} ({sentiment_score:.3f})"):
                
                # Create columns for better layout
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("**📄 Weekly Summary:**")
                    st.write(summary)
                    
                    st.markdown("**📅 Date Range:**")
                    st.write(date_range)
                
                with col2:
                    st.markdown("**📊 Key Metrics:**")
                    st.metric("Posts", post_count)
                    st.metric("Sentiment Score", f"{sentiment_score:.3f}")
                    
                    st.markdown("**🏷️ Main Topics:**")
                    # Display topics as tags
                    if topics_processed and isinstance(topics_processed, list):
                        for topic in topics_processed:
                            st.markdown(f"• {topic}")
                    else:
                        st.markdown("• No topics available")
    
    def get_thread_title(self, thread_id: str) -> str:
        """Get thread title from Contents sheet if available"""
        try:
            # Try forum_analysis_output folder first, then root
            file_paths = [
                'forum_analysis_output/master_forum_analysis.xlsx',
                'master_forum_analysis.xlsx'
            ]
            
            for file_path in file_paths:
                if Path(file_path).exists():
                    contents_df = pd.read_excel(file_path, sheet_name='Contents')
                    thread_info = contents_df[contents_df['Thread ID'].astype(str) == str(thread_id)]
                    if not thread_info.empty:
                        return thread_info.iloc[0]['Thread Title']
                    break
        except:
            pass
        return f"Thread {thread_id}"
    
    def create_horizontal_timeline_with_wordcloud(self, thread_name: str = None) -> None:
        """Create horizontal timeline with interactive word cloud"""
        if thread_name:
            df = self.get_thread_data(thread_name)
        else:
            df = self.get_combined_data()
        
        if df.empty:
            st.warning("No data available")
            return
        
        # Get unique weeks for slider
        weeks = sorted(df['Week'].unique())
        
        if not weeks:
            st.warning("No weeks found in data")
            return
        
        st.subheader("🕒 Interactive Timeline with Topic Word Cloud")
        
        # Create timeline slider
        if len(weeks) > 1:
            selected_week_idx = st.slider(
                "Select Week for Word Cloud",
                min_value=0,
                max_value=len(weeks) - 1,
                value=len(weeks) - 1  # Default to most recent
            )
            selected_week = weeks[selected_week_idx]
            st.info(f"Selected Week: **{selected_week}**")
        else:
            selected_week = weeks[0]
            st.info(f"Showing data for: {selected_week}")
        
        # Display word cloud for selected week
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Get topics for selected week
            week_data = df[df['Week'] == selected_week]
            if not week_data.empty:
                # Get topics for this specific week
                topic_counts = self.extract_topics_for_wordcloud(thread_name, (selected_week, selected_week))
                
                if topic_counts:
                    st.markdown("**🏷️ Topics for Selected Week:**")
                    
                    # Sort topics by frequency
                    sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
                    
                    # Create visual representation
                    html_tags = []
                    max_count = max(topic_counts.values()) if topic_counts else 1
                    for topic, count in sorted_topics:
                        # Size based on frequency (between 1em and 2.5em)
                        size = 1 + (count / max_count) * 1.5
                        color = f"hsl({hash(topic) % 360}, 70%, 50%)"
                        html_tags.append(f'<span style="font-size: {size}em; color: {color}; margin: 5px; font-weight: bold;">{topic}</span>')
                    
                    # Display as HTML
                    html_content = f'<div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px;">{" ".join(html_tags)}</div>'
                    st.markdown(html_content, unsafe_allow_html=True)
                else:
                    st.info("No topics found for this week")
            else:
                st.info("No data available for selected week")
        
        with col2:
            # Show week details
            if not week_data.empty:
                row = week_data.iloc[0]
                st.markdown("**📊 Week Details:**")
                st.metric("Posts", row['Post_Count'])
                st.metric("Sentiment", f"{row['Sentiment_Score']:.3f}")
                st.markdown(f"**Label:** {row['Sentiment_Label']}")
                
                if 'Date_Range' in row and pd.notna(row['Date_Range']):
                    st.markdown(f"**Date Range:** {row['Date_Range']}")
    
    def get_summary_metrics(self, thread_name: str = None) -> Dict:
        """Get summary metrics for dashboard"""
        if thread_name:
            df = self.get_thread_data(thread_name)
        else:
            df = self.get_combined_data()
        
        if df.empty:
            return {}
        
        total_posts = df['Post_Count'].sum()
        total_weeks = len(df)
        avg_sentiment = df['Sentiment_Score'].mean()
        
        # Get most recent week
        latest_week = df['Week'].max() if 'Week' in df.columns else "Unknown"
        
        return {
            'total_posts': total_posts,
            'total_weeks': total_weeks,
            'avg_sentiment': round(avg_sentiment, 3),
            'latest_week': latest_week,
            'threads_count': len(self.forum_data) if not thread_name else 1
        }