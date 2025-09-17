#!/usr/bin/env python3
"""
Multi-Agent Venture Studio - Streamlit Dashboard

A simple, elegant interface for reviewing signals and monitoring the system:
- Real-time signal discovery and review
- Interactive filtering and search
- Signal scoring visualization
- System health monitoring
- Analytics and trending detection

Run with: streamlit run streamlit_app.py
"""

import asyncio
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from persistence.database import init_database
from persistence.repositories import get_signal_repository
from agents.persistent_reddit_scout import PersistentRedditSignalsScout
from agents.claude_analyst import ClaudeAnalystAgent

# Page config
st.set_page_config(
    page_title="Multi-Agent Venture Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #e5e7eb;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.signal-card {
    background: white;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #e5e7eb;
    margin-bottom: 1rem;
}

.score-high { color: #059669; font-weight: bold; }
.score-medium { color: #d97706; font-weight: bold; }
.score-low { color: #dc2626; font-weight: bold; }

.header-icon { font-size: 1.2em; margin-right: 0.5rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_database_stats():
    """Load database statistics with caching"""
    async def _get_stats():
        await init_database()
        repo = get_signal_repository()
        return await repo.get_signal_stats()

    return asyncio.run(_get_stats())


@st.cache_data(ttl=60)  # Cache for 1 minute
def load_signals(limit=100, min_score=0.0, source_filter=None, search_query=None):
    """Load signals with caching"""
    async def _get_signals():
        await init_database()
        repo = get_signal_repository()

        if search_query:
            return await repo.search_signals(search_query, limit)
        elif source_filter and source_filter != "All Sources":
            return await repo.get_signals_by_source(source_filter, limit, min_score)
        else:
            return await repo.get_top_signals(limit, min_score)

    return asyncio.run(_get_signals())


def get_score_color_class(score):
    """Get CSS class for score color"""
    if score >= 0.6:
        return "score-high"
    elif score >= 0.3:
        return "score-medium"
    else:
        return "score-low"


def format_score(score):
    """Format score as percentage"""
    return f"{score * 100:.1f}%"


def main():
    """Main Streamlit app"""

    # Header
    st.markdown("# 🤖 Multi-Agent Venture Studio")
    st.markdown("AI-powered business opportunity discovery and validation platform")

    # Sidebar
    with st.sidebar:
        st.markdown("## Controls")

        # Refresh button
        if st.button("🔄 Refresh Data", key="refresh"):
            st.cache_data.clear()
            st.rerun()

        # Discovery section
        st.markdown("### Signal Discovery")

        # Discover new signals
        if st.button("🔍 Discover New Signals", key="discover"):
            with st.spinner("Discovering signals..."):
                try:
                    async def discover():
                        scout = PersistentRedditSignalsScout()
                        await scout.start()
                        try:
                            signals = await scout.discover_signals(
                                ['entrepreneur', 'startups'],
                                limit=10
                            )
                            return len(signals)
                        finally:
                            await scout.stop()

                    count = asyncio.run(discover())
                    st.success(f"✅ Discovered {count} signals!")
                    st.cache_data.clear()
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error: {e}")

        # Filters
        st.markdown("### Filters")

        min_score = st.slider(
            "Minimum Score",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="Filter signals by minimum final score"
        )

        limit = st.selectbox(
            "Number of Signals",
            [25, 50, 100, 200],
            index=1,
            help="Maximum number of signals to display"
        )

        # Source filter
        try:
            stats = load_database_stats()
            sources = ["All Sources"] + list(stats['sources'].keys())
            source_filter = st.selectbox("Source", sources)
        except:
            source_filter = "All Sources"

        # Search
        search_query = st.text_input(
            "Search Signals",
            placeholder="Search in titles and content...",
            help="Search for specific keywords in signal content"
        )

    # Main content area
    try:
        # Load data
        stats = load_database_stats()
        signals = load_signals(limit, min_score, source_filter, search_query)

        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "📊 Total Signals",
                f"{stats['total_signals']:,}",
                delta=f"{stats['recent_signals_24h']} in 24h"
            )

        with col2:
            st.metric(
                "🎯 Avg Final Score",
                format_score(stats['average_final_score']),
                help="Average final score across all signals"
            )

        with col3:
            st.metric(
                "💡 Avg Opportunity",
                format_score(stats['average_opportunity_score']),
                help="Average opportunity score"
            )

        with col4:
            st.metric(
                "📈 Sources",
                len(stats['sources']),
                help="Number of different signal sources"
            )

        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📋 Signals", "📊 Analytics", "🔥 Trending"])

        with tab1:
            st.markdown(f"### Signal Review ({len(signals)} signals)")

            if not signals:
                st.info("No signals found matching your criteria. Try adjusting the filters or discovering new signals.")
            else:
                # Signal list
                for signal in signals:
                    with st.container():
                        # Signal header
                        col1, col2, col3 = st.columns([3, 1, 1])

                        with col1:
                            st.markdown(f"**{signal.title}**")
                            st.caption(f"🔗 {signal.source} • {signal.discovered_at.strftime('%Y-%m-%d %H:%M')}")

                        with col2:
                            score_class = get_score_color_class(signal.final_score)
                            st.markdown(f"<span class='{score_class}'>Final: {format_score(signal.final_score)}</span>",
                                      unsafe_allow_html=True)

                        with col3:
                            if signal.url:
                                st.markdown(f"[🔗 View Source]({signal.url})")

                        # Signal details in expander
                        with st.expander("View Details"):
                            col1, col2 = st.columns(2)

                            with col1:
                                st.markdown("**Scores:**")
                                st.markdown(f"• Opportunity: {format_score(signal.opportunity_score)}")
                                st.markdown(f"• Engagement: {format_score(signal.engagement_score)}")
                                st.markdown(f"• Urgency: {format_score(signal.urgency_score)}")
                                st.markdown(f"• Confidence: {format_score(signal.confidence)}")

                            with col2:
                                if signal.keywords:
                                    st.markdown("**Keywords:**")
                                    keyword_badges = " ".join([f"`{kw}`" for kw in signal.keywords[:8]])
                                    st.markdown(keyword_badges)

                                if signal.signals_found:
                                    st.markdown("**Signals Found:**")
                                    signal_badges = " ".join([f"`{sf}`" for sf in signal.signals_found[:5]])
                                    st.markdown(signal_badges)

                            if signal.content:
                                st.markdown("**Content:**")
                                st.markdown(f"> {signal.content[:300]}...")

                            # Claude AI Analysis
                            st.markdown("---")
                            if st.button(f"🤖 Get Claude Analysis", key=f"claude_{signal.id}"):
                                with st.spinner("Analyzing signal with Claude AI..."):
                                    async def analyze_signal():
                                        try:
                                            # Mock signal object for analysis
                                            class MockSignal:
                                                def __init__(self, signal_id, title, content, source):
                                                    self.id = signal_id
                                                    self.title = title
                                                    self.content = content
                                                    self.source = source
                                                    self.final_score = 0.5
                                                    self.keywords = []
                                                    self.signals_found = []

                                            mock_signal = MockSignal(signal.id, signal.title, signal.content or "", signal.source)

                                            analyst = ClaudeAnalystAgent()
                                            if analyst.client:
                                                analysis = await analyst.analyze_signal(mock_signal)
                                                return analysis
                                            else:
                                                return None
                                        except Exception as e:
                                            st.error(f"Analysis failed: {e}")
                                            return None

                                    analysis = asyncio.run(analyze_signal())

                                    if analysis:
                                        st.success("✅ Claude Analysis Complete!")

                                        # Display analysis results
                                        col_a, col_b = st.columns(2)
                                        with col_a:
                                            st.metric("🎯 Business Potential", f"{analysis.business_potential:.1%}")
                                            st.metric("🔮 Confidence", f"{analysis.confidence:.1%}")
                                        with col_b:
                                            st.metric("📊 Market Size", analysis.market_size_estimate)
                                            st.metric("🔄 Opportunity Type", analysis.opportunity_type.title())

                                        st.markdown(f"**💰 Investment Thesis:**")
                                        st.markdown(f"> {analysis.investment_thesis}")

                                        if analysis.risk_factors:
                                            st.markdown(f"**⚠️ Key Risks:** {', '.join(analysis.risk_factors[:2])}")

                                        with st.expander("📋 Detailed Reasoning"):
                                            st.markdown(analysis.reasoning)
                                    else:
                                        st.error("❌ Claude analysis not available. Check API configuration.")

                        st.divider()

        with tab2:
            st.markdown("### Analytics Dashboard")

            if signals:
                # Convert to DataFrame for analysis
                df = pd.DataFrame([{
                    'title': s.title,
                    'source': s.source,
                    'final_score': s.final_score,
                    'opportunity_score': s.opportunity_score,
                    'engagement_score': s.engagement_score,
                    'urgency_score': s.urgency_score,
                    'discovered_at': s.discovered_at,
                    'keywords_count': len(s.keywords) if s.keywords else 0,
                    'signals_found_count': len(s.signals_found) if s.signals_found else 0
                } for s in signals])

                col1, col2 = st.columns(2)

                with col1:
                    # Score distribution
                    fig_scores = px.histogram(
                        df,
                        x='final_score',
                        nbins=20,
                        title="Final Score Distribution",
                        labels={'final_score': 'Final Score', 'count': 'Number of Signals'}
                    )
                    fig_scores.update_layout(height=400)
                    st.plotly_chart(fig_scores, use_container_width=True)

                with col2:
                    # Signals by source
                    source_counts = df['source'].value_counts()
                    fig_sources = px.pie(
                        values=source_counts.values,
                        names=source_counts.index,
                        title="Signals by Source"
                    )
                    fig_sources.update_layout(height=400)
                    st.plotly_chart(fig_sources, use_container_width=True)

                # Timeline
                if len(df) > 1:
                    df_timeline = df.set_index('discovered_at').resample('H').size().reset_index()
                    df_timeline.columns = ['Hour', 'Signals']

                    fig_timeline = px.line(
                        df_timeline,
                        x='Hour',
                        y='Signals',
                        title="Signal Discovery Timeline",
                        labels={'Hour': 'Time', 'Signals': 'Signals Discovered'}
                    )
                    fig_timeline.update_layout(height=400)
                    st.plotly_chart(fig_timeline, use_container_width=True)

                # Score correlation matrix
                score_cols = ['final_score', 'opportunity_score', 'engagement_score', 'urgency_score']
                correlation_matrix = df[score_cols].corr()

                fig_corr = px.imshow(
                    correlation_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="Score Correlation Matrix",
                    color_continuous_scale="RdBu_r"
                )
                fig_corr.update_layout(height=400)
                st.plotly_chart(fig_corr, use_container_width=True)

            else:
                st.info("No data available for analytics. Discover some signals first!")

        with tab3:
            st.markdown("### Trending Signals")

            # Get trending signals
            try:
                with st.spinner("Loading trending signals..."):
                    async def get_trending():
                        scout = PersistentRedditSignalsScout()
                        await scout.start()
                        try:
                            return await scout.get_trending_signals(hours=48)
                        finally:
                            await scout.stop()

                    trending_signals = asyncio.run(get_trending())

                if trending_signals:
                    st.markdown(f"Found {len(trending_signals)} trending signals in the last 48 hours:")

                    for i, signal in enumerate(trending_signals[:10], 1):
                        with st.container():
                            col1, col2, col3 = st.columns([0.5, 4, 1])

                            with col1:
                                st.markdown(f"**#{i}**")

                            with col2:
                                st.markdown(f"**{signal.title}**")
                                st.caption(f"r/{signal.subreddit} • {signal.num_comments} comments • Score: {signal.final_score:.2f}")

                                if signal.signals_found:
                                    badges = " ".join([f"`{sf.split(':')[1] if ':' in sf else sf}`" for sf in signal.signals_found[:4]])
                                    st.markdown(badges)

                            with col3:
                                if hasattr(signal, 'permalink') and signal.permalink:
                                    st.markdown(f"[🔗 Reddit](https://reddit.com{signal.permalink})")

                            st.divider()
                else:
                    st.info("No trending signals found. Try discovering more signals!")

            except Exception as e:
                st.error(f"Error loading trending signals: {e}")

        # Footer with system info
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.caption(f"🕒 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        with col2:
            st.caption(f"💾 Database: {stats['total_signals']} signals stored")

        with col3:
            st.caption("🤖 Multi-Agent Venture Studio v1.0")

    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.info("💡 Make sure to run `python examples/persistence_demo.py` first to populate the database!")


if __name__ == "__main__":
    main()