"""
Signal Clustering & Theme Detection

This module provides intelligent clustering of signals to identify emerging market themes
and group related business opportunities using NLP similarity analysis.

Features:
- Text similarity clustering using TF-IDF and cosine similarity
- Automatic theme detection and naming
- Trend analysis across time periods
- Integration with existing signal database
- Visualization support for dashboard

Methods:
- Content-based clustering (title + description)
- Keyword-based grouping
- Source-aware clustering
- Temporal theme tracking
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter, defaultdict
import json
import logging
import re

# Optional scikit-learn imports (graceful fallback)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import DBSCAN, KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Handle imports for different execution contexts
try:
    from ..persistence.repositories import get_signal_repository
    from ..persistence.models import Signal
    from ..persistence.database import init_database
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from persistence.repositories import get_signal_repository
    from persistence.models import Signal
    from persistence.database import init_database

logger = logging.getLogger(__name__)


@dataclass
class SignalCluster:
    """Represents a cluster of related signals"""
    cluster_id: str
    theme_name: str
    signals: List[Signal]
    keywords: List[str]
    strength: float  # 0-1, based on cohesion
    size: int
    avg_score: float
    sources: List[str]
    created_at: datetime
    description: str


@dataclass
class ThemeTrend:
    """Represents trending theme over time"""
    theme_name: str
    signal_count: int
    avg_score: float
    growth_rate: float  # signals per day
    keywords: List[str]
    time_span: timedelta
    first_seen: datetime
    last_seen: datetime


class SignalClusteringEngine:
    """
    Advanced signal clustering engine for theme detection

    Uses multiple approaches:
    1. Content similarity (TF-IDF + cosine similarity)
    2. Keyword overlap clustering
    3. Source-aware grouping
    4. Temporal pattern analysis
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Clustering parameters
        self.min_cluster_size = self.config.get('min_cluster_size', 2)
        self.max_clusters = self.config.get('max_clusters', 20)
        self.similarity_threshold = self.config.get('similarity_threshold', 0.3)

        # Text preprocessing
        self.stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'near', 'beside',
            'a', 'an', 'is', 'was', 'are', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
            'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their'
        }

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for clustering"""
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def extract_keywords(self, texts: List[str], max_keywords: int = 10) -> List[str]:
        """Extract meaningful keywords from a collection of texts"""
        if not texts:
            return []

        # Combine all texts
        combined_text = ' '.join(texts)
        words = combined_text.lower().split()

        # Filter out stop words and short words
        meaningful_words = [
            word for word in words
            if len(word) > 2 and word not in self.stop_words
        ]

        # Count frequency
        word_counts = Counter(meaningful_words)

        # Return top keywords
        return [word for word, count in word_counts.most_common(max_keywords)]

    async def cluster_signals_content(self, signals: List[Signal]) -> List[SignalCluster]:
        """
        Cluster signals based on content similarity using TF-IDF
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available, falling back to keyword clustering")
            return await self.cluster_signals_keywords(signals)

        if len(signals) < self.min_cluster_size:
            return []

        try:
            # Prepare text data
            texts = []
            for signal in signals:
                text_parts = [signal.title or ""]
                if signal.content:
                    text_parts.append(signal.content[:500])  # Limit content length
                if signal.keywords:
                    text_parts.extend(signal.keywords)

                combined_text = ' '.join(text_parts)
                preprocessed = self.preprocess_text(combined_text)
                texts.append(preprocessed)

            # TF-IDF Vectorization
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )

            tfidf_matrix = vectorizer.fit_transform(texts)

            # Compute similarity matrix
            similarity_matrix = cosine_similarity(tfidf_matrix)

            # DBSCAN clustering with cosine distance
            clustering = DBSCAN(
                eps=1 - self.similarity_threshold,  # Distance threshold
                min_samples=self.min_cluster_size,
                metric='cosine'
            )

            # Use TF-IDF vectors directly for cosine distance
            cluster_labels = clustering.fit_predict(tfidf_matrix.toarray())

            # Group signals by cluster
            clusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                if label != -1:  # -1 is noise in DBSCAN
                    clusters[label].append(signals[i])

            # Create SignalCluster objects
            signal_clusters = []
            for cluster_id, cluster_signals in clusters.items():
                if len(cluster_signals) >= self.min_cluster_size:
                    cluster = await self.create_cluster(
                        cluster_id=f"content_{cluster_id}",
                        signals=cluster_signals
                    )
                    signal_clusters.append(cluster)

            logger.info(f"✅ Content clustering complete: {len(signal_clusters)} clusters from {len(signals)} signals")
            return signal_clusters

        except Exception as e:
            logger.error(f"Content clustering failed: {e}")
            return []

    async def cluster_signals_keywords(self, signals: List[Signal]) -> List[SignalCluster]:
        """
        Cluster signals based on keyword overlap (fallback method)
        """
        if len(signals) < self.min_cluster_size:
            return []

        try:
            # Group signals by overlapping keywords
            keyword_groups = defaultdict(list)

            for signal in signals:
                signal_keywords = set()

                # Extract keywords from different sources
                if signal.keywords:
                    signal_keywords.update([k.lower() for k in signal.keywords])

                if signal.signals_found:
                    signal_keywords.update([s.lower() for s in signal.signals_found])

                # Extract from title
                if signal.title:
                    title_words = self.preprocess_text(signal.title).split()
                    signal_keywords.update([w for w in title_words if len(w) > 3])

                # Find the best matching group or create new one
                best_group = None
                best_overlap = 0

                for group_key, group_signals in keyword_groups.items():
                    group_keywords = set(group_key.split(','))
                    overlap = len(signal_keywords.intersection(group_keywords))

                    if overlap > best_overlap and overlap >= 2:
                        best_overlap = overlap
                        best_group = group_key

                if best_group:
                    keyword_groups[best_group].append(signal)
                else:
                    # Create new group
                    if signal_keywords:
                        new_key = ','.join(sorted(list(signal_keywords))[:5])
                        keyword_groups[new_key].append(signal)

            # Create clusters from groups
            signal_clusters = []
            for i, (keywords_str, cluster_signals) in enumerate(keyword_groups.items()):
                if len(cluster_signals) >= self.min_cluster_size:
                    cluster = await self.create_cluster(
                        cluster_id=f"keyword_{i}",
                        signals=cluster_signals
                    )
                    signal_clusters.append(cluster)

            logger.info(f"✅ Keyword clustering complete: {len(signal_clusters)} clusters from {len(signals)} signals")
            return signal_clusters

        except Exception as e:
            logger.error(f"Keyword clustering failed: {e}")
            return []

    async def create_cluster(self, cluster_id: str, signals: List[Signal]) -> SignalCluster:
        """Create a SignalCluster object with computed metadata"""

        # Extract text for analysis
        all_texts = []
        for signal in signals:
            text_parts = [signal.title or ""]
            if signal.content:
                text_parts.append(signal.content[:200])
            all_texts.extend(text_parts)

        # Extract keywords
        keywords = self.extract_keywords(all_texts, max_keywords=10)

        # Generate theme name
        theme_name = self.generate_theme_name(signals, keywords)

        # Calculate metrics
        avg_score = np.mean([signal.final_score for signal in signals]) if signals else 0.0
        sources = list(set([signal.source for signal in signals]))

        # Calculate cluster strength (cohesion)
        strength = min(len(signals) / 10.0, 1.0)  # Simple metric based on size

        # Generate description
        description = self.generate_cluster_description(signals, keywords)

        return SignalCluster(
            cluster_id=cluster_id,
            theme_name=theme_name,
            signals=signals,
            keywords=keywords,
            strength=strength,
            size=len(signals),
            avg_score=avg_score,
            sources=sources,
            created_at=datetime.utcnow(),
            description=description
        )

    def generate_theme_name(self, signals: List[Signal], keywords: List[str]) -> str:
        """Generate a descriptive theme name for a cluster"""

        # Priority keywords for theme naming
        tech_keywords = {'ai', 'ml', 'blockchain', 'api', 'saas', 'platform', 'app', 'tool'}
        business_keywords = {'startup', 'business', 'venture', 'market', 'growth'}
        domain_keywords = {'fintech', 'healthtech', 'edtech', 'devtools', 'ecommerce'}

        # Find the most relevant keywords
        relevant_keywords = []

        for keyword in keywords[:5]:
            if keyword in tech_keywords:
                relevant_keywords.append(keyword.upper())
            elif keyword in business_keywords:
                relevant_keywords.append(keyword.title())
            elif keyword in domain_keywords:
                relevant_keywords.append(keyword.title())
            else:
                relevant_keywords.append(keyword.title())

        if relevant_keywords:
            if len(relevant_keywords) == 1:
                return f"{relevant_keywords[0]} Opportunities"
            elif len(relevant_keywords) == 2:
                return f"{relevant_keywords[0]} & {relevant_keywords[1]}"
            else:
                return f"{relevant_keywords[0]} + {relevant_keywords[1]} Ecosystem"

        # Fallback based on signal sources
        sources = set([signal.source for signal in signals])
        if len(sources) == 1:
            return f"{list(sources)[0].title()} Opportunities"
        else:
            return f"Mixed Market Signals"

    def generate_cluster_description(self, signals: List[Signal], keywords: List[str]) -> str:
        """Generate a description for the cluster"""

        signal_count = len(signals)
        avg_score = np.mean([signal.final_score for signal in signals])
        sources = set([signal.source for signal in signals])

        # Time span
        dates = [signal.discovered_at for signal in signals if signal.discovered_at]
        time_span = ""
        if dates:
            earliest = min(dates)
            latest = max(dates)
            days_span = (latest - earliest).days
            if days_span > 0:
                time_span = f"spanning {days_span} days"
            else:
                time_span = "from today"

        description = f"Cluster of {signal_count} signals with average score {avg_score:.2f}. "
        description += f"Sources: {', '.join(sources)}. "

        if time_span:
            description += f"Discovered {time_span}. "

        if keywords:
            description += f"Key themes: {', '.join(keywords[:5])}."

        return description

    async def detect_trending_themes(self, days_back: int = 7) -> List[ThemeTrend]:
        """
        Detect trending themes by analyzing clusters over time
        """
        try:
            # Get signals from the specified time period
            await init_database()
            signal_repo = get_signal_repository()

            cutoff_date = datetime.utcnow() - timedelta(days=days_back)

            # This would need a method to get signals by date range
            # For now, get all signals and filter
            all_signals = await signal_repo.get_top_signals(limit=1000, min_score=0.1)
            recent_signals = [
                signal for signal in all_signals
                if signal.discovered_at and signal.discovered_at >= cutoff_date
            ]

            if len(recent_signals) < 3:
                return []

            # Cluster recent signals
            clusters = await self.cluster_signals_content(recent_signals)

            # Analyze trends
            trends = []
            for cluster in clusters:
                if cluster.size >= 2:  # Minimum for trend analysis

                    # Calculate growth rate
                    signal_dates = [s.discovered_at for s in cluster.signals if s.discovered_at]
                    if signal_dates:
                        earliest = min(signal_dates)
                        latest = max(signal_dates)
                        time_span = latest - earliest

                        if time_span.total_seconds() > 0:
                            growth_rate = cluster.size / max(time_span.days, 1)
                        else:
                            growth_rate = cluster.size  # All signals on same day

                        trend = ThemeTrend(
                            theme_name=cluster.theme_name,
                            signal_count=cluster.size,
                            avg_score=cluster.avg_score,
                            growth_rate=growth_rate,
                            keywords=cluster.keywords,
                            time_span=time_span,
                            first_seen=earliest,
                            last_seen=latest
                        )
                        trends.append(trend)

            # Sort by growth rate and signal quality
            trends.sort(key=lambda t: (t.growth_rate * t.avg_score), reverse=True)

            logger.info(f"✅ Trending themes analysis complete: {len(trends)} trends detected")
            return trends[:10]  # Top 10 trends

        except Exception as e:
            logger.error(f"Trending themes detection failed: {e}")
            return []

    async def get_cluster_insights(self, clusters: List[SignalCluster]) -> Dict[str, Any]:
        """
        Generate insights from signal clusters
        """

        if not clusters:
            return {"total_clusters": 0, "insights": []}

        insights = []

        # Overall statistics
        total_signals = sum(cluster.size for cluster in clusters)
        avg_cluster_size = total_signals / len(clusters)

        insights.append(f"Identified {len(clusters)} market themes from {total_signals} signals")
        insights.append(f"Average cluster size: {avg_cluster_size:.1f} signals")

        # Top themes by size
        largest_clusters = sorted(clusters, key=lambda c: c.size, reverse=True)[:3]
        if largest_clusters:
            insights.append("Largest themes:")
            for i, cluster in enumerate(largest_clusters, 1):
                insights.append(f"  {i}. {cluster.theme_name} ({cluster.size} signals)")

        # Top themes by quality
        highest_quality = sorted(clusters, key=lambda c: c.avg_score, reverse=True)[:3]
        if highest_quality:
            insights.append("Highest quality themes:")
            for i, cluster in enumerate(highest_quality, 1):
                insights.append(f"  {i}. {cluster.theme_name} (score: {cluster.avg_score:.2f})")

        # Source diversity
        all_sources = set()
        for cluster in clusters:
            all_sources.update(cluster.sources)
        insights.append(f"Signals from {len(all_sources)} sources: {', '.join(all_sources)}")

        return {
            "total_clusters": len(clusters),
            "total_signals": total_signals,
            "avg_cluster_size": avg_cluster_size,
            "insights": insights,
            "top_themes": [c.theme_name for c in largest_clusters],
            "sources": list(all_sources)
        }


# Example usage and testing
async def main():
    """Demo the signal clustering engine"""

    print("🔍 Signal Clustering & Theme Detection Demo")
    print("=" * 45)

    clustering_engine = SignalClusteringEngine()

    try:
        # Get signals from database
        await init_database()
        signal_repo = get_signal_repository()
        signals = await signal_repo.get_top_signals(limit=50, min_score=0.2)

        if not signals:
            print("ℹ️  No signals found for clustering")
            print("💡 Run signal discovery first to populate the database")
            return

        print(f"📊 Analyzing {len(signals)} signals for clustering...")

        # Perform clustering
        clusters = await clustering_engine.cluster_signals_content(signals)

        if clusters:
            print(f"\n🎯 Found {len(clusters)} market themes:")
            print("-" * 50)

            for i, cluster in enumerate(clusters, 1):
                print(f"\n{i}. {cluster.theme_name}")
                print(f"   Size: {cluster.size} signals | Score: {cluster.avg_score:.2f}")
                print(f"   Keywords: {', '.join(cluster.keywords[:5])}")
                print(f"   Sources: {', '.join(cluster.sources)}")
                print(f"   Description: {cluster.description}")

        # Trending themes analysis
        print(f"\n📈 Analyzing trending themes...")
        trends = await clustering_engine.detect_trending_themes(days_back=7)

        if trends:
            print(f"\n🔥 Top trending themes (last 7 days):")
            print("-" * 40)

            for i, trend in enumerate(trends[:5], 1):
                print(f"{i}. {trend.theme_name}")
                print(f"   {trend.signal_count} signals | Growth: {trend.growth_rate:.1f}/day")
                print(f"   Quality: {trend.avg_score:.2f} | Keywords: {', '.join(trend.keywords[:3])}")

        # Generate insights
        insights = await clustering_engine.get_cluster_insights(clusters)

        print(f"\n💡 Market Insights:")
        print("-" * 20)
        for insight in insights["insights"]:
            print(f"• {insight}")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())