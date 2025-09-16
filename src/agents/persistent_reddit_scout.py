"""
Persistent Reddit Signals Scout

Enhanced version of RedditSignalsScout that:
- Saves all discovered signals to database
- Avoids duplicate signals with intelligent deduplication
- Provides signal history and analytics
- Integrates with the venture studio persistence layer

This demonstrates:
- Database integration with async agents
- Deduplication strategies for external data
- Historical analysis and trend detection
- Performance optimization with bulk operations
"""

import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any

from .reddit_signals_scout import RedditSignalsScout, RedditSignal

# Handle imports based on execution context
try:
    from ..persistence.database import init_database
    from ..persistence.repositories import get_signal_repository
except ImportError:
    # When run as main module or from examples
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from persistence.database import init_database
    from persistence.repositories import get_signal_repository


class PersistentRedditSignalsScout(RedditSignalsScout):
    """
    Reddit Signals Scout with database persistence

    Enhanced features:
    - Automatic signal storage in database
    - Intelligent deduplication (same source + title)
    - Historical signal analysis
    - Batch operations for performance
    - Signal trend detection
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.signal_repo = None
        self.enable_persistence = config.get('enable_persistence', True) if config else True

    async def start(self):
        """Initialize Reddit client and database connection"""
        # Initialize parent (Reddit API client)
        await super().start()

        # Initialize database connection
        if self.enable_persistence:
            await init_database()
            self.signal_repo = get_signal_repository()

    async def discover_signals(self,
                             subreddits: Optional[List[str]] = None,
                             timeframe: str = 'day',
                             limit: Optional[int] = None,
                             force_refresh: bool = False) -> List[RedditSignal]:
        """
        Discover and persist Reddit signals

        Args:
            subreddits: List of subreddit names
            timeframe: Reddit timeframe ('hour', 'day', 'week', etc.)
            limit: Max signals per subreddit
            force_refresh: Skip cache and fetch fresh data

        Returns:
            List of RedditSignal objects (may include cached signals)
        """

        # Get fresh signals from Reddit API
        fresh_signals = await super().discover_signals(subreddits, timeframe, limit)

        if not self.enable_persistence or not self.signal_repo:
            return fresh_signals

        # Persist new signals to database
        persisted_signals = await self._persist_signals(fresh_signals)

        # Optionally merge with recent cached signals for better coverage
        if not force_refresh:
            cached_signals = await self._get_recent_cached_signals(subreddits, timeframe)
            return self._merge_signal_lists(persisted_signals, cached_signals)

        return persisted_signals

    async def _persist_signals(self, signals: List[RedditSignal]) -> List[RedditSignal]:
        """Persist signals to database with deduplication"""

        persisted_signals = []

        for signal in signals:
            try:
                # Convert RedditSignal to database format
                db_signal = await self.signal_repo.create_signal(
                    signal_id=signal.id,
                    source=signal.source,
                    title=signal.title,
                    content=signal.content,
                    url=signal.url or signal.permalink,
                    agent_id=self.id if hasattr(self, 'id') else 'reddit-signals-scout',
                    signal_timestamp=datetime.fromtimestamp(signal.created_utc),
                    raw_score=float(signal.score),
                    opportunity_score=signal.opportunity_score,
                    engagement_score=signal.engagement_score,
                    urgency_score=signal.urgency_score,
                    final_score=signal.final_score,
                    confidence=0.8,  # Default confidence for Reddit signals
                    source_metadata={
                        'author': signal.author,
                        'subreddit': signal.subreddit,
                        'upvotes': signal.score,
                        'comments': signal.num_comments,
                        'permalink': signal.permalink
                    },
                    keywords=signal.keywords,
                    signals_found=signal.signals_found,
                    category='reddit-signal'
                )

                persisted_signals.append(signal)

            except Exception as e:
                # Log but don't fail - might be duplicate or other issue
                print(f"Warning: Could not persist signal {signal.id}: {e}")
                persisted_signals.append(signal)  # Include in results anyway

        return persisted_signals

    async def _get_recent_cached_signals(self,
                                       subreddits: Optional[List[str]] = None,
                                       timeframe: str = 'day',
                                       hours: int = 6) -> List[RedditSignal]:
        """Get recently cached signals from database"""

        if not self.signal_repo:
            return []

        try:
            # Calculate time threshold
            since = datetime.utcnow().replace(hour=datetime.utcnow().hour - hours)

            # Get signals from database
            if subreddits:
                all_cached = []
                for subreddit in subreddits:
                    cached = await self.signal_repo.get_signals_by_source(
                        source=f"reddit:r/{subreddit}",
                        limit=20,
                        min_score=0.3
                    )
                    all_cached.extend(cached)
            else:
                all_cached = await self.signal_repo.get_top_signals(
                    limit=50,
                    min_score=0.3,
                    since=since
                )

            # Convert database signals back to RedditSignal format
            reddit_signals = []
            for db_signal in all_cached:
                reddit_signal = self._db_signal_to_reddit_signal(db_signal)
                reddit_signals.append(reddit_signal)

            return reddit_signals

        except Exception as e:
            print(f"Warning: Could not get cached signals: {e}")
            return []

    def _db_signal_to_reddit_signal(self, db_signal) -> RedditSignal:
        """Convert database signal back to RedditSignal"""

        metadata = db_signal.source_metadata or {}

        return RedditSignal(
            id=db_signal.id,
            source=db_signal.source,
            content=db_signal.content or "",
            title=db_signal.title,
            url=db_signal.url or "",
            author=metadata.get('author', '[unknown]'),
            subreddit=metadata.get('subreddit', 'unknown'),
            score=int(metadata.get('upvotes', 0)),
            num_comments=int(metadata.get('comments', 0)),
            created_utc=db_signal.signal_timestamp.timestamp() if db_signal.signal_timestamp else 0,
            permalink=metadata.get('permalink', ''),
            engagement_score=db_signal.engagement_score,
            opportunity_score=db_signal.opportunity_score,
            urgency_score=db_signal.urgency_score,
            final_score=db_signal.final_score,
            timestamp=db_signal.discovered_at,
            keywords=db_signal.keywords or [],
            signals_found=db_signal.signals_found or []
        )

    def _merge_signal_lists(self,
                          fresh_signals: List[RedditSignal],
                          cached_signals: List[RedditSignal]) -> List[RedditSignal]:
        """Merge fresh and cached signals, removing duplicates"""

        # Create lookup for fresh signals
        fresh_ids = {signal.id for signal in fresh_signals}

        # Add cached signals that aren't in fresh results
        merged = list(fresh_signals)
        for cached in cached_signals:
            if cached.id not in fresh_ids:
                merged.append(cached)

        # Sort by final score
        merged.sort(key=lambda s: s.final_score, reverse=True)

        return merged

    async def get_signal_history(self,
                               subreddit: Optional[str] = None,
                               days: int = 7,
                               min_score: float = 0.3) -> List[RedditSignal]:
        """Get historical signals from database"""

        if not self.signal_repo:
            return []

        try:
            since = datetime.utcnow().replace(day=datetime.utcnow().day - days)

            if subreddit:
                db_signals = await self.signal_repo.get_signals_by_source(
                    source=f"reddit:r/{subreddit}",
                    limit=100,
                    min_score=min_score
                )
            else:
                db_signals = await self.signal_repo.get_top_signals(
                    limit=100,
                    min_score=min_score,
                    since=since
                )

            # Convert to RedditSignal format
            return [self._db_signal_to_reddit_signal(db_signal) for db_signal in db_signals]

        except Exception as e:
            print(f"Error getting signal history: {e}")
            return []

    async def get_trending_signals(self, hours: int = 24) -> List[RedditSignal]:
        """Get signals that are trending (high engagement velocity)"""

        history = await self.get_signal_history(days=1, min_score=0.2)

        # Calculate trending score based on recency and engagement
        trending = []
        for signal in history:
            age_hours = (datetime.utcnow() - signal.timestamp).total_seconds() / 3600

            if age_hours <= hours:
                # Calculate velocity: comments per hour since creation
                velocity = signal.num_comments / max(age_hours, 1)
                trending_score = signal.final_score * (1 + velocity / 10)

                signal.final_score = trending_score  # Update for sorting
                trending.append(signal)

        trending.sort(key=lambda s: s.final_score, reverse=True)
        return trending[:20]

    async def analyze_signal_patterns(self, days: int = 30) -> Dict[str, Any]:
        """Analyze patterns in historical signals"""

        if not self.signal_repo:
            return {"error": "Persistence not enabled"}

        try:
            stats = await self.signal_repo.get_signal_stats()

            # Get recent signals for pattern analysis
            recent_signals = await self.get_signal_history(days=days, min_score=0.1)

            # Analyze patterns
            subreddit_performance = {}
            keyword_frequency = {}
            daily_counts = {}

            for signal in recent_signals:
                # Subreddit performance
                if signal.subreddit not in subreddit_performance:
                    subreddit_performance[signal.subreddit] = {
                        'count': 0, 'avg_score': 0, 'total_score': 0
                    }

                perf = subreddit_performance[signal.subreddit]
                perf['count'] += 1
                perf['total_score'] += signal.final_score
                perf['avg_score'] = perf['total_score'] / perf['count']

                # Keyword frequency
                for keyword in signal.keywords or []:
                    keyword_frequency[keyword] = keyword_frequency.get(keyword, 0) + 1

                # Daily counts
                day = signal.timestamp.date().isoformat()
                daily_counts[day] = daily_counts.get(day, 0) + 1

            # Sort results
            top_subreddits = sorted(
                subreddit_performance.items(),
                key=lambda x: x[1]['avg_score'],
                reverse=True
            )[:10]

            top_keywords = sorted(
                keyword_frequency.items(),
                key=lambda x: x[1],
                reverse=True
            )[:20]

            return {
                "database_stats": stats,
                "analysis_period_days": days,
                "total_signals_analyzed": len(recent_signals),
                "top_performing_subreddits": top_subreddits,
                "trending_keywords": top_keywords,
                "daily_signal_counts": daily_counts,
                "average_opportunity_score": sum(s.opportunity_score for s in recent_signals) / len(recent_signals) if recent_signals else 0
            }

        except Exception as e:
            return {"error": f"Analysis failed: {e}"}


# Example usage and testing
async def main():
    """Demo the persistent Reddit scout"""

    print("🤖 Persistent Reddit Signals Scout Demo")
    print("=" * 50)

    scout = PersistentRedditSignalsScout({
        'enable_persistence': True
    })

    await scout.start()

    try:
        # Discover fresh signals
        print("Discovering fresh Reddit signals...")
        signals = await scout.discover_signals(['entrepreneur'], limit=10)
        print(f"Found {len(signals)} signals")

        # Get signal history
        print("\nGetting signal history...")
        history = await scout.get_signal_history(days=1)
        print(f"Historical signals: {len(history)}")

        # Analyze patterns
        print("\nAnalyzing signal patterns...")
        analysis = await scout.analyze_signal_patterns(days=7)
        print(f"Database stats: {analysis.get('database_stats', {})}")

        print("\n✅ Persistent Reddit scout demo completed!")

    finally:
        await scout.stop()


if __name__ == "__main__":
    asyncio.run(main())