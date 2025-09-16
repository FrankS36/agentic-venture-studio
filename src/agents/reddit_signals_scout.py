"""
Reddit Signals Scout Agent

Discovers business signals from Reddit using async HTTP requests.
Focuses on entrepreneurship and business-related subreddits to identify:
- Pain points and problems people are discussing
- New trends and opportunities
- Market validation requests
- Business idea discussions

This agent demonstrates:
- Real API integration with proper error handling
- Rate limiting and respectful API usage
- Signal scoring based on engagement metrics
- Async HTTP client management
"""

import asyncio
import aiohttp
import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
from urllib.parse import urlencode

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv is optional - continue without it
    pass


@dataclass
class RedditSignal:
    """Enhanced signal with Reddit-specific metadata"""
    id: str
    source: str
    content: str
    title: str
    url: str
    author: str
    subreddit: str
    score: int  # Reddit upvotes
    num_comments: int
    created_utc: float
    permalink: str

    # Calculated fields
    engagement_score: float = 0.0
    opportunity_score: float = 0.0
    urgency_score: float = 0.0
    final_score: float = 0.0

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    keywords: List[str] = field(default_factory=list)
    signals_found: List[str] = field(default_factory=list)


class RedditRateLimiter:
    """Simple rate limiter for Reddit API requests"""

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = []
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Wait if necessary to respect rate limits"""
        async with self.lock:
            now = time.time()

            # Remove requests older than 1 minute
            self.requests = [req_time for req_time in self.requests
                           if now - req_time < 60]

            # Check if we need to wait
            if len(self.requests) >= self.requests_per_minute:
                sleep_time = 60 - (now - self.requests[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    return await self.acquire()

            # Record this request
            self.requests.append(now)


class RedditSignalsScout:
    """Async agent for discovering business signals from Reddit"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.client_id = os.getenv('REDDIT_CLIENT_ID', '')
        self.client_secret = os.getenv('REDDIT_CLIENT_SECRET', '')
        self.user_agent = os.getenv('REDDIT_USER_AGENT',
                                   'agentic-venture-studio/1.0')

        # Rate limiting
        requests_per_minute = int(os.getenv('REDDIT_REQUESTS_PER_MINUTE', '60'))
        self.rate_limiter = RedditRateLimiter(requests_per_minute)

        # Configuration
        self.min_score = int(os.getenv('REDDIT_MIN_SCORE', '10'))
        self.min_comments = int(os.getenv('REDDIT_MIN_COMMENTS', '5'))
        self.search_limit = int(os.getenv('REDDIT_SEARCH_LIMIT', '25'))

        # Default subreddits for business signals
        default_subs = os.getenv('REDDIT_DEFAULT_SUBREDDITS',
                               'entrepreneur,startups,SaaS,digitalnomad,business')
        self.default_subreddits = [s.strip() for s in default_subs.split(',')]

        # Keywords that indicate business opportunities (expanded list)
        self.opportunity_keywords = {
            'pain_points': ['problem', 'issue', 'frustrating', 'difficult', 'hate', 'annoying',
                          'wish there was', 'need a solution', 'struggle', 'challenge', 'pain',
                          'broken', 'sucks', 'terrible', 'awful', 'worst', 'fix'],
            'market_gaps': ['no one is doing', 'missing', 'doesnt exist', 'would pay for',
                          'market gap', 'untapped', 'opportunity', 'niche', 'underserved',
                          'no good options', 'lacking', 'absent'],
            'trends': ['trending', 'popular', 'growing', 'emerging', 'new', 'future of',
                     'next big thing', 'boom', 'exploding', 'hot', 'viral', 'rising',
                     'increase', 'demand', 'growth'],
            'validation': ['would you use', 'feedback', 'validate', 'market research',
                         'survey', 'interested in', 'thoughts', 'opinions', 'advice',
                         'help', 'input', 'review'],
            'business_signals': ['startup', 'business', 'entrepreneur', 'idea', 'venture',
                               'saas', 'product', 'service', 'launch', 'mvp', 'prototype',
                               'revenue', 'customers', 'users', 'scaling']
        }

        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None
        self.access_token: Optional[str] = None
        self.token_expires: float = 0

    async def start(self):
        """Initialize the Reddit client"""
        self.session = aiohttp.ClientSession(
            headers={'User-Agent': self.user_agent},
            timeout=aiohttp.ClientTimeout(total=30)
        )

        # Get access token if credentials are provided
        if self.client_id and self.client_secret:
            await self._authenticate()

    async def stop(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()

    async def _authenticate(self):
        """Get Reddit API access token"""
        if not self.session:
            raise RuntimeError("Session not initialized")

        # Check if token is still valid
        if self.access_token and time.time() < self.token_expires:
            return

        auth = aiohttp.BasicAuth(self.client_id, self.client_secret)
        data = {
            'grant_type': 'client_credentials'
        }

        try:
            await self.rate_limiter.acquire()
            async with self.session.post(
                'https://www.reddit.com/api/v1/access_token',
                auth=auth,
                data=data
            ) as response:
                if response.status == 200:
                    token_data = await response.json()
                    self.access_token = token_data['access_token']
                    self.token_expires = time.time() + token_data['expires_in'] - 60

                    # Update session headers
                    self.session.headers.update({
                        'Authorization': f'Bearer {self.access_token}'
                    })
                else:
                    print(f"Authentication failed: {response.status}")

        except Exception as e:
            print(f"Authentication error: {e}")

    async def discover_signals(self,
                             subreddits: Optional[List[str]] = None,
                             timeframe: str = 'day',
                             limit: Optional[int] = None) -> List[RedditSignal]:
        """
        Discover business signals from specified subreddits

        Args:
            subreddits: List of subreddit names (without r/)
            timeframe: 'hour', 'day', 'week', 'month', 'year', 'all'
            limit: Max posts per subreddit (default from config)
        """
        if not self.session:
            raise RuntimeError("Reddit client not started. Call start() first.")

        subreddits = subreddits or self.default_subreddits
        limit = limit or self.search_limit

        all_signals = []

        # Process each subreddit concurrently
        tasks = [
            self._scan_subreddit(subreddit, timeframe, limit)
            for subreddit in subreddits
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Error scanning r/{subreddits[i]}: {result}")
            else:
                all_signals.extend(result)

        # Score and rank signals
        scored_signals = await self._score_signals(all_signals)

        # Sort by final score
        scored_signals.sort(key=lambda s: s.final_score, reverse=True)

        return scored_signals

    async def _scan_subreddit(self,
                            subreddit: str,
                            timeframe: str,
                            limit: int) -> List[RedditSignal]:
        """Scan a specific subreddit for signals"""

        signals = []

        # Try multiple sorting methods to get diverse content
        sort_methods = ['hot', 'top', 'new']
        posts_per_method = max(1, limit // len(sort_methods))

        for sort_method in sort_methods:
            try:
                posts = await self._fetch_posts(subreddit, sort_method,
                                              timeframe, posts_per_method)

                for post in posts:
                    signal = await self._post_to_signal(post, subreddit)
                    if signal and self._is_relevant_signal(signal):
                        signals.append(signal)

            except Exception as e:
                print(f"Error fetching {sort_method} posts from r/{subreddit}: {e}")

        return signals

    async def _fetch_posts(self,
                         subreddit: str,
                         sort: str,
                         timeframe: str,
                         limit: int) -> List[Dict[str, Any]]:
        """Fetch posts from Reddit API"""

        if not self.session:
            return []

        # Ensure we have a valid token
        if self.client_id and self.client_secret:
            await self._authenticate()

        # Build URL
        base_url = 'https://oauth.reddit.com' if self.access_token else 'https://www.reddit.com'
        url = f"{base_url}/r/{subreddit}/{sort}.json"

        params = {
            'limit': limit,
            't': timeframe,
            'raw_json': 1
        }

        try:
            await self.rate_limiter.acquire()

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return [child['data'] for child in data['data']['children']]
                else:
                    print(f"Reddit API error {response.status} for r/{subreddit}")
                    return []

        except Exception as e:
            print(f"Request error for r/{subreddit}: {e}")
            return []

    async def _post_to_signal(self, post: Dict[str, Any], subreddit: str) -> Optional[RedditSignal]:
        """Convert Reddit post to RedditSignal"""

        try:
            return RedditSignal(
                id=post['id'],
                source='reddit',
                content=post.get('selftext', '')[:500],  # Limit content length
                title=post['title'],
                url=post.get('url', ''),
                author=post.get('author', '[deleted]'),
                subreddit=subreddit,
                score=post.get('score', 0),
                num_comments=post.get('num_comments', 0),
                created_utc=post.get('created_utc', 0),
                permalink=f"https://reddit.com{post.get('permalink', '')}"
            )
        except Exception as e:
            print(f"Error converting post to signal: {e}")
            return None

    def _is_relevant_signal(self, signal: RedditSignal) -> bool:
        """Filter signals based on relevance criteria"""

        # Basic filters
        if signal.score < self.min_score:
            return False

        if signal.num_comments < self.min_comments:
            return False

        # Age filter (not too old)
        age_hours = (time.time() - signal.created_utc) / 3600
        if age_hours > 24 * 7:  # Older than a week
            return False

        # Content quality filter
        text = (signal.title + ' ' + signal.content).lower()

        # Skip common low-quality posts
        skip_phrases = ['ama', 'ask me anything', 'daily thread', 'weekly thread',
                       'what are you working on', 'how was your week']

        if any(phrase in text for phrase in skip_phrases):
            return False

        return True

    async def _score_signals(self, signals: List[RedditSignal]) -> List[RedditSignal]:
        """Score signals based on business opportunity potential"""

        for signal in signals:
            # Engagement score (0-1)
            max_score = max((s.score for s in signals), default=1)
            max_comments = max((s.num_comments for s in signals), default=1)

            signal.engagement_score = (
                0.7 * (signal.score / max_score) +
                0.3 * (signal.num_comments / max_comments)
            )

            # Opportunity score based on keywords (0-1)
            signal.opportunity_score = self._calculate_opportunity_score(signal)

            # Urgency score based on recency and engagement (0-1)
            signal.urgency_score = self._calculate_urgency_score(signal)

            # Final weighted score
            signal.final_score = (
                0.4 * signal.opportunity_score +
                0.3 * signal.engagement_score +
                0.3 * signal.urgency_score
            )

        return signals

    def _calculate_opportunity_score(self, signal: RedditSignal) -> float:
        """Calculate opportunity score based on keyword analysis"""

        text = (signal.title + ' ' + signal.content).lower()
        score = 0.0
        signals_found = []

        for category, keywords in self.opportunity_keywords.items():
            category_score = 0
            for keyword in keywords:
                if keyword in text:
                    category_score += 1
                    signals_found.append(f"{category}:{keyword}")

            # Normalize by category size and add to total
            if category_score > 0:
                score += min(category_score / len(keywords), 0.5)

        signal.signals_found = signals_found
        return min(score, 1.0)

    def _calculate_urgency_score(self, signal: RedditSignal) -> float:
        """Calculate urgency based on recency and engagement velocity"""

        # Age factor (newer is better)
        age_hours = (time.time() - signal.created_utc) / 3600
        age_score = max(0, 1 - (age_hours / 24))  # Decay over 24 hours

        # Engagement velocity (comments per hour)
        if age_hours > 0:
            velocity = signal.num_comments / age_hours
            velocity_score = min(velocity / 10, 1.0)  # Normalize
        else:
            velocity_score = 0

        return 0.7 * age_score + 0.3 * velocity_score


# Example usage and testing
async def main():
    """Example usage of RedditSignalsScout"""

    scout = RedditSignalsScout()
    await scout.start()

    try:
        print("🔍 Discovering Reddit signals...")
        signals = await scout.discover_signals(
            subreddits=['entrepreneur', 'startups'],
            timeframe='day',
            limit=10
        )

        print(f"\n📊 Found {len(signals)} signals:")
        for i, signal in enumerate(signals[:5], 1):
            print(f"\n{i}. {signal.title}")
            print(f"   r/{signal.subreddit} | Score: {signal.score} | Comments: {signal.num_comments}")
            print(f"   Opportunity: {signal.opportunity_score:.2f} | Final: {signal.final_score:.2f}")
            if signal.signals_found:
                print(f"   Signals: {', '.join(signal.signals_found[:3])}")

    finally:
        await scout.stop()


if __name__ == "__main__":
    asyncio.run(main())