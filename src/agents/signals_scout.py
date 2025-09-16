"""
Signals Scout Agent: Python Implementation

This agent demonstrates modern Python patterns for autonomous agents:
- Async/await for I/O operations
- Type hints for clarity and IDE support
- Dataclasses for clean data structures
- Context managers for resource management

Learning Focus: How Python's async ecosystem and type system
create more maintainable and debuggable agent code.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
import json
import random
import math


@dataclass
class Signal:
    """A market signal discovered by the agent"""
    id: str
    source: str
    title: str
    content: str
    url: str
    timestamp: datetime
    engagement: Dict[str, int] = field(default_factory=dict)
    score: float = 0.0
    confidence: float = 0.0
    category: str = 'general'
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoringFactors:
    """Factors that contribute to signal scoring"""
    engagement: float = 0.0
    novelty: float = 0.0
    market_size: float = 0.0
    urgency: float = 0.0
    feasibility: float = 0.0


@dataclass
class AgentMetrics:
    """Performance metrics for the agent"""
    signals_found: int = 0
    accuracy: float = 0.0
    average_score: float = 0.0
    processing_time: float = 0.0
    last_scan: Optional[datetime] = None


class SignalsScoutAgent:
    """
    Advanced Python implementation of the Signals Scout

    Learning Note: Python's async/await syntax makes I/O-heavy operations
    like web scraping much more natural than callback-based approaches.
    Type hints provide excellent IDE support and catch errors early.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.id = 'signals-scout'
        self.capabilities = [
            'discover_signals',
            'score_signals',
            'filter_signals',
            'categorize_signals'
        ]

        # Configuration with defaults
        self.config = {
            'scan_interval': 60,          # seconds
            'min_signal_score': 0.3,      # filter threshold
            'max_signals_per_source': 100,
            'sources': ['reddit', 'twitter', 'news', 'patents'],
            'timeframe_hours': 24,
            'parallel_sources': True,     # scan sources concurrently
            'cache_ttl': 300,            # 5 minutes
            **(config or {})
        }

        # Internal state
        self.state = {
            'is_active': False,
            'discovered_signals': [],
            'patterns': {},              # learned patterns for signal quality
            'source_reliability': {},    # track source quality over time
            'cache': {}                 # simple in-memory cache
        }

        self.metrics = AgentMetrics()

        # Setup logging
        self.logger = logging.getLogger(f'agent.{self.id}')
        self.logger.setLevel(logging.INFO)

        # Initialize mock data sources
        self._init_data_sources()

        self.logger.info(f"🕵️ Signals Scout initialized with sources: {', '.join(self.config['sources'])}")

    async def discover_signals(self, sources: Optional[List[str]] = None,
                             timeframe: str = '24h',
                             min_quality: Optional[float] = None,
                             **kwargs) -> List[Signal]:
        """
        Main discovery method with proper async handling

        Learning Note: Python's async/await makes concurrent I/O operations
        natural and efficient. Type hints make the interface self-documenting.
        """
        start_time = time.time()

        try:
            self.state['is_active'] = True
            sources = sources or self.config['sources']
            min_quality = min_quality or self.config['min_signal_score']

            self.logger.info(f"🔍 Starting signal discovery: sources={sources}, timeframe={timeframe}")

            # Parse timeframe
            timeframe_hours = self._parse_timeframe(timeframe)

            # Discover signals from sources
            if self.config['parallel_sources']:
                # Concurrent execution for better performance
                tasks = [self._scan_source(source, timeframe_hours) for source in sources]
                raw_signal_sets = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                # Sequential execution
                raw_signal_sets = []
                for source in sources:
                    try:
                        signals = await self._scan_source(source, timeframe_hours)
                        raw_signal_sets.append(signals)
                    except Exception as e:
                        self.logger.warning(f"Failed to scan {source}: {str(e)}")
                        raw_signal_sets.append([])

            # Aggregate successful results
            all_raw_signals = []
            for result in raw_signal_sets:
                if isinstance(result, Exception):
                    self.logger.warning(f"Source scan failed: {str(result)}")
                elif isinstance(result, list):
                    all_raw_signals.extend(result)

            self.logger.info(f"📡 Discovered {len(all_raw_signals)} raw signals")

            # Apply intelligence pipeline
            scored_signals = await self._score_signals(all_raw_signals)
            filtered_signals = self._filter_signals(scored_signals, min_quality)
            categorized_signals = self._categorize_signals(filtered_signals)

            # Update learning and metrics
            self._update_learning_patterns(categorized_signals)
            self._update_metrics(categorized_signals, time.time() - start_time)

            # Limit results
            final_signals = categorized_signals[:self.config['max_signals_per_source']]

            self.logger.info(f"✅ Processed {len(final_signals)} high-quality signals")
            return final_signals

        except Exception as error:
            self.logger.error(f"❌ Signal discovery failed: {str(error)}")
            raise

        finally:
            self.state['is_active'] = False

    async def _scan_source(self, source: str, timeframe_hours: float) -> List[Signal]:
        """
        Scan a specific data source for signals

        Learning Note: Async I/O with proper error handling and caching
        improves both performance and reliability.
        """
        cache_key = f"{source}:{timeframe_hours}"

        # Check cache first
        if cache_key in self.state['cache']:
            cache_entry = self.state['cache'][cache_key]
            if (datetime.now() - cache_entry['timestamp']).total_seconds() < self.config['cache_ttl']:
                self.logger.debug(f"📦 Cache hit for {source}")
                return cache_entry['data']

        self.logger.info(f"🔍 Scanning {source} for signals ({timeframe_hours}h)")

        # Simulate API latency with jitter
        await asyncio.sleep(random.uniform(0.5, 2.0))

        try:
            # Get source data
            source_data = self._get_source_data(source)

            # Filter by timeframe
            cutoff_time = datetime.now() - timedelta(hours=timeframe_hours)
            recent_data = [
                item for item in source_data
                if datetime.fromisoformat(item['timestamp']) > cutoff_time
            ]

            # Convert to Signal objects
            signals = [self._create_signal(item, source) for item in recent_data]

            # Cache results
            self.state['cache'][cache_key] = {
                'data': signals,
                'timestamp': datetime.now()
            }

            self.logger.info(f"📊 Found {len(signals)} signals from {source}")
            return signals

        except Exception as error:
            self.logger.error(f"Failed to scan {source}: {str(error)}")
            return []

    async def _score_signals(self, signals: List[Signal]) -> List[Signal]:
        """
        Score signals based on learned patterns and heuristics

        Learning Note: Async processing allows for complex scoring that might
        involve external APIs (sentiment analysis, market data, etc.)
        """
        self.logger.info(f"🎯 Scoring {len(signals)} signals")

        scored_signals = []

        for signal in signals:
            try:
                # Calculate scoring factors
                factors = await self._calculate_scoring_factors(signal)

                # Weighted combination
                score = (
                    factors.engagement * 0.25 +
                    factors.novelty * 0.20 +
                    factors.market_size * 0.25 +
                    factors.urgency * 0.15 +
                    factors.feasibility * 0.15
                )

                # Apply learned patterns
                pattern_bonus = self._apply_learned_patterns(signal)
                score = min(1.0, score + pattern_bonus)

                # Calculate confidence based on factor consistency
                confidence = self._calculate_confidence(factors)

                # Update signal
                signal.score = score
                signal.confidence = confidence
                signal.metadata['scoring_factors'] = factors.__dict__

                scored_signals.append(signal)

            except Exception as error:
                self.logger.warning(f"Failed to score signal {signal.id}: {str(error)}")
                signal.score = 0.0
                signal.confidence = 0.0
                scored_signals.append(signal)

        return scored_signals

    async def _calculate_scoring_factors(self, signal: Signal) -> ScoringFactors:
        """Calculate detailed scoring factors for a signal"""

        # In a real implementation, these could be async calls to external services
        factors = ScoringFactors()

        # Engagement scoring
        engagement = signal.engagement
        total_engagement = sum(engagement.values())
        factors.engagement = min(1.0, total_engagement / 1000)  # Normalize to 0-1

        # Novelty scoring (check against recent signals)
        factors.novelty = await self._calculate_novelty(signal)

        # Market size indicators
        factors.market_size = self._assess_market_potential(signal)

        # Urgency indicators
        factors.urgency = self._assess_urgency(signal)

        # Feasibility assessment
        factors.feasibility = self._assess_feasibility(signal)

        return factors

    async def _calculate_novelty(self, signal: Signal) -> float:
        """Calculate how novel this signal is compared to recent discoveries"""
        recent_signals = self.state['discovered_signals'][-100:]  # Last 100 signals

        if not recent_signals:
            return 1.0  # First signal is perfectly novel

        # Simple similarity calculation (in production, use proper NLP)
        similarities = []
        for recent in recent_signals:
            similarity = self._calculate_similarity(signal.content, recent.content)
            similarities.append(similarity)

        max_similarity = max(similarities) if similarities else 0.0
        return max(0.1, 1.0 - max_similarity)  # Higher novelty = lower similarity

    def _filter_signals(self, signals: List[Signal], min_score: float) -> List[Signal]:
        """Filter signals based on quality thresholds"""
        filtered = [s for s in signals if s.score >= min_score]
        self.logger.info(f"🔍 Filtered {len(signals)} → {len(filtered)} signals (min score: {min_score})")
        return filtered

    def _categorize_signals(self, signals: List[Signal]) -> List[Signal]:
        """Categorize signals into business domains"""
        for signal in signals:
            signal.category = self._determine_category(signal)
            signal.tags = self._extract_tags(signal)

        return signals

    def _determine_category(self, signal: Signal) -> str:
        """Determine the business category for a signal"""
        content = signal.content.lower()

        if any(word in content for word in ['ai', 'ml', 'artificial intelligence', 'machine learning']):
            return 'ai-tech'
        elif any(word in content for word in ['health', 'medical', 'healthcare']):
            return 'healthcare'
        elif any(word in content for word in ['finance', 'fintech', 'banking', 'payment']):
            return 'fintech'
        elif any(word in content for word in ['education', 'learning', 'edtech']):
            return 'edtech'
        elif any(word in content for word in ['climate', 'sustainability', 'green', 'renewable']):
            return 'climate-tech'
        else:
            return 'general'

    def _extract_tags(self, signal: Signal) -> List[str]:
        """Extract relevant tags from signal content"""
        content = signal.content.lower()
        tags = []

        # Technology tags
        if 'saas' in content: tags.append('saas')
        if 'api' in content: tags.append('api')
        if 'mobile' in content: tags.append('mobile')
        if 'blockchain' in content: tags.append('blockchain')

        # Business model tags
        if any(word in content for word in ['b2b', 'enterprise']): tags.append('b2b')
        if any(word in content for word in ['b2c', 'consumer']): tags.append('b2c')
        if 'marketplace' in content: tags.append('marketplace')

        # Urgency tags
        if any(word in content for word in ['urgent', 'crisis', 'critical']): tags.append('urgent')
        if any(word in content for word in ['trend', 'trending', 'viral']): tags.append('trending')

        return tags

    def _assess_market_potential(self, signal: Signal) -> float:
        """Assess the market potential indicated by the signal"""
        content = signal.content.lower()
        score = 0.3  # Base score

        # Market size indicators
        if any(word in content for word in ['billion', 'market', 'industry']):
            score += 0.3
        if any(word in content for word in ['enterprise', 'b2b', 'business']):
            score += 0.2
        if any(word in content for word in ['global', 'worldwide', 'international']):
            score += 0.1
        if any(word in content for word in ['niche', 'specific', 'specialized']):
            score -= 0.1

        return min(1.0, max(0.1, score))

    def _assess_urgency(self, signal: Signal) -> float:
        """Assess the urgency indicated by the signal"""
        content = signal.content.lower()
        urgent_words = ['urgent', 'immediate', 'now', 'crisis', 'critical', 'emergency', 'asap']

        urgent_count = sum(1 for word in urgent_words if word in content)
        return min(1.0, urgent_count * 0.25)

    def _assess_feasibility(self, signal: Signal) -> float:
        """Assess how feasible a solution might be"""
        content = signal.content.lower()
        score = 0.5  # Neutral starting point

        # Positive indicators
        if any(word in content for word in ['simple', 'easy', 'straightforward']):
            score += 0.3
        if any(word in content for word in ['existing', 'available', 'ready']):
            score += 0.2

        # Negative indicators
        if any(word in content for word in ['complex', 'complicated', 'difficult']):
            score -= 0.3
        if any(word in content for word in ['impossible', 'unrealistic', 'unfeasible']):
            score -= 0.4

        return min(1.0, max(0.1, score))

    def _apply_learned_patterns(self, signal: Signal) -> float:
        """Apply learned patterns for signal quality prediction"""
        pattern = self._extract_pattern(signal)
        learned = self.state['patterns'].get(pattern)

        if learned and learned.get('count', 0) > 5:
            # Boost/penalty based on historical performance
            avg_score = learned.get('avg_score', 0.5)
            return (avg_score - 0.5) * 0.1  # Max 5% bonus/penalty

        return 0.0

    def _calculate_confidence(self, factors: ScoringFactors) -> float:
        """Calculate confidence based on scoring factor consistency"""
        factor_values = [factors.engagement, factors.novelty, factors.market_size,
                        factors.urgency, factors.feasibility]

        # Confidence is higher when factors are more consistent
        mean_score = sum(factor_values) / len(factor_values)
        variance = sum((f - mean_score) ** 2 for f in factor_values) / len(factor_values)

        # Lower variance = higher confidence
        return max(0.1, 1.0 - variance)

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity (Jaccard)"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def _update_learning_patterns(self, signals: List[Signal]) -> None:
        """Update learned patterns based on signal outcomes"""
        for signal in signals:
            pattern = self._extract_pattern(signal)

            if pattern not in self.state['patterns']:
                self.state['patterns'][pattern] = {'count': 0, 'avg_score': 0.0}

            pattern_data = self.state['patterns'][pattern]
            pattern_data['count'] += 1
            pattern_data['avg_score'] = (
                (pattern_data['avg_score'] * (pattern_data['count'] - 1) + signal.score) /
                pattern_data['count']
            )

        self.logger.info(f"🧠 Updated learning patterns: {len(self.state['patterns'])} patterns tracked")

    def _extract_pattern(self, signal: Signal) -> str:
        """Extract a pattern key from a signal for learning"""
        # Simple pattern: category + primary keywords
        words = signal.content.lower().split()
        keywords = [w for w in words if len(w) > 4][:3]  # Top 3 meaningful words
        pattern = f"{signal.category}:{'-'.join(sorted(keywords))}"
        return pattern

    def _update_metrics(self, signals: List[Signal], processing_time: float) -> None:
        """Update agent performance metrics"""
        self.metrics.signals_found += len(signals)
        self.metrics.processing_time = processing_time
        self.metrics.last_scan = datetime.now()

        if signals:
            self.metrics.average_score = sum(s.score for s in signals) / len(signals)

        # In production, accuracy would be calculated based on feedback
        self.metrics.accuracy = 0.75  # Placeholder

    def _create_signal(self, item: Dict[str, Any], source: str) -> Signal:
        """Create a Signal object from raw data"""
        return Signal(
            id=f"{source}-{item['id']}",
            source=source,
            title=item['title'],
            content=item['content'],
            url=item['url'],
            timestamp=datetime.fromisoformat(item['timestamp']),
            engagement=item.get('engagement', {}),
            metadata={'extracted_at': datetime.now().isoformat()}
        )

    def _parse_timeframe(self, timeframe: str) -> float:
        """Parse timeframe string to hours"""
        if timeframe.endswith('h'):
            return float(timeframe[:-1])
        elif timeframe.endswith('d'):
            return float(timeframe[:-1]) * 24
        elif timeframe.endswith('w'):
            return float(timeframe[:-1]) * 24 * 7
        else:
            return 24.0  # Default to 24 hours

    def _get_source_data(self, source: str) -> List[Dict[str, Any]]:
        """Get mock data for a source (replace with real APIs in production)"""
        return self.mock_data_sources.get(source, [])

    def _init_data_sources(self) -> None:
        """Initialize mock data sources"""
        base_time = datetime.now()

        self.mock_data_sources = {
            'reddit': [
                {
                    'id': 'r1',
                    'title': 'Why is there no good solution for healthcare data integration?',
                    'content': 'Working in healthcare and we desperately need a simple way to manage patient data across multiple EMR systems. Current solutions are expensive and complex.',
                    'url': 'https://reddit.com/r/healthcare/post1',
                    'timestamp': (base_time - timedelta(hours=2)).isoformat(),
                    'engagement': {'likes': 150, 'comments': 45, 'shares': 12}
                },
                {
                    'id': 'r2',
                    'title': 'AI automation for small businesses is broken',
                    'content': 'Small businesses need AI tools that work out of the box. Most enterprise AI is too complex and expensive for SMBs.',
                    'url': 'https://reddit.com/r/startups/post2',
                    'timestamp': (base_time - timedelta(hours=5)).isoformat(),
                    'engagement': {'likes': 89, 'comments': 23, 'shares': 7}
                },
                {
                    'id': 'r3',
                    'title': 'Climate tech funding surge creates opportunities',
                    'content': 'Venture funding for climate tech startups reached $16B this quarter. Huge opportunity for carbon tracking and renewable energy solutions.',
                    'url': 'https://reddit.com/r/climatetech/post3',
                    'timestamp': (base_time - timedelta(hours=1)).isoformat(),
                    'engagement': {'likes': 234, 'comments': 67, 'shares': 89}
                }
            ],
            'twitter': [
                {
                    'id': 't1',
                    'title': 'Thread: Developer productivity crisis',
                    'content': 'Why is there still no good way to monitor microservices in real-time? Existing tools are either too expensive or too complex for small teams.',
                    'url': 'https://twitter.com/dev/status1',
                    'timestamp': (base_time - timedelta(hours=1)).isoformat(),
                    'engagement': {'likes': 234, 'comments': 67, 'shares': 89}
                },
                {
                    'id': 't2',
                    'title': 'EdTech revolution in remote learning',
                    'content': 'Post-pandemic education needs better remote collaboration tools. Students and teachers struggling with current platforms.',
                    'url': 'https://twitter.com/edtech/status2',
                    'timestamp': (base_time - timedelta(hours=3)).isoformat(),
                    'engagement': {'likes': 156, 'comments': 34, 'shares': 22}
                }
            ],
            'news': [
                {
                    'id': 'n1',
                    'title': 'New regulation creates compliance burden for startups',
                    'content': 'Recent financial regulations require small businesses to implement complex reporting systems, creating demand for simplified compliance tools.',
                    'url': 'https://news.com/article1',
                    'timestamp': (base_time - timedelta(hours=3)).isoformat(),
                    'engagement': {'likes': 45, 'comments': 12, 'shares': 23}
                },
                {
                    'id': 'n2',
                    'title': 'Supply chain disruptions drive automation demand',
                    'content': 'Global supply chain issues are forcing companies to invest in AI-powered logistics and inventory management solutions.',
                    'url': 'https://news.com/article2',
                    'timestamp': (base_time - timedelta(hours=6)).isoformat(),
                    'engagement': {'likes': 78, 'comments': 15, 'shares': 31}
                }
            ],
            'patents': [
                {
                    'id': 'p1',
                    'title': 'Method for automated document processing using ML',
                    'content': 'System and method for using machine learning to automatically extract structured data from unstructured documents in enterprise environments.',
                    'url': 'https://patents.com/patent1',
                    'timestamp': (base_time - timedelta(hours=12)).isoformat(),
                    'engagement': {'likes': 5, 'comments': 1, 'shares': 2}
                }
            ]
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            'id': self.id,
            'active': self.state['is_active'],
            'metrics': {
                'signals_found': self.metrics.signals_found,
                'accuracy': self.metrics.accuracy,
                'average_score': self.metrics.average_score,
                'processing_time': self.metrics.processing_time,
                'last_scan': self.metrics.last_scan.isoformat() if self.metrics.last_scan else None
            },
            'learning': {
                'patterns_learned': len(self.state['patterns']),
                'cache_size': len(self.state['cache'])
            },
            'config': self.config
        }