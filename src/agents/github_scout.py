"""
GitHub Trending Scout Agent

This agent monitors GitHub for trending repositories to identify emerging technology
opportunities and early-stage adoption signals.

Features:
- Trending repository discovery across languages and timeframes
- Repository growth metrics (stars, forks, contributors)
- Technology stack analysis and opportunity assessment
- Integration with existing signal database
- Async operations for performance

Data Sources:
- GitHub Trending API (unofficial)
- GitHub API for detailed repository metrics
- Repository content analysis for business potential
"""

import asyncio
import aiohttp
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import re

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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
class GitHubRepo:
    """GitHub repository data"""
    name: str
    full_name: str
    description: str
    url: str
    language: str
    stars: int
    forks: int
    created_at: str
    updated_at: str
    topics: List[str]
    license: Optional[str]
    contributors_count: Optional[int] = None
    commits_last_month: Optional[int] = None
    issues_count: Optional[int] = None
    pull_requests_count: Optional[int] = None


class GitHubScout:
    """
    GitHub trending repository scout for tech opportunity discovery

    Monitors trending repositories to identify:
    - Emerging technology patterns
    - Early-stage adoption signals
    - Open source business opportunities
    - Developer tool innovations
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # GitHub API configuration
        self.github_token = os.getenv('GITHUB_TOKEN', '')  # Optional for higher rate limits
        self.session: Optional[aiohttp.ClientSession] = None

        # Rate limiting
        self.requests_per_hour = int(os.getenv('GITHUB_REQUESTS_PER_HOUR', '60'))
        self.last_request_time = datetime.utcnow()

        # Business opportunity keywords
        self.business_keywords = [
            'saas', 'platform', 'api', 'framework', 'tool', 'service',
            'automation', 'dashboard', 'analytics', 'monitoring', 'security',
            'ai', 'ml', 'machine-learning', 'artificial-intelligence',
            'blockchain', 'crypto', 'web3', 'defi', 'nft',
            'mobile', 'ios', 'android', 'react-native', 'flutter',
            'web', 'frontend', 'backend', 'fullstack', 'javascript',
            'python', 'golang', 'rust', 'typescript', 'node',
            'docker', 'kubernetes', 'cloud', 'aws', 'azure', 'gcp',
            'database', 'sql', 'nosql', 'redis', 'mongodb',
            'startup', 'business', 'commerce', 'fintech', 'healthtech'
        ]

        # Trending categories to monitor
        self.trending_periods = ['daily', 'weekly', 'monthly']
        self.languages = ['', 'javascript', 'python', 'typescript', 'go', 'rust', 'java']

    async def start(self):
        """Initialize the GitHub scout"""
        headers = {
            'User-Agent': 'agentic-venture-studio/1.0',
            'Accept': 'application/vnd.github.v3+json'
        }

        if self.github_token:
            headers['Authorization'] = f'token {self.github_token}'

        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
        timeout = aiohttp.ClientTimeout(total=30, connect=10)

        self.session = aiohttp.ClientSession(
            headers=headers,
            connector=connector,
            timeout=timeout
        )

        logger.info("✅ GitHub Scout initialized")

    async def stop(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
            logger.info("🔌 GitHub Scout session closed")

    async def discover_trending_repos(self,
                                    period: str = 'daily',
                                    language: str = '',
                                    limit: Optional[int] = None) -> List[GitHubRepo]:
        """
        Discover trending repositories from GitHub

        Args:
            period: 'daily', 'weekly', or 'monthly'
            language: Programming language filter (empty for all)
            limit: Maximum number of repos to return

        Returns:
            List of GitHubRepo objects
        """

        if not self.session:
            raise RuntimeError("Scout not started. Call start() first.")

        try:
            # Use GitHub Search API to find trending repos
            # We'll search for recently created repos with high star velocity
            query_parts = []

            # Date range based on period
            if period == 'daily':
                since_date = (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d')
                query_parts.append(f'created:>{since_date}')
            elif period == 'weekly':
                since_date = (datetime.utcnow() - timedelta(days=7)).strftime('%Y-%m-%d')
                query_parts.append(f'created:>{since_date}')
            else:  # monthly
                since_date = (datetime.utcnow() - timedelta(days=30)).strftime('%Y-%m-%d')
                query_parts.append(f'created:>{since_date}')

            # Language filter
            if language:
                query_parts.append(f'language:{language}')

            # Minimum stars to filter quality
            query_parts.append('stars:>10')

            query = ' '.join(query_parts)

            url = 'https://api.github.com/search/repositories'
            params = {
                'q': query,
                'sort': 'stars',
                'order': 'desc',
                'per_page': limit or 25
            }

            logger.info(f"🔍 Searching GitHub trending ({period}, {language or 'all languages'})")

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    repos = []

                    for item in data.get('items', []):
                        repo = GitHubRepo(
                            name=item['name'],
                            full_name=item['full_name'],
                            description=item.get('description') or '',
                            url=item['html_url'],
                            language=item.get('language') or '',
                            stars=item['stargazers_count'],
                            forks=item['forks_count'],
                            created_at=item['created_at'],
                            updated_at=item['updated_at'],
                            topics=item.get('topics', []),
                            license=item.get('license', {}).get('name') if item.get('license') else None
                        )
                        repos.append(repo)

                    logger.info(f"📊 Found {len(repos)} trending repositories")
                    return repos

                elif response.status == 403:
                    logger.warning("⚠️ GitHub API rate limit hit")
                    return []
                else:
                    logger.error(f"❌ GitHub API error {response.status}")
                    return []

        except Exception as e:
            logger.error(f"Error discovering trending repos: {e}")
            return []

    async def enrich_repo_data(self, repo: GitHubRepo) -> GitHubRepo:
        """
        Enrich repository with additional metrics

        Args:
            repo: Basic repository data

        Returns:
            Enriched repository data
        """

        if not self.session:
            return repo

        try:
            # Get additional repository details
            api_url = f"https://api.github.com/repos/{repo.full_name}"

            async with self.session.get(api_url) as response:
                if response.status == 200:
                    data = await response.json()

                    # Update with additional data
                    repo.issues_count = data.get('open_issues_count', 0)

            # Get contributor count
            contributors_url = f"https://api.github.com/repos/{repo.full_name}/contributors"
            async with self.session.get(contributors_url, params={'per_page': 1}) as response:
                if response.status == 200:
                    # GitHub returns link header with pagination info
                    link_header = response.headers.get('Link', '')
                    if 'last' in link_header:
                        # Parse last page number from link header
                        import re
                        match = re.search(r'page=(\d+)>; rel="last"', link_header)
                        if match:
                            repo.contributors_count = int(match.group(1))
                    else:
                        # Single page of contributors
                        contributors = await response.json()
                        repo.contributors_count = len(contributors)

        except Exception as e:
            logger.warning(f"Failed to enrich repo {repo.full_name}: {e}")

        return repo

    def calculate_business_potential(self, repo: GitHubRepo) -> float:
        """
        Calculate business potential score for a repository

        Args:
            repo: Repository data

        Returns:
            Business potential score (0-1)
        """

        score = 0.0

        # Base score from popularity metrics
        star_score = min(repo.stars / 1000, 0.3)  # Up to 0.3 for 1000+ stars
        fork_score = min(repo.forks / 200, 0.1)   # Up to 0.1 for 200+ forks

        score += star_score + fork_score

        # Business keyword matching
        description = repo.description or ""
        text_to_check = f"{repo.name} {description} {' '.join(repo.topics)}".lower()

        keyword_matches = sum(1 for keyword in self.business_keywords
                            if keyword in text_to_check)
        keyword_score = min(keyword_matches * 0.05, 0.3)  # Up to 0.3 for keyword relevance

        score += keyword_score

        # Language relevance (business-oriented languages)
        business_languages = {
            'javascript': 0.15, 'typescript': 0.15, 'python': 0.1,
            'go': 0.1, 'rust': 0.1, 'java': 0.08, 'kotlin': 0.08,
            'swift': 0.08, 'dart': 0.08
        }

        if repo.language:
            lang_lower = repo.language.lower()
            if lang_lower in business_languages:
                score += business_languages[lang_lower]

        # Recency bonus (newer repos get slight boost)
        try:
            created = datetime.fromisoformat(repo.created_at.replace('Z', '+00:00'))
            days_old = (datetime.utcnow().replace(tzinfo=created.tzinfo) - created).days
            if days_old < 30:
                score += 0.1  # Recent creation bonus
        except:
            pass

        # Topic relevance
        business_topics = {
            'api', 'framework', 'library', 'tool', 'saas', 'platform',
            'automation', 'ai', 'machine-learning', 'blockchain', 'web3',
            'mobile', 'web', 'frontend', 'backend', 'cloud', 'docker',
            'kubernetes', 'database', 'security', 'monitoring'
        }

        topic_matches = sum(1 for topic in repo.topics if topic in business_topics)
        score += min(topic_matches * 0.05, 0.15)

        return min(score, 1.0)

    def repo_to_signal(self, repo: GitHubRepo) -> Signal:
        """
        Convert GitHub repository to Signal object

        Args:
            repo: Repository data

        Returns:
            Signal object for database storage
        """

        # Calculate scores
        business_potential = self.calculate_business_potential(repo)

        # Generate signal content
        content = f"""
**Repository:** {repo.full_name}
**Description:** {repo.description}
**Language:** {repo.language}
**Stars:** {repo.stars:,} | **Forks:** {repo.forks:,}
**Created:** {repo.created_at}
**Topics:** {', '.join(repo.topics) if repo.topics else 'None'}
**License:** {repo.license or 'Not specified'}

**Growth Indicators:**
- Star velocity: High (trending)
- Community engagement: {repo.contributors_count or 'Unknown'} contributors
- Issues: {repo.issues_count or 'Unknown'} open issues

**Business Potential Analysis:**
This repository shows potential for commercialization based on:
- Technology relevance and modern stack
- Community adoption and growth
- Business-applicable use cases
- Developer tool or platform opportunity

**URL:** {repo.url}
        """.strip()

        # Determine opportunity signals
        signals_found = []

        if repo.stars > 100:
            signals_found.append('high-community-interest')
        if repo.forks > 20:
            signals_found.append('active-development')
        if any(keyword in repo.description.lower() for keyword in ['api', 'framework', 'tool']):
            signals_found.append('developer-tool-opportunity')
        if any(keyword in repo.description.lower() for keyword in ['ai', 'ml', 'machine']):
            signals_found.append('ai-technology-trend')
        if repo.language in ['javascript', 'typescript', 'python', 'go']:
            signals_found.append('business-relevant-tech')

        # Keywords for search
        keywords = [repo.language] if repo.language else []
        keywords.extend(repo.topics[:5])  # Limit topics
        keywords.extend(['github', 'trending', 'open-source'])

        return Signal(
            id=f"github-{repo.full_name.replace('/', '-')}-{datetime.utcnow().strftime('%Y%m%d')}",
            source='github',
            title=f"Trending: {repo.name} - {repo.description[:50]}{'...' if len(repo.description) > 50 else ''}",
            content=content,
            url=repo.url,
            agent_id='github-scout',
            opportunity_score=business_potential,
            engagement_score=min(repo.stars / 1000, 1.0),  # Normalized star count
            final_score=business_potential * 0.7 + min(repo.stars / 1000, 1.0) * 0.3,
            keywords=keywords,
            signals_found=signals_found,
            source_metadata={
                'github_stars': repo.stars,
                'github_forks': repo.forks,
                'github_language': repo.language,
                'github_topics': repo.topics,
                'github_contributors': repo.contributors_count,
                'github_created': repo.created_at
            }
        )

    async def discover_signals(self,
                             periods: Optional[List[str]] = None,
                             languages: Optional[List[str]] = None,
                             limit: Optional[int] = None) -> List[Signal]:
        """
        Discover GitHub signals and convert to Signal objects

        Args:
            periods: List of periods to check ('daily', 'weekly', 'monthly')
            languages: List of languages to monitor
            limit: Maximum signals per period/language combination

        Returns:
            List of Signal objects
        """

        periods = periods or ['daily', 'weekly']
        languages = languages or ['', 'javascript', 'python', 'typescript']
        limit = limit or 5

        all_signals = []

        try:
            # Get signals from database to avoid duplicates
            await init_database()
            signal_repo = get_signal_repository()

            for period in periods:
                for language in languages:
                    try:
                        repos = await self.discover_trending_repos(
                            period=period,
                            language=language,
                            limit=limit
                        )

                        for repo in repos:
                            # Enrich with additional data
                            enriched_repo = await self.enrich_repo_data(repo)

                            # Convert to signal
                            signal = self.repo_to_signal(enriched_repo)

                            # Check for duplicates (simplified - just create new signals)
                            try:
                                # Save to database using correct method signature
                                await signal_repo.create_signal(
                                    signal_id=signal.id,
                                    source=signal.source,
                                    title=signal.title,
                                    content=signal.content,
                                    url=signal.url,
                                    agent_id=signal.agent_id,
                                    opportunity_score=signal.opportunity_score,
                                    engagement_score=signal.engagement_score,
                                    final_score=signal.final_score,
                                    keywords=signal.keywords,
                                    signals_found=signal.signals_found,
                                    source_metadata=signal.source_metadata
                                )
                                all_signals.append(signal)
                                logger.info(f"💾 Saved GitHub signal: {signal.title}")
                            except Exception as e:
                                logger.warning(f"Failed to save signal {repo.full_name}: {e}")
                                continue

                        # Rate limiting pause
                        await asyncio.sleep(1)

                    except Exception as e:
                        logger.error(f"Error processing {period}/{language}: {e}")
                        continue

        except Exception as e:
            logger.error(f"Error in GitHub signal discovery: {e}")

        logger.info(f"✅ GitHub discovery complete: {len(all_signals)} new signals")
        return all_signals


# Example usage and testing
async def main():
    """Demo the GitHub scout agent"""

    print("🐙 GitHub Trending Scout Demo")
    print("=" * 40)

    scout = GitHubScout()

    try:
        await scout.start()

        print("🔍 Discovering trending GitHub repositories...")
        signals = await scout.discover_signals(
            periods=['daily', 'weekly'],
            languages=['', 'python', 'javascript'],
            limit=3
        )

        if signals:
            print(f"\n📊 Found {len(signals)} GitHub signals:")
            print("-" * 60)

            for i, signal in enumerate(signals, 1):
                print(f"\n{i}. {signal.title}")
                print(f"   Score: {signal.final_score:.2f}")
                print(f"   Language: {signal.metadata.get('github_language', 'Unknown')}")
                print(f"   Stars: {signal.metadata.get('github_stars', 0):,}")
                print(f"   URL: {signal.url}")

                if signal.signals_found:
                    print(f"   Signals: {', '.join(signal.signals_found)}")
        else:
            print("ℹ️  No new GitHub signals discovered")

    except Exception as e:
        print(f"❌ Demo failed: {e}")

    finally:
        await scout.stop()


if __name__ == "__main__":
    asyncio.run(main())