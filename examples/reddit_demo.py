#!/usr/bin/env python3
"""
Reddit Integration Demo: Real Business Signal Discovery

This demo showcases Reddit API integration for discovering real business signals:
- Live Reddit data from entrepreneurship subreddits
- Intelligent signal scoring and ranking
- Opportunity keyword detection
- Rate limiting and error handling

Run this to see real Reddit signals being discovered and scored!
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from agents.reddit_signals_scout import RedditSignalsScout
from agents.thesis_synthesizer import ThesisSynthesizerAgent
from observability.telemetry import AgentTelemetrySystem, TelemetryConfig


async def demo_reddit_signals():
    """Demo 1: Basic Reddit signal discovery"""

    print("🔍 Demo 1: Reddit Signal Discovery")
    print("=" * 50)

    scout = RedditSignalsScout()
    await scout.start()

    try:
        # Discover signals from entrepreneur subreddits
        signals = await scout.discover_signals(
            subreddits=['entrepreneur', 'startups', 'SaaS'],
            timeframe='day',
            limit=15
        )

        print(f"\n📊 Found {len(signals)} business signals:")
        print("-" * 50)

        for i, signal in enumerate(signals[:8], 1):
            print(f"\n{i}. {signal.title}")
            print(f"   📍 r/{signal.subreddit} by u/{signal.author}")
            print(f"   ⬆️  {signal.score} upvotes | 💬 {signal.num_comments} comments")
            print(f"   🎯 Opportunity: {signal.opportunity_score:.2f} | Final: {signal.final_score:.2f}")

            if signal.signals_found:
                print(f"   🚨 Key signals: {', '.join(signal.signals_found[:3])}")

            if signal.content:
                preview = signal.content[:100].replace('\n', ' ')
                print(f"   📝 \"{preview}...\"")

        print(f"\n✅ Reddit integration successful!")

        return signals

    finally:
        await scout.stop()


async def demo_signal_analysis():
    """Demo 2: Advanced signal analysis and scoring"""

    print("\n\n🧠 Demo 2: Signal Analysis & Scoring")
    print("=" * 50)

    scout = RedditSignalsScout()
    await scout.start()

    try:
        # Get signals from multiple timeframes
        print("Analyzing signals from different timeframes...")

        day_signals = await scout.discover_signals(['entrepreneur'], 'day', 10)
        week_signals = await scout.discover_signals(['entrepreneur'], 'week', 10)

        print(f"\n📈 Signal Analysis:")
        print(f"   Day signals: {len(day_signals)} (avg score: {sum(s.final_score for s in day_signals)/len(day_signals):.2f})")
        print(f"   Week signals: {len(week_signals)} (avg score: {sum(s.final_score for s in week_signals)/len(week_signals):.2f})")

        # Analyze top opportunities
        all_signals = day_signals + week_signals
        top_opportunities = [s for s in all_signals if s.opportunity_score > 0.3]

        print(f"\n🎯 High-Opportunity Signals ({len(top_opportunities)}):")
        print("-" * 40)

        for signal in sorted(top_opportunities, key=lambda s: s.opportunity_score, reverse=True)[:5]:
            print(f"\n• {signal.title[:60]}...")
            print(f"  Opportunity: {signal.opportunity_score:.2f} | Engagement: {signal.engagement_score:.2f}")
            print(f"  Signals: {', '.join(signal.signals_found)}")

        return top_opportunities

    finally:
        await scout.stop()


async def demo_integrated_workflow():
    """Demo 3: Full workflow with Reddit + Thesis Synthesis"""

    print("\n\n🤖 Demo 3: Integrated Multi-Agent Workflow")
    print("=" * 50)

    # Setup telemetry
    telemetry = AgentTelemetrySystem(TelemetryConfig())

    # Create agents
    reddit_scout = RedditSignalsScout()
    synthesizer = ThesisSynthesizerAgent({
        'min_cluster_size': 2,
        'similarity_threshold': 0.4,
        'max_theses_per_cluster': 2
    })

    await reddit_scout.start()
    # Note: ThesisSynthesizerAgent doesn't need start/stop methods

    try:
        # Step 1: Discover Reddit signals
        print("Step 1: Discovering Reddit signals...")
        reddit_signals = await reddit_scout.discover_signals(
            subreddits=['entrepreneur', 'startups'],
            timeframe='day',
            limit=12
        )

        # Convert Reddit signals to standard Signal format for thesis synthesizer
        from agents.signals_scout import Signal
        converted_signals = []

        for rs in reddit_signals:
            signal = Signal(
                id=rs.id,
                source=f"reddit:r/{rs.subreddit}",
                title=rs.title,
                content=rs.content or "",
                url=rs.url or rs.permalink,
                timestamp=rs.timestamp,
                score=rs.final_score
            )
            converted_signals.append(signal)

        print(f"   ✅ Found {len(converted_signals)} Reddit signals")

        # Step 2: Synthesize theses
        print("\nStep 2: Synthesizing business theses...")
        theses = await synthesizer.cluster_signals(converted_signals)

        print(f"   ✅ Generated {len(theses)} business theses")

        # Step 3: Display results
        print("\n📋 Generated Business Theses:")
        print("-" * 40)

        for i, thesis in enumerate(theses, 1):
            print(f"\n{i}. {thesis['hypothesis']}")
            print(f"   Domain: {thesis['domain']}")
            print(f"   Supporting signals: {len(thesis['supporting_signals'])}")
            print(f"   Confidence: {thesis['confidence']:.2f}")
            print(f"   Market evidence: {', '.join(thesis['market_evidence'][:3])}")

        # Performance metrics
        workflow_time = time.time()
        print(f"\n⚡ Workflow Performance:")
        print(f"   Total signals processed: {len(reddit_signals)}")
        print(f"   Business theses generated: {len(theses)}")
        print(f"   High-confidence theses: {len([t for t in theses if t['confidence'] > 0.6])}")

        return {
            'signals': reddit_signals,
            'theses': theses,
            'performance': {
                'signals_count': len(reddit_signals),
                'theses_count': len(theses),
                'high_confidence_count': len([t for t in theses if t['confidence'] > 0.6])
            }
        }

    finally:
        await reddit_scout.stop()
        # Note: ThesisSynthesizerAgent doesn't need stop method


async def main():
    """Run all Reddit integration demos"""

    print("🤖 Multi-Agent Venture Studio: Reddit Integration Demo")
    print("=" * 60)
    print("This demo shows real Reddit API integration for business signal discovery")
    print("No credentials required - using public Reddit API\n")

    start_time = time.time()

    try:
        # Run all demos
        signals = await demo_reddit_signals()
        opportunities = await demo_signal_analysis()
        workflow_results = await demo_integrated_workflow()

        # Summary
        total_time = time.time() - start_time

        print("\n\n🎉 Demo Complete!")
        print("=" * 50)
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Signals discovered: {len(signals)}")
        print(f"High-opportunity signals: {len(opportunities)}")
        print(f"Business theses generated: {workflow_results['performance']['theses_count']}")
        print(f"High-confidence theses: {workflow_results['performance']['high_confidence_count']}")

        print("\n✅ Reddit integration successful! Ready for production use.")

        # Optional: Show setup instructions
        print("\n📋 Next Steps:")
        print("1. Get Reddit API credentials for higher rate limits:")
        print("   https://www.reddit.com/prefs/apps")
        print("2. Copy .env.template to .env and add your credentials")
        print("3. Explore other subreddits for different market signals")
        print("4. Integrate with your business validation pipeline")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Make sure you have internet connection and Reddit is accessible")
        raise


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    asyncio.run(main())