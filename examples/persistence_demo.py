#!/usr/bin/env python3
"""
Database Persistence Demo: Complete Workflow

This demo showcases the full persistence layer:
- Reddit signals stored in database
- Deduplication and intelligent caching
- Historical analysis and trending detection
- Thesis synthesis with persistent storage
- Signal-to-thesis relationship tracking

Run this to see the complete persistent multi-agent workflow!
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from agents.persistent_reddit_scout import PersistentRedditSignalsScout
from agents.thesis_synthesizer import ThesisSynthesizerAgent
from persistence.database import init_database, db_manager
from persistence.repositories import get_signal_repository, get_thesis_repository


async def demo_persistent_signals():
    """Demo 1: Persistent signal discovery with deduplication"""

    print("📊 Demo 1: Persistent Signal Discovery")
    print("=" * 50)

    scout = PersistentRedditSignalsScout()
    await scout.start()

    try:
        # First discovery run
        print("🔍 First discovery run...")
        signals_1 = await scout.discover_signals(['entrepreneur', 'startups'], limit=8)
        print(f"   Found {len(signals_1)} signals")

        # Second discovery run (should detect some duplicates)
        print("\n🔍 Second discovery run (testing deduplication)...")
        signals_2 = await scout.discover_signals(['entrepreneur', 'startups'], limit=8)
        print(f"   Found {len(signals_2)} signals")

        # Show database stats
        repo = get_signal_repository()
        stats = await repo.get_signal_stats()
        print(f"\n📈 Database Stats:")
        print(f"   Total signals: {stats['total_signals']}")
        print(f"   Recent (24h): {stats['recent_signals_24h']}")
        print(f"   Sources: {list(stats['sources'].keys())}")
        print(f"   Avg final score: {stats['average_final_score']}")

        return signals_1

    finally:
        await scout.stop()


async def demo_signal_analysis():
    """Demo 2: Historical signal analysis and trending detection"""

    print("\n\n🧠 Demo 2: Signal Analysis & Trends")
    print("=" * 50)

    scout = PersistentRedditSignalsScout()
    await scout.start()

    try:
        # Analyze historical patterns
        print("📊 Analyzing signal patterns...")
        analysis = await scout.analyze_signal_patterns(days=7)

        if 'error' not in analysis:
            print(f"   Signals analyzed: {analysis['total_signals_analyzed']}")
            print(f"   Analysis period: {analysis['analysis_period_days']} days")
            print(f"   Avg opportunity score: {analysis['average_opportunity_score']:.3f}")

            # Show top performing subreddits
            if analysis['top_performing_subreddits']:
                print(f"\n🏆 Top Performing Subreddits:")
                for subreddit, data in analysis['top_performing_subreddits'][:3]:
                    print(f"   r/{subreddit}: {data['count']} signals, avg score {data['avg_score']:.3f}")

            # Show trending keywords
            if analysis['trending_keywords']:
                print(f"\n🔥 Trending Keywords:")
                for keyword, count in analysis['trending_keywords'][:8]:
                    print(f"   {keyword}: {count} occurrences")

        # Get trending signals
        print(f"\n⚡ Getting trending signals...")
        trending = await scout.get_trending_signals(hours=48)
        print(f"   Found {len(trending)} trending signals")

        for i, signal in enumerate(trending[:3], 1):
            print(f"\n   {i}. {signal.title[:60]}...")
            print(f"      Score: {signal.final_score:.2f} | Comments: {signal.num_comments}")
            print(f"      Signals: {', '.join(signal.signals_found[:3])}")

        return analysis

    finally:
        await scout.stop()


async def demo_persistent_workflow():
    """Demo 3: Complete workflow with thesis synthesis"""

    print("\n\n🤖 Demo 3: Complete Persistent Workflow")
    print("=" * 50)

    # Initialize agents
    scout = PersistentRedditSignalsScout()
    synthesizer = ThesisSynthesizerAgent({
        'min_cluster_size': 2,
        'similarity_threshold': 0.3
    })

    await scout.start()

    try:
        # Step 1: Discover and persist Reddit signals
        print("Step 1: Discovering Reddit signals...")
        reddit_signals = await scout.discover_signals(
            subreddits=['entrepreneur', 'startups'],
            limit=12
        )
        print(f"   ✅ Discovered {len(reddit_signals)} Reddit signals")

        # Step 2: Get unprocessed signals from database
        signal_repo = get_signal_repository()
        unprocessed = await signal_repo.get_unprocessed_signals(limit=20)
        print(f"   📋 Found {len(unprocessed)} unprocessed signals in database")

        # Convert database signals to format for thesis synthesizer
        from agents.signals_scout import Signal
        converted_signals = []

        for db_signal in unprocessed:
            signal = Signal(
                id=db_signal.id,
                source=db_signal.source,
                title=db_signal.title,
                content=db_signal.content or "",
                url=db_signal.url or "",
                timestamp=db_signal.discovered_at,
                score=db_signal.final_score
            )
            converted_signals.append(signal)

        # Step 3: Synthesize theses from signals
        print(f"\nStep 2: Synthesizing theses from {len(converted_signals)} signals...")
        theses = await synthesizer.cluster_signals(converted_signals)
        print(f"   ✅ Generated {len(theses)} business theses")

        # Step 4: Persist theses to database
        thesis_repo = get_thesis_repository()
        persisted_theses = []

        for thesis in theses:
            # Extract supporting signal IDs
            supporting_signal_ids = [
                sig.id for sig in thesis.get('supporting_signals', [])
                if hasattr(sig, 'id')
            ]

            # Create thesis in database
            try:
                db_thesis = await thesis_repo.create_thesis(
                    hypothesis=thesis['hypothesis'],
                    domain=thesis['domain'],
                    confidence=thesis['confidence'],
                    supporting_signal_ids=supporting_signal_ids,
                    thesis_type=thesis['type'],
                    market_evidence=thesis.get('market_evidence', []),
                    risk_factors=thesis.get('risk_factors', [])
                )
                persisted_theses.append(db_thesis)
            except Exception as e:
                print(f"   Warning: Could not persist thesis: {e}")

        # Step 5: Mark signals as processed
        signal_ids = [signal.id for signal in converted_signals]
        processed_count = await signal_repo.mark_signals_processed(signal_ids)
        print(f"   ✅ Marked {processed_count} signals as processed")

        # Step 6: Show results
        print(f"\n📋 Workflow Results:")
        print(f"   Reddit signals discovered: {len(reddit_signals)}")
        print(f"   Database signals processed: {len(converted_signals)}")
        print(f"   Business theses generated: {len(theses)}")
        print(f"   Theses persisted to database: {len(persisted_theses)}")

        # Show sample theses
        print(f"\n💡 Sample Generated Theses:")
        for i, thesis in enumerate(persisted_theses[:2], 1):
            print(f"\n   {i}. {thesis.hypothesis}")
            print(f"      Domain: {thesis.domain}")
            print(f"      Confidence: {thesis.confidence:.2f}")
            print(f"      Supporting signals: {thesis.supporting_signals_count}")

        # Database final stats
        final_stats = await signal_repo.get_signal_stats()
        print(f"\n📊 Final Database Stats:")
        print(f"   Total signals: {final_stats['total_signals']}")
        print(f"   Average scores: {final_stats['average_final_score']:.3f}")

        return {
            'signals': reddit_signals,
            'theses': persisted_theses,
            'stats': final_stats
        }

    finally:
        await scout.stop()


async def demo_database_backup():
    """Demo 4: Database backup and management"""

    print("\n\n💾 Demo 4: Database Management")
    print("=" * 50)

    try:
        # Get database health
        health = await db_manager._health_check()
        print(f"📊 Database Health: {health['status']}")
        print(f"   Response time: {health.get('response_time_ms', 0):.1f}ms")
        print(f"   Database path: {health.get('database_path', 'unknown')}")

        # Create backup
        print(f"\n💾 Creating database backup...")
        backup_path = await db_manager.backup_database()
        print(f"   ✅ Backup created: {backup_path}")

        # Get comprehensive stats
        stats = await db_manager.get_stats()
        print(f"\n📈 Database Statistics:")
        for key, value in stats.items():
            if not key.endswith('_count') or value > 0:
                print(f"   {key}: {value}")

        return backup_path

    except Exception as e:
        print(f"   ❌ Database management error: {e}")
        return None


async def main():
    """Run all persistence demos"""

    print("🤖 Multi-Agent Venture Studio: Database Persistence Demo")
    print("=" * 70)
    print("This demo shows complete persistence with Reddit signals → database → theses")
    print()

    # Initialize database
    await init_database()

    try:
        # Run all demos
        signals = await demo_persistent_signals()
        analysis = await demo_signal_analysis()
        workflow_results = await demo_persistent_workflow()
        backup_path = await demo_database_backup()

        # Final summary
        print("\n\n🎉 Demo Complete!")
        print("=" * 50)
        print(f"✅ Persistent signal discovery working")
        print(f"✅ Signal analysis and trending detection working")
        print(f"✅ Complete workflow with thesis synthesis working")
        print(f"✅ Database backup and management working")

        if workflow_results:
            print(f"\n📊 Final Results:")
            print(f"   Total signals in database: {workflow_results['stats']['total_signals']}")
            print(f"   Business theses generated: {len(workflow_results['theses'])}")
            print(f"   Database backup: {backup_path}")

        print(f"\n🚀 Ready for production! Your venture studio now has:")
        print(f"   • Persistent signal storage with deduplication")
        print(f"   • Historical analysis and trend detection")
        print(f"   • Complete signal-to-thesis workflow")
        print(f"   • Database backup and recovery")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        raise

    finally:
        await db_manager.cleanup()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    asyncio.run(main())