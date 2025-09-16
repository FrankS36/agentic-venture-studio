#!/usr/bin/env python3
"""
Launch Script for Multi-Agent Venture Studio Dashboard

Simple launcher that:
1. Ensures database is populated with sample data
2. Launches the Streamlit dashboard
3. Opens browser automatically

Usage:
    python run_dashboard.py
"""

import asyncio
import subprocess
import sys
import webbrowser
from pathlib import Path
from time import sleep

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


async def ensure_database_ready():
    """Ensure database exists and has some sample data"""
    try:
        from persistence.database import init_database
        from persistence.repositories import get_signal_repository

        print("🔄 Initializing database...")
        await init_database()

        # Check if we have any signals
        repo = get_signal_repository()
        stats = await repo.get_signal_stats()

        if stats['total_signals'] == 0:
            print("📊 Database is empty. Running discovery to populate with sample data...")

            # Import and run Reddit scout to get some data
            from agents.persistent_reddit_scout import PersistentRedditSignalsScout

            scout = PersistentRedditSignalsScout()
            await scout.start()

            try:
                signals = await scout.discover_signals(['entrepreneur', 'startups'], limit=15)
                print(f"✅ Discovered {len(signals)} signals for the dashboard")
            finally:
                await scout.stop()

        else:
            print(f"✅ Database ready with {stats['total_signals']} signals")

    except Exception as e:
        print(f"⚠️  Database setup warning: {e}")
        print("   Dashboard will still work, but may have limited data")


def launch_streamlit():
    """Launch the Streamlit dashboard"""
    print("🚀 Launching Streamlit dashboard...")

    try:
        # Launch Streamlit
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.headless", "false"  # Allow opening browser
        ])

        print("🌐 Dashboard starting at http://localhost:8501")
        print("📊 Opening browser in 3 seconds...")

        # Wait a moment then open browser
        sleep(3)
        webbrowser.open("http://localhost:8501")

        print("\n🎉 Dashboard is ready!")
        print("\nFeatures available:")
        print("  📋 Signal Review - Browse and filter discovered signals")
        print("  📊 Analytics - View charts and patterns")
        print("  🔥 Trending - See signals with high engagement")
        print("  🔍 Discovery - Find new signals from Reddit")
        print("\nPress Ctrl+C to stop the dashboard")

        # Wait for user to stop
        process.wait()

    except KeyboardInterrupt:
        print("\n👋 Stopping dashboard...")
        process.terminate()
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")


async def main():
    """Main launcher function"""
    print("🤖 Multi-Agent Venture Studio Dashboard Launcher")
    print("=" * 55)

    # Ensure database is ready
    await ensure_database_ready()

    # Launch dashboard
    launch_streamlit()


if __name__ == "__main__":
    asyncio.run(main())