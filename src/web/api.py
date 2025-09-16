"""
FastAPI Web Interface for Multi-Agent Venture Studio

This module provides a clean REST API and web interface for:
- Reviewing discovered signals with filtering and search
- Monitoring system performance and database health
- Analyzing signal trends and patterns
- Managing the venture studio workflow

Features:
- Real-time signal dashboard
- Advanced filtering and search
- Signal scoring visualization
- System health monitoring
- Responsive design for mobile/desktop
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Handle imports for different execution contexts
try:
    from ..persistence.database import init_database, db_manager
    from ..persistence.repositories import get_signal_repository, get_thesis_repository
    from ..agents.persistent_reddit_scout import PersistentRedditSignalsScout
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from persistence.database import init_database, db_manager
    from persistence.repositories import get_signal_repository, get_thesis_repository
    from agents.persistent_reddit_scout import PersistentRedditSignalsScout

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Agent Venture Studio",
    description="AI-powered business opportunity discovery and validation platform",
    version="1.0.0"
)

# Setup templates and static files
templates_dir = Path(__file__).parent / "templates"
static_dir = Path(__file__).parent / "static"

templates_dir.mkdir(exist_ok=True)
static_dir.mkdir(exist_ok=True)

templates = Jinja2Templates(directory=str(templates_dir))
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Global state
reddit_scout: Optional[PersistentRedditSignalsScout] = None


# Pydantic models for API responses
class SignalResponse(BaseModel):
    id: str
    source: str
    title: str
    content: Optional[str]
    url: Optional[str]
    final_score: float
    opportunity_score: float
    engagement_score: float
    urgency_score: float
    confidence: float
    discovered_at: datetime
    keywords: List[str]
    signals_found: List[str]
    source_metadata: Dict[str, Any]

class DatabaseStats(BaseModel):
    total_signals: int
    recent_signals_24h: int
    sources: Dict[str, int]
    average_final_score: float
    average_opportunity_score: float

class SystemHealth(BaseModel):
    status: str
    response_time_ms: float
    database_path: str
    connection_count: int
    error_count: int
    last_check: str


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize database and Reddit scout on startup"""
    global reddit_scout

    try:
        # Initialize database
        await init_database()
        logger.info("✅ Database initialized for web interface")

        # Initialize Reddit scout
        reddit_scout = PersistentRedditSignalsScout()
        await reddit_scout.start()
        logger.info("✅ Reddit scout initialized for web interface")

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global reddit_scout

    if reddit_scout:
        await reddit_scout.stop()

    await db_manager.cleanup()
    logger.info("🧹 Web interface shutdown complete")


# Web Routes
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/signals", response_class=HTMLResponse)
async def signals_page(request: Request):
    """Signals review page"""
    return templates.TemplateResponse("signals.html", {"request": request})


@app.get("/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request):
    """Analytics and trends page"""
    return templates.TemplateResponse("analytics.html", {"request": request})


# API Routes
@app.get("/api/signals", response_model=List[SignalResponse])
async def get_signals(
    limit: int = Query(50, ge=1, le=200),
    min_score: float = Query(0.0, ge=0.0, le=1.0),
    source: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    since_hours: Optional[int] = Query(None, ge=1, le=720)  # Max 30 days
):
    """Get signals with filtering options"""

    try:
        signal_repo = get_signal_repository()

        if search:
            # Search functionality
            signals = await signal_repo.search_signals(search, limit)
        elif source:
            # Filter by source
            signals = await signal_repo.get_signals_by_source(source, limit, min_score)
        else:
            # Get top signals with optional time filter
            since = None
            if since_hours:
                since = datetime.utcnow() - timedelta(hours=since_hours)

            signals = await signal_repo.get_top_signals(limit, min_score, since)

        # Convert to response format
        signal_responses = []
        for signal in signals:
            signal_response = SignalResponse(
                id=signal.id,
                source=signal.source,
                title=signal.title,
                content=signal.content,
                url=signal.url,
                final_score=signal.final_score,
                opportunity_score=signal.opportunity_score,
                engagement_score=signal.engagement_score,
                urgency_score=signal.urgency_score,
                confidence=signal.confidence,
                discovered_at=signal.discovered_at,
                keywords=signal.keywords or [],
                signals_found=signal.signals_found or [],
                source_metadata=signal.source_metadata or {}
            )
            signal_responses.append(signal_response)

        return signal_responses

    except Exception as e:
        logger.error(f"Error getting signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/signals/{signal_id}", response_model=SignalResponse)
async def get_signal(signal_id: str):
    """Get a specific signal by ID"""

    try:
        signal_repo = get_signal_repository()
        signal = await signal_repo.get_signal(signal_id)

        if not signal:
            raise HTTPException(status_code=404, detail="Signal not found")

        return SignalResponse(
            id=signal.id,
            source=signal.source,
            title=signal.title,
            content=signal.content,
            url=signal.url,
            final_score=signal.final_score,
            opportunity_score=signal.opportunity_score,
            engagement_score=signal.engagement_score,
            urgency_score=signal.urgency_score,
            confidence=signal.confidence,
            discovered_at=signal.discovered_at,
            keywords=signal.keywords or [],
            signals_found=signal.signals_found or [],
            source_metadata=signal.source_metadata or {}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting signal {signal_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats", response_model=DatabaseStats)
async def get_database_stats():
    """Get database statistics"""

    try:
        signal_repo = get_signal_repository()
        stats = await signal_repo.get_signal_stats()

        return DatabaseStats(
            total_signals=stats['total_signals'],
            recent_signals_24h=stats['recent_signals_24h'],
            sources=stats['sources'],
            average_final_score=stats['average_final_score'],
            average_opportunity_score=stats['average_opportunity_score']
        )

    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health", response_model=SystemHealth)
async def get_system_health():
    """Get system health status"""

    try:
        health = await db_manager._health_check()

        return SystemHealth(
            status=health['status'],
            response_time_ms=health.get('response_time_ms', 0),
            database_path=health.get('database_path', 'unknown'),
            connection_count=health.get('connection_count', 0),
            error_count=health.get('error_count', 0),
            last_check=health['last_check']
        )

    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/discover")
async def discover_signals(
    subreddits: List[str] = ["entrepreneur", "startups"],
    limit: int = 10
):
    """Trigger signal discovery"""

    global reddit_scout

    if not reddit_scout:
        raise HTTPException(status_code=503, detail="Reddit scout not available")

    try:
        signals = await reddit_scout.discover_signals(subreddits, limit=limit)

        return {
            "success": True,
            "signals_discovered": len(signals),
            "subreddits": subreddits,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error discovering signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics")
async def get_analytics(days: int = Query(7, ge=1, le=30)):
    """Get analytics and trends"""

    global reddit_scout

    if not reddit_scout:
        raise HTTPException(status_code=503, detail="Reddit scout not available")

    try:
        analysis = await reddit_scout.analyze_signal_patterns(days=days)

        if 'error' in analysis:
            raise HTTPException(status_code=500, detail=analysis['error'])

        return analysis

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trending")
async def get_trending_signals(hours: int = Query(24, ge=1, le=168)):  # Max 1 week
    """Get trending signals"""

    global reddit_scout

    if not reddit_scout:
        raise HTTPException(status_code=503, detail="Reddit scout not available")

    try:
        trending = await reddit_scout.get_trending_signals(hours=hours)

        # Convert to response format
        trending_responses = []
        for signal in trending:
            trending_response = {
                "id": signal.id,
                "title": signal.title,
                "source": signal.source,
                "final_score": signal.final_score,
                "opportunity_score": signal.opportunity_score,
                "engagement_score": signal.engagement_score,
                "num_comments": signal.num_comments,
                "score": signal.score,
                "timestamp": signal.timestamp.isoformat(),
                "signals_found": signal.signals_found,
                "subreddit": signal.subreddit,
                "permalink": signal.permalink
            }
            trending_responses.append(trending_response)

        return trending_responses

    except Exception as e:
        logger.error(f"Error getting trending signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Custom 404 page"""
    return templates.TemplateResponse("404.html", {"request": request}, status_code=404)


@app.exception_handler(500)
async def server_error_handler(request: Request, exc):
    """Custom 500 page"""
    return templates.TemplateResponse("500.html", {"request": request}, status_code=500)


# Development server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)