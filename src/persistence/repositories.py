"""
Repository Pattern for Multi-Agent Venture Studio

This module implements the repository pattern for type-safe database operations:
- SignalRepository: CRUD operations for market signals
- ThesisRepository: Business thesis management
- ExperimentRepository: Validation experiment tracking
- DecisionRepository: Stage gate decisions

Benefits:
- Type-safe async operations
- Centralized business logic
- Easy testing and mocking
- Clean separation of concerns
- Optimized queries with proper indexing
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from uuid import uuid4

from sqlalchemy import select, update, delete, func, and_, or_, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .models import Signal, Thesis, ThesisSignal, Experiment, ExperimentResult, Decision, AgentActivity
from .database import DatabaseManager, db_manager

logger = logging.getLogger(__name__)


class SignalRepository:
    """Repository for managing market signals with deduplication and scoring"""

    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or db_manager

    async def create_signal(self,
                          signal_id: str,
                          source: str,
                          title: str,
                          content: Optional[str] = None,
                          url: Optional[str] = None,
                          agent_id: str = "unknown",
                          signal_timestamp: Optional[datetime] = None,
                          raw_score: float = 0.0,
                          opportunity_score: float = 0.0,
                          engagement_score: float = 0.0,
                          urgency_score: float = 0.0,
                          final_score: float = 0.0,
                          confidence: float = 0.0,
                          source_metadata: Optional[Dict[str, Any]] = None,
                          keywords: Optional[List[str]] = None,
                          signals_found: Optional[List[str]] = None,
                          category: Optional[str] = None) -> Signal:
        """Create a new signal with deduplication check"""

        async with self.db.session() as session:
            # Check for duplicates
            existing = await session.execute(
                select(Signal).where(
                    and_(
                        Signal.source == source,
                        Signal.title == title
                    )
                )
            )

            if existing.scalar_one_or_none():
                logger.info(f"Signal already exists: {signal_id}")
                return existing.scalar_one()

            # Create new signal
            signal = Signal(
                id=signal_id,
                source=source,
                title=title,
                content=content,
                url=url,
                agent_id=agent_id,
                signal_timestamp=signal_timestamp or datetime.utcnow(),
                raw_score=raw_score,
                opportunity_score=opportunity_score,
                engagement_score=engagement_score,
                urgency_score=urgency_score,
                final_score=final_score,
                confidence=confidence,
                source_metadata=source_metadata,
                keywords=keywords or [],
                signals_found=signals_found or [],
                category=category
            )

            session.add(signal)
            await session.commit()
            await session.refresh(signal)

            logger.info(f"✅ Created signal: {signal_id} (score: {final_score:.2f})")
            return signal

    async def get_signal(self, signal_id: str) -> Optional[Signal]:
        """Get a signal by ID"""
        async with self.db.session() as session:
            result = await session.execute(
                select(Signal).where(Signal.id == signal_id)
            )
            return result.scalar_one_or_none()

    async def get_signals_by_source(self,
                                  source: str,
                                  limit: int = 100,
                                  min_score: float = 0.0) -> List[Signal]:
        """Get signals from a specific source"""
        async with self.db.session() as session:
            result = await session.execute(
                select(Signal)
                .where(
                    and_(
                        Signal.source.like(f"{source}%"),
                        Signal.final_score >= min_score
                    )
                )
                .order_by(desc(Signal.final_score))
                .limit(limit)
            )
            return result.scalars().all()

    async def get_top_signals(self,
                            limit: int = 50,
                            min_score: float = 0.3,
                            since: Optional[datetime] = None) -> List[Signal]:
        """Get top-scoring signals"""
        async with self.db.session() as session:
            query = select(Signal).where(Signal.final_score >= min_score)

            if since:
                query = query.where(Signal.discovered_at >= since)

            query = query.order_by(desc(Signal.final_score)).limit(limit)

            result = await session.execute(query)
            return result.scalars().all()

    async def get_unprocessed_signals(self, limit: int = 100) -> List[Signal]:
        """Get signals that haven't been processed by thesis synthesizer"""
        async with self.db.session() as session:
            result = await session.execute(
                select(Signal)
                .where(Signal.processed == False)
                .order_by(desc(Signal.final_score))
                .limit(limit)
            )
            return result.scalars().all()

    async def mark_signals_processed(self, signal_ids: List[str]) -> int:
        """Mark signals as processed"""
        async with self.db.session() as session:
            result = await session.execute(
                update(Signal)
                .where(Signal.id.in_(signal_ids))
                .values(processed=True, processed_at=datetime.utcnow())
            )
            await session.commit()
            return result.rowcount

    async def search_signals(self,
                           query: str,
                           limit: int = 50) -> List[Signal]:
        """Search signals by content"""
        async with self.db.session() as session:
            result = await session.execute(
                select(Signal)
                .where(
                    or_(
                        Signal.title.like(f"%{query}%"),
                        Signal.content.like(f"%{query}%")
                    )
                )
                .order_by(desc(Signal.final_score))
                .limit(limit)
            )
            return result.scalars().all()

    async def get_signal_stats(self) -> Dict[str, Any]:
        """Get signal statistics"""
        async with self.db.session() as session:
            # Total signals
            total_count = await session.execute(select(func.count(Signal.id)))

            # Signals by source
            source_stats = await session.execute(
                select(Signal.source, func.count(Signal.id))
                .group_by(Signal.source)
            )

            # Recent signals (last 24 hours)
            recent_count = await session.execute(
                select(func.count(Signal.id))
                .where(Signal.discovered_at >= datetime.utcnow() - timedelta(hours=24))
            )

            # Average scores
            avg_scores = await session.execute(
                select(
                    func.avg(Signal.final_score),
                    func.avg(Signal.opportunity_score),
                    func.avg(Signal.engagement_score)
                )
            )

            scores = avg_scores.first()

            return {
                "total_signals": total_count.scalar(),
                "recent_signals_24h": recent_count.scalar(),
                "sources": dict(source_stats.all()),
                "average_final_score": round(scores[0] or 0, 3),
                "average_opportunity_score": round(scores[1] or 0, 3),
                "average_engagement_score": round(scores[2] or 0, 3)
            }


class ThesisRepository:
    """Repository for managing business theses"""

    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or db_manager

    async def create_thesis(self,
                          hypothesis: str,
                          domain: str,
                          confidence: float,
                          supporting_signal_ids: List[str],
                          agent_id: str = "thesis-synthesizer",
                          thesis_type: str = "problem-solution",
                          market_size_estimate: Optional[str] = None,
                          target_audience: Optional[str] = None,
                          market_evidence: Optional[List[str]] = None,
                          risk_factors: Optional[List[str]] = None) -> Thesis:
        """Create a new business thesis"""

        thesis_id = str(uuid4())

        async with self.db.session() as session:
            # Create thesis
            thesis = Thesis(
                id=thesis_id,
                hypothesis=hypothesis,
                domain=domain,
                confidence=confidence,
                supporting_signals_count=len(supporting_signal_ids),
                market_size_estimate=market_size_estimate,
                target_audience=target_audience,
                market_evidence=market_evidence or [],
                risk_factors=risk_factors or [],
                thesis_type=thesis_type,
                created_by_agent=agent_id
            )

            session.add(thesis)

            # Link supporting signals
            for signal_id in supporting_signal_ids:
                thesis_signal = ThesisSignal(
                    thesis_id=thesis_id,
                    signal_id=signal_id,
                    weight=1.0  # Could be calculated based on signal relevance
                )
                session.add(thesis_signal)

            await session.commit()
            await session.refresh(thesis)

            logger.info(f"✅ Created thesis: {thesis_id} (confidence: {confidence:.2f})")
            return thesis

    async def get_thesis(self, thesis_id: str) -> Optional[Thesis]:
        """Get thesis with related signals"""
        async with self.db.session() as session:
            result = await session.execute(
                select(Thesis)
                .options(selectinload(Thesis.thesis_signals))
                .where(Thesis.id == thesis_id)
            )
            return result.scalar_one_or_none()

    async def get_active_theses(self, limit: int = 50) -> List[Thesis]:
        """Get active theses (not killed)"""
        async with self.db.session() as session:
            result = await session.execute(
                select(Thesis)
                .where(Thesis.status != 'killed')
                .order_by(desc(Thesis.confidence))
                .limit(limit)
            )
            return result.scalars().all()

    async def update_thesis_status(self, thesis_id: str, status: str, stage: str) -> bool:
        """Update thesis status and stage"""
        async with self.db.session() as session:
            result = await session.execute(
                update(Thesis)
                .where(Thesis.id == thesis_id)
                .values(status=status, current_stage=stage, last_updated=datetime.utcnow())
            )
            await session.commit()
            return result.rowcount > 0

    async def get_thesis_with_signals(self, thesis_id: str) -> Optional[Tuple[Thesis, List[Signal]]]:
        """Get thesis with its supporting signals"""
        async with self.db.session() as session:
            # Get thesis
            thesis_result = await session.execute(
                select(Thesis).where(Thesis.id == thesis_id)
            )
            thesis = thesis_result.scalar_one_or_none()

            if not thesis:
                return None

            # Get supporting signals
            signals_result = await session.execute(
                select(Signal)
                .join(ThesisSignal)
                .where(ThesisSignal.thesis_id == thesis_id)
                .order_by(desc(ThesisSignal.weight))
            )
            signals = signals_result.scalars().all()

            return thesis, signals


class ExperimentRepository:
    """Repository for managing validation experiments"""

    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or db_manager

    async def create_experiment(self,
                              thesis_id: str,
                              name: str,
                              experiment_type: str,
                              success_metrics: Dict[str, Any],
                              success_thresholds: Dict[str, float],
                              assigned_agent: str,
                              budget: float = 0.0,
                              duration_days: int = 7,
                              description: Optional[str] = None) -> Experiment:
        """Create a new experiment"""

        experiment_id = str(uuid4())

        async with self.db.session() as session:
            experiment = Experiment(
                id=experiment_id,
                thesis_id=thesis_id,
                name=name,
                description=description,
                experiment_type=experiment_type,
                budget=budget,
                duration_days=duration_days,
                success_metrics=success_metrics,
                success_thresholds=success_thresholds,
                assigned_agent=assigned_agent
            )

            session.add(experiment)
            await session.commit()
            await session.refresh(experiment)

            logger.info(f"✅ Created experiment: {experiment_id} for thesis {thesis_id}")
            return experiment

    async def start_experiment(self, experiment_id: str) -> bool:
        """Mark experiment as started"""
        async with self.db.session() as session:
            result = await session.execute(
                update(Experiment)
                .where(Experiment.id == experiment_id)
                .values(status='running', started_at=datetime.utcnow())
            )
            await session.commit()
            return result.rowcount > 0

    async def record_result(self,
                          experiment_id: str,
                          metric_name: str,
                          metric_value: float,
                          metric_unit: Optional[str] = None,
                          sample_size: Optional[int] = None,
                          confidence_interval: Optional[Tuple[float, float]] = None,
                          notes: Optional[str] = None) -> ExperimentResult:
        """Record an experiment result"""

        result_id = str(uuid4())

        async with self.db.session() as session:
            result = ExperimentResult(
                id=result_id,
                experiment_id=experiment_id,
                metric_name=metric_name,
                metric_value=metric_value,
                metric_unit=metric_unit,
                sample_size=sample_size,
                confidence_interval=list(confidence_interval) if confidence_interval else None,
                notes=notes
            )

            session.add(result)
            await session.commit()

            logger.info(f"📊 Recorded result: {metric_name}={metric_value} for experiment {experiment_id}")
            return result

    async def complete_experiment(self,
                                experiment_id: str,
                                outcome: str,
                                confidence_level: float) -> bool:
        """Mark experiment as completed"""
        async with self.db.session() as session:
            result = await session.execute(
                update(Experiment)
                .where(Experiment.id == experiment_id)
                .values(
                    status='completed',
                    completed_at=datetime.utcnow(),
                    outcome=outcome,
                    confidence_level=confidence_level
                )
            )
            await session.commit()
            return result.rowcount > 0


class DecisionRepository:
    """Repository for managing stage gate decisions"""

    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or db_manager

    async def record_decision(self,
                            thesis_id: str,
                            stage: str,
                            verdict: str,
                            rationale: str,
                            reviewer_agent: str,
                            reviewer_confidence: float,
                            evidence_summary: Optional[Dict[str, Any]] = None,
                            key_metrics: Optional[Dict[str, float]] = None,
                            next_actions: Optional[List[str]] = None) -> Decision:
        """Record a stage gate decision"""

        decision_id = str(uuid4())

        async with self.db.session() as session:
            decision = Decision(
                id=decision_id,
                thesis_id=thesis_id,
                stage=stage,
                verdict=verdict,
                rationale=rationale,
                reviewer_agent=reviewer_agent,
                reviewer_confidence=reviewer_confidence,
                evidence_summary=evidence_summary,
                key_metrics=key_metrics,
                next_actions=next_actions or []
            )

            session.add(decision)
            await session.commit()

            logger.info(f"⚖️  Decision recorded: {verdict} for thesis {thesis_id} at stage {stage}")
            return decision


# Convenience factory functions
def get_signal_repository(db: Optional[DatabaseManager] = None) -> SignalRepository:
    """Get a SignalRepository instance"""
    return SignalRepository(db)

def get_thesis_repository(db: Optional[DatabaseManager] = None) -> ThesisRepository:
    """Get a ThesisRepository instance"""
    return ThesisRepository(db)

def get_experiment_repository(db: Optional[DatabaseManager] = None) -> ExperimentRepository:
    """Get an ExperimentRepository instance"""
    return ExperimentRepository(db)

def get_decision_repository(db: Optional[DatabaseManager] = None) -> DecisionRepository:
    """Get a DecisionRepository instance"""
    return DecisionRepository(db)