"""
Database Models for Multi-Agent Venture Studio

This module defines the complete data model for the venture studio system:
- Signals: Market opportunities and trends discovered by agents
- Theses: Business hypotheses synthesized from signals
- Experiments: Validation tests run on theses
- Results: Outcomes and metrics from experiments
- Decisions: Go/no-go decisions at each stage gate

The schema is designed for:
- High-performance async operations
- Type safety with SQLAlchemy 2.0
- Audit trails and observability
- Easy querying and analytics
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import (
    Column, Integer, String, Text, Float, DateTime, Boolean, JSON,
    ForeignKey, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func
import uuid


class Base(DeclarativeBase):
    """Base class for all database models"""
    pass


class Signal(Base):
    """Market signals discovered by agents"""
    __tablename__ = 'signals'

    # Primary identification
    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    source: Mapped[str] = mapped_column(String(100), nullable=False)  # e.g., "reddit:r/entrepreneur"
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    content: Mapped[Optional[str]] = mapped_column(Text)
    url: Mapped[Optional[str]] = mapped_column(String(1000))

    # Metadata
    discovered_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    signal_timestamp: Mapped[Optional[datetime]] = mapped_column(DateTime)  # Original timestamp from source
    agent_id: Mapped[str] = mapped_column(String(100), nullable=False)

    # Scoring and analysis
    raw_score: Mapped[float] = mapped_column(Float, default=0.0)
    opportunity_score: Mapped[float] = mapped_column(Float, default=0.0)
    engagement_score: Mapped[float] = mapped_column(Float, default=0.0)
    urgency_score: Mapped[float] = mapped_column(Float, default=0.0)
    final_score: Mapped[float] = mapped_column(Float, default=0.0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)

    # Source-specific data (Reddit upvotes, Twitter likes, etc.)
    source_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)

    # Signal categorization
    keywords: Mapped[Optional[List[str]]] = mapped_column(JSON)
    signals_found: Mapped[Optional[List[str]]] = mapped_column(JSON)  # Detected opportunity signals
    category: Mapped[Optional[str]] = mapped_column(String(100))

    # Processing status
    processed: Mapped[bool] = mapped_column(Boolean, default=False)
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    # Relationships
    thesis_signals = relationship("ThesisSignal", back_populates="signal")

    # Indexes for performance
    __table_args__ = (
        Index('idx_signals_source', 'source'),
        Index('idx_signals_discovered_at', 'discovered_at'),
        Index('idx_signals_final_score', 'final_score'),
        Index('idx_signals_processed', 'processed'),
        Index('idx_signals_agent_source', 'agent_id', 'source'),
    )


class Thesis(Base):
    """Business theses synthesized from signals"""
    __tablename__ = 'theses'

    # Primary identification
    id: Mapped[str] = mapped_column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    hypothesis: Mapped[str] = mapped_column(Text, nullable=False)
    domain: Mapped[str] = mapped_column(String(200), nullable=False)

    # Analysis
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    market_size_estimate: Mapped[Optional[str]] = mapped_column(String(100))
    target_audience: Mapped[Optional[str]] = mapped_column(Text)
    competitive_landscape: Mapped[Optional[str]] = mapped_column(Text)

    # Evidence and support
    supporting_signals_count: Mapped[int] = mapped_column(Integer, default=0)
    market_evidence: Mapped[Optional[List[str]]] = mapped_column(JSON)
    risk_factors: Mapped[Optional[List[str]]] = mapped_column(JSON)

    # Thesis categorization
    thesis_type: Mapped[str] = mapped_column(String(50), nullable=False)  # problem-solution, market-gap, etc.
    urgency_level: Mapped[str] = mapped_column(String(20), default='medium')  # low, medium, high
    complexity_level: Mapped[str] = mapped_column(String(20), default='medium')

    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    created_by_agent: Mapped[str] = mapped_column(String(100), nullable=False)
    last_updated: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    # Workflow status
    status: Mapped[str] = mapped_column(String(20), default='draft')  # draft, validated, testing, scaled, killed
    current_stage: Mapped[str] = mapped_column(String(50), default='synthesis')

    # Relationships
    thesis_signals = relationship("ThesisSignal", back_populates="thesis")
    experiments = relationship("Experiment", back_populates="thesis")
    decisions = relationship("Decision", back_populates="thesis")

    # Constraints and indexes
    __table_args__ = (
        Index('idx_theses_confidence', 'confidence'),
        Index('idx_theses_created_at', 'created_at'),
        Index('idx_theses_status', 'status'),
        Index('idx_theses_domain', 'domain'),
        CheckConstraint('confidence >= 0 AND confidence <= 1', name='check_confidence_range'),
        CheckConstraint('supporting_signals_count >= 0', name='check_signals_count'),
    )


class ThesisSignal(Base):
    """Many-to-many relationship between theses and signals"""
    __tablename__ = 'thesis_signals'

    thesis_id: Mapped[str] = mapped_column(String(50), ForeignKey('theses.id'), primary_key=True)
    signal_id: Mapped[str] = mapped_column(String(50), ForeignKey('signals.id'), primary_key=True)
    weight: Mapped[float] = mapped_column(Float, default=1.0)  # How much this signal contributes
    added_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationships
    thesis = relationship("Thesis", back_populates="thesis_signals")
    signal = relationship("Signal", back_populates="thesis_signals")


class Experiment(Base):
    """Validation experiments run on business theses"""
    __tablename__ = 'experiments'

    # Primary identification
    id: Mapped[str] = mapped_column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    thesis_id: Mapped[str] = mapped_column(String(50), ForeignKey('theses.id'), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)

    # Experiment configuration
    experiment_type: Mapped[str] = mapped_column(String(50), nullable=False)  # fake-door, smoke-ad, etc.
    budget: Mapped[float] = mapped_column(Float, default=0.0)
    duration_days: Mapped[int] = mapped_column(Integer, default=7)

    # Success criteria
    success_metrics: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    success_thresholds: Mapped[Dict[str, float]] = mapped_column(JSON, nullable=False)

    # Timing
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    # Status and results
    status: Mapped[str] = mapped_column(String(20), default='planned')  # planned, running, completed, failed
    outcome: Mapped[Optional[str]] = mapped_column(String(20))  # success, failure, inconclusive
    confidence_level: Mapped[Optional[float]] = mapped_column(Float)

    # Agent assignment
    assigned_agent: Mapped[str] = mapped_column(String(100), nullable=False)

    # Relationships
    thesis = relationship("Thesis", back_populates="experiments")
    results = relationship("ExperimentResult", back_populates="experiment")

    # Indexes
    __table_args__ = (
        Index('idx_experiments_thesis_id', 'thesis_id'),
        Index('idx_experiments_status', 'status'),
        Index('idx_experiments_created_at', 'created_at'),
        Index('idx_experiments_type', 'experiment_type'),
    )


class ExperimentResult(Base):
    """Results and metrics from experiments"""
    __tablename__ = 'experiment_results'

    # Primary identification
    id: Mapped[str] = mapped_column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    experiment_id: Mapped[str] = mapped_column(String(50), ForeignKey('experiments.id'), nullable=False)

    # Metric data
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    metric_unit: Mapped[Optional[str]] = mapped_column(String(50))

    # Statistical data
    sample_size: Mapped[Optional[int]] = mapped_column(Integer)
    confidence_interval: Mapped[Optional[List[float]]] = mapped_column(JSON)  # [lower, upper]
    p_value: Mapped[Optional[float]] = mapped_column(Float)

    # Timing
    recorded_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    measurement_date: Mapped[Optional[datetime]] = mapped_column(DateTime)

    # Additional context
    notes: Mapped[Optional[str]] = mapped_column(Text)
    raw_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)

    # Relationships
    experiment = relationship("Experiment", back_populates="results")

    # Constraints
    __table_args__ = (
        Index('idx_results_experiment_id', 'experiment_id'),
        Index('idx_results_metric_name', 'metric_name'),
        Index('idx_results_recorded_at', 'recorded_at'),
        UniqueConstraint('experiment_id', 'metric_name', 'recorded_at', name='uq_experiment_metric_time'),
    )


class Decision(Base):
    """Go/no-go decisions at stage gates"""
    __tablename__ = 'decisions'

    # Primary identification
    id: Mapped[str] = mapped_column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    thesis_id: Mapped[str] = mapped_column(String(50), ForeignKey('theses.id'), nullable=False)

    # Decision details
    stage: Mapped[str] = mapped_column(String(50), nullable=False)  # signals, validation, mvp, etc.
    verdict: Mapped[str] = mapped_column(String(20), nullable=False)  # go, no-go, pivot
    rationale: Mapped[str] = mapped_column(Text, nullable=False)

    # Evidence summary
    evidence_summary: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    key_metrics: Mapped[Optional[Dict[str, float]]] = mapped_column(JSON)

    # Decision maker
    reviewer_agent: Mapped[str] = mapped_column(String(100), nullable=False)
    reviewer_confidence: Mapped[float] = mapped_column(Float, nullable=False)

    # Timing
    decided_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Follow-up actions
    next_actions: Mapped[Optional[List[str]]] = mapped_column(JSON)
    assigned_agents: Mapped[Optional[List[str]]] = mapped_column(JSON)

    # Relationships
    thesis = relationship("Thesis", back_populates="decisions")

    # Indexes
    __table_args__ = (
        Index('idx_decisions_thesis_id', 'thesis_id'),
        Index('idx_decisions_stage', 'stage'),
        Index('idx_decisions_verdict', 'verdict'),
        Index('idx_decisions_decided_at', 'decided_at'),
    )


class AgentActivity(Base):
    """Audit trail of agent activities"""
    __tablename__ = 'agent_activities'

    # Primary identification
    id: Mapped[str] = mapped_column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_id: Mapped[str] = mapped_column(String(100), nullable=False)
    activity_type: Mapped[str] = mapped_column(String(50), nullable=False)

    # Activity details
    description: Mapped[str] = mapped_column(String(500), nullable=False)
    target_entity_type: Mapped[Optional[str]] = mapped_column(String(50))  # signal, thesis, experiment
    target_entity_id: Mapped[Optional[str]] = mapped_column(String(50))

    # Context and metadata
    context: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    outcome: Mapped[Optional[str]] = mapped_column(String(20))  # success, failure, partial
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer)

    # Timing
    started_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    # Indexes for audit queries
    __table_args__ = (
        Index('idx_activities_agent_id', 'agent_id'),
        Index('idx_activities_started_at', 'started_at'),
        Index('idx_activities_type', 'activity_type'),
        Index('idx_activities_target', 'target_entity_type', 'target_entity_id'),
    )