"""
Multi-Agent Telemetry System: Python Implementation

This module provides comprehensive observability for multi-agent systems:
- Distributed tracing with async context management
- Structured logging with proper correlation
- Performance metrics collection and aggregation
- Real-time monitoring and alerting

Learning Focus: How Python's async ecosystem and type system
create better observability than callback-based approaches.
"""

import asyncio
import logging
import time
import json
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Callable, Union, AsyncContextManager
from datetime import datetime, timedelta
from enum import Enum
from contextlib import asynccontextmanager
from collections import defaultdict, deque
import functools
import inspect


class TraceStatus(Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Span:
    """A span within a distributed trace"""
    id: str
    name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: TraceStatus = TraceStatus.ACTIVE


@dataclass
class Trace:
    """A distributed trace across multiple agents"""
    id: str
    agent_id: str
    operation: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)
    spans: List[Span] = field(default_factory=list)
    tags: Dict[str, Any] = field(default_factory=dict)
    status: TraceStatus = TraceStatus.ACTIVE
    error: Optional[str] = None


@dataclass
class Metric:
    """A time-series metric measurement"""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, Any] = field(default_factory=dict)
    unit: str = "count"


@dataclass
class Event:
    """A structured event for audit logging"""
    id: str
    type: str
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None
    span_id: Optional[str] = None


@dataclass
class Alert:
    """A system alert"""
    id: str
    type: str
    severity: AlertSeverity
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    status: str = "active"


@dataclass
class AgentInfo:
    """Information about a registered agent"""
    id: str
    capabilities: List[str]
    registered_at: datetime
    status: str = "active"
    metrics: Dict[str, Any] = field(default_factory=dict)


class TelemetryConfig:
    """Configuration for the telemetry system"""
    def __init__(self):
        self.enable_tracing = True
        self.enable_metrics = True
        self.enable_logging = True
        self.log_level = logging.INFO
        self.metrics_interval = 5.0  # seconds
        self.trace_buffer_size = 10000
        self.event_buffer_size = 50000
        self.persist_metrics = False  # Set to True for production
        self.alert_thresholds = {
            'error_rate': 0.05,
            'avg_response_time': 10.0,
            'memory_usage': 0.8
        }


class AgentTelemetrySystem:
    """
    Production-grade telemetry system for multi-agent systems

    Learning Note: Python's async context managers and type system
    make distributed tracing much more elegant than manual instrumentation.
    """

    def __init__(self, config: Optional[TelemetryConfig] = None):
        self.config = config or TelemetryConfig()

        # Core state
        self.traces: Dict[str, Trace] = {}
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.events: deque = deque(maxlen=self.config.event_buffer_size)
        self.alerts: List[Alert] = []
        self.agents: Dict[str, AgentInfo] = {}

        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)

        # Performance tracking
        self.start_time = time.time()
        self.system_metrics = {
            'traces_created': 0,
            'spans_created': 0,
            'events_recorded': 0,
            'alerts_triggered': 0
        }

        # Setup logging
        self.logger = logging.getLogger('telemetry')
        self.logger.setLevel(self.config.log_level)

        # Start background tasks
        self._background_tasks = []
        if self.config.enable_metrics:
            self._start_metrics_collection()

        self.logger.info("📊 Python Telemetry system initialized")

    async def register_agent(self, agent_id: str, agent_instance: Any, capabilities: List[str] = None) -> AgentInfo:
        """
        Register an agent for monitoring

        Learning Note: Python's async context allows for clean agent
        lifecycle management and automatic instrumentation.
        """
        capabilities = capabilities or []

        agent_info = AgentInfo(
            id=agent_id,
            capabilities=capabilities,
            registered_at=datetime.now(),
            metrics={
                'tasks_completed': 0,
                'tasks_started': 0,
                'total_duration': 0.0,
                'error_count': 0,
                'last_activity': datetime.now()
            }
        )

        self.agents[agent_id] = agent_info

        # Instrument agent methods
        await self._instrument_agent(agent_id, agent_instance, capabilities)

        self.logger.info(f"📋 Registered agent for monitoring: {agent_id}")
        await self._emit_metric('agent.registered', 1, {'agent_id': agent_id})

        return agent_info

    async def _instrument_agent(self, agent_id: str, agent_instance: Any, capabilities: List[str]) -> None:
        """
        Instrument agent methods with automatic tracing

        Learning Note: Python's decorators and introspection make
        automatic instrumentation much cleaner than proxy patterns.
        """
        for method_name in capabilities:
            if hasattr(agent_instance, method_name):
                original_method = getattr(agent_instance, method_name)

                # Create instrumented version
                instrumented = self._create_instrumented_method(agent_id, method_name, original_method)

                # Replace the method
                setattr(agent_instance, method_name, instrumented)

    def _create_instrumented_method(self, agent_id: str, method_name: str, original_method: Callable) -> Callable:
        """Create an instrumented version of an agent method"""

        if inspect.iscoroutinefunction(original_method):
            @functools.wraps(original_method)
            async def async_instrumented(*args, **kwargs):
                async with self.trace(agent_id, method_name, {'args_count': len(args), 'kwargs_count': len(kwargs)}) as trace:
                    try:
                        await self._record_event('method.started', {
                            'agent_id': agent_id,
                            'method': method_name,
                            'trace_id': trace.id
                        })

                        result = await original_method(*args, **kwargs)

                        await self._record_event('method.completed', {
                            'agent_id': agent_id,
                            'method': method_name,
                            'trace_id': trace.id,
                            'success': True
                        })

                        return result

                    except Exception as error:
                        await self._record_event('method.error', {
                            'agent_id': agent_id,
                            'method': method_name,
                            'trace_id': trace.id,
                            'error': str(error)
                        })
                        raise

            return async_instrumented
        else:
            @functools.wraps(original_method)
            def sync_instrumented(*args, **kwargs):
                # For sync methods, we can't use async context managers
                trace_id = self._start_trace_sync(agent_id, method_name)

                try:
                    result = original_method(*args, **kwargs)
                    self._end_trace_sync(trace_id, True)
                    return result
                except Exception as error:
                    self._end_trace_sync(trace_id, False, str(error))
                    raise

            return sync_instrumented

    @asynccontextmanager
    async def trace(self, agent_id: str, operation: str, context: Dict[str, Any] = None) -> AsyncContextManager[Trace]:
        """
        Async context manager for distributed tracing

        Learning Note: Python's async context managers provide elegant
        resource management and automatic cleanup for distributed traces.
        """
        trace_id = str(uuid.uuid4())
        start_time = time.time()

        trace = Trace(
            id=trace_id,
            agent_id=agent_id,
            operation=operation,
            start_time=start_time,
            context=context or {}
        )

        self.traces[trace_id] = trace
        self.system_metrics['traces_created'] += 1

        if self.config.enable_tracing:
            self.logger.debug(f"🔍 Started trace: {trace_id} [{agent_id}.{operation}]")

        try:
            await self._emit_metric('trace.started', 1, {'agent_id': agent_id, 'operation': operation})
            yield trace

            # Successful completion
            trace.end_time = time.time()
            trace.duration = trace.end_time - trace.start_time
            trace.status = TraceStatus.COMPLETED

            await self._emit_metric('trace.completed', 1, {
                'agent_id': agent_id,
                'operation': operation,
                'duration': trace.duration,
                'success': True
            })

            # Update agent metrics
            agent = self.agents.get(agent_id)
            if agent:
                agent.metrics['tasks_completed'] += 1
                agent.metrics['total_duration'] += trace.duration
                agent.metrics['last_activity'] = datetime.now()

        except Exception as error:
            # Error completion
            trace.end_time = time.time()
            trace.duration = trace.end_time - trace.start_time
            trace.status = TraceStatus.FAILED
            trace.error = str(error)

            await self._emit_metric('trace.completed', 1, {
                'agent_id': agent_id,
                'operation': operation,
                'duration': trace.duration,
                'success': False
            })

            # Update agent error metrics
            agent = self.agents.get(agent_id)
            if agent:
                agent.metrics['error_count'] += 1

            # Check for alert conditions
            await self._check_alert_conditions(agent_id, error)

            raise

        finally:
            if self.config.enable_tracing:
                self.logger.debug(f"✅ Completed trace: {trace_id} ({trace.duration:.3f}s)")

    @asynccontextmanager
    async def span(self, trace_id: str, span_name: str, tags: Dict[str, Any] = None) -> AsyncContextManager[Span]:
        """
        Create a span within an existing trace

        Learning Note: Spans allow for detailed timing of sub-operations
        within a larger workflow. Essential for performance optimization.
        """
        span_id = str(uuid.uuid4())
        start_time = time.time()

        span = Span(
            id=span_id,
            name=span_name,
            start_time=start_time,
            tags=tags or {}
        )

        trace = self.traces.get(trace_id)
        if trace:
            trace.spans.append(span)

        self.system_metrics['spans_created'] += 1

        try:
            yield span

            span.end_time = time.time()
            span.duration = span.end_time - span.start_time
            span.status = TraceStatus.COMPLETED

        except Exception as error:
            span.end_time = time.time()
            span.duration = span.end_time - span.start_time
            span.status = TraceStatus.FAILED
            span.logs.append({
                'timestamp': time.time(),
                'level': 'error',
                'message': str(error)
            })
            raise

    async def _record_event(self, event_type: str, data: Dict[str, Any] = None) -> str:
        """
        Record a structured event

        Learning Note: Structured events provide the audit trail needed
        for debugging complex multi-agent interactions.
        """
        event_id = str(uuid.uuid4())
        event = Event(
            id=event_id,
            type=event_type,
            timestamp=datetime.now(),
            data=data or {},
            tags=self._extract_event_tags(data or {}),
            trace_id=data.get('trace_id') if data else None
        )

        self.events.append(event)
        self.system_metrics['events_recorded'] += 1

        if self.config.enable_logging:
            self._log_event(event)

        # Emit to event handlers
        await self._emit_event_to_handlers(event)

        return event_id

    async def _emit_metric(self, metric_name: str, value: float, tags: Dict[str, Any] = None) -> None:
        """Emit a metric measurement"""
        metric = Metric(
            name=metric_name,
            value=value,
            timestamp=time.time(),
            tags=tags or {}
        )

        self.metrics[metric_name].append(metric)

        if self.config.enable_metrics:
            self.logger.debug(f"📈 Metric: {metric_name} = {value} {tags or ''}")

        # Emit to metric handlers
        await self._emit_event_to_handlers(metric, 'metric')

    async def query_metrics(self, metric_name: str, time_range: float = 300.0) -> Dict[str, Any]:
        """
        Query metrics within a time range

        Learning Note: Metric queries enable dashboards and alerting.
        Time-based aggregations reveal system trends and patterns.
        """
        metric_data = self.metrics.get(metric_name, deque())
        cutoff_time = time.time() - time_range

        recent_metrics = [m for m in metric_data if m.timestamp >= cutoff_time]

        if not recent_metrics:
            return {'count': 0, 'sum': 0, 'avg': 0, 'min': 0, 'max': 0, 'latest': 0}

        values = [m.value for m in recent_metrics]

        return {
            'count': len(recent_metrics),
            'sum': sum(values),
            'avg': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'latest': values[-1] if values else 0,
            'data': [asdict(m) for m in recent_metrics]
        }

    async def generate_health_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive system health report

        Learning Note: Health reports provide operational visibility
        into multi-agent system performance and reliability.
        """
        uptime = time.time() - self.start_time
        active_traces = [t for t in self.traces.values() if t.status == TraceStatus.ACTIVE]

        # Agent health summary
        agent_health = []
        for agent in self.agents.values():
            error_rate = agent.metrics['error_count'] / max(1, agent.metrics['tasks_completed'])
            avg_duration = agent.metrics['total_duration'] / max(1, agent.metrics['tasks_completed'])

            agent_health.append({
                'agent_id': agent.id,
                'status': agent.status,
                'tasks_completed': agent.metrics['tasks_completed'],
                'error_rate': error_rate,
                'avg_duration': avg_duration,
                'last_activity': agent.metrics['last_activity'].isoformat()
            })

        # Recent alerts
        recent_alerts = [alert for alert in self.alerts if
                        (datetime.now() - alert.timestamp).total_seconds() < 3600]

        return {
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': uptime,
            'system': {
                'traces_created': self.system_metrics['traces_created'],
                'spans_created': self.system_metrics['spans_created'],
                'events_recorded': self.system_metrics['events_recorded'],
                'alerts_triggered': self.system_metrics['alerts_triggered']
            },
            'agents': {
                'total': len(self.agents),
                'active': len([a for a in self.agents.values() if a.status == 'active']),
                'health': agent_health
            },
            'traces': {
                'active': len(active_traces),
                'total_completed': len(self.traces) - len(active_traces)
            },
            'events': {
                'total': len(self.events),
                'recent_errors': len([e for e in self.events
                                    if e.type.endswith('error') and
                                    (datetime.now() - e.timestamp).total_seconds() < 300])
            },
            'alerts': {
                'total': len(self.alerts),
                'recent': len(recent_alerts),
                'active': [asdict(alert) for alert in recent_alerts if alert.status == 'active']
            }
        }

    async def _check_alert_conditions(self, agent_id: str, error: Exception) -> None:
        """Check if alert conditions are met"""
        agent = self.agents.get(agent_id)
        if not agent:
            return

        # Check error rate
        error_rate = agent.metrics['error_count'] / max(1, agent.metrics['tasks_completed'])
        if error_rate > self.config.alert_thresholds['error_rate']:
            await self._create_alert(
                'high_error_rate',
                AlertSeverity.ERROR,
                f"High error rate for agent {agent_id}: {error_rate:.2%}",
                {'agent_id': agent_id, 'error_rate': error_rate}
            )

    async def _create_alert(self, alert_type: str, severity: AlertSeverity, message: str, context: Dict[str, Any] = None) -> str:
        """Create a new system alert"""
        alert_id = str(uuid.uuid4())
        alert = Alert(
            id=alert_id,
            type=alert_type,
            severity=severity,
            message=message,
            context=context or {}
        )

        self.alerts.append(alert)
        self.system_metrics['alerts_triggered'] += 1

        self.logger.warning(f"🚨 ALERT [{severity.value}]: {message}")

        # Emit alert event
        await self._emit_event_to_handlers(alert, 'alert')

        return alert_id

    def on(self, event_type: str, handler: Callable) -> None:
        """Register an event handler"""
        self.event_handlers[event_type].append(handler)

    async def _emit_event_to_handlers(self, event_data: Any, event_type: str = None) -> None:
        """Emit events to registered handlers"""
        if event_type is None:
            event_type = getattr(event_data, 'type', 'unknown')

        handlers = self.event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                if inspect.iscoroutinefunction(handler):
                    await handler(event_data)
                else:
                    handler(event_data)
            except Exception as error:
                self.logger.error(f"Event handler failed: {str(error)}")

    def _start_metrics_collection(self) -> None:
        """Start background metrics collection"""
        async def collect_system_metrics():
            while True:
                try:
                    # System-level metrics
                    await self._emit_metric('system.active_traces', len([t for t in self.traces.values() if t.status == TraceStatus.ACTIVE]))
                    await self._emit_metric('system.active_agents', len([a for a in self.agents.values() if a.status == 'active']))
                    await self._emit_metric('system.events_buffered', len(self.events))

                    # Agent-specific metrics
                    for agent in self.agents.values():
                        await self._emit_metric('agent.tasks_completed', agent.metrics['tasks_completed'], {'agent_id': agent.id})
                        await self._emit_metric('agent.error_count', agent.metrics['error_count'], {'agent_id': agent.id})

                    await asyncio.sleep(self.config.metrics_interval)

                except Exception as error:
                    self.logger.error(f"Metrics collection failed: {str(error)}")
                    await asyncio.sleep(self.config.metrics_interval)

        task = asyncio.create_task(collect_system_metrics())
        self._background_tasks.append(task)

    def _extract_event_tags(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Extract tags from event data"""
        tags = {}
        if 'agent_id' in data:
            tags['agent'] = str(data['agent_id'])
        if 'trace_id' in data:
            tags['trace'] = str(data['trace_id'])
        if 'method' in data:
            tags['method'] = str(data['method'])
        return tags

    def _log_event(self, event: Event) -> None:
        """Log an event with appropriate level"""
        level = logging.INFO
        if event.type.endswith('error'):
            level = logging.ERROR
        elif event.type.endswith('warning'):
            level = logging.WARNING

        self.logger.log(level, f"Event: {event.type} - {json.dumps(event.data, default=str)}")

    # Synchronous trace methods for non-async contexts
    def _start_trace_sync(self, agent_id: str, operation: str) -> str:
        """Start a trace synchronously (for non-async methods)"""
        trace_id = str(uuid.uuid4())
        trace = Trace(
            id=trace_id,
            agent_id=agent_id,
            operation=operation,
            start_time=time.time()
        )
        self.traces[trace_id] = trace
        return trace_id

    def _end_trace_sync(self, trace_id: str, success: bool, error: str = None) -> None:
        """End a trace synchronously"""
        trace = self.traces.get(trace_id)
        if trace:
            trace.end_time = time.time()
            trace.duration = trace.end_time - trace.start_time
            trace.status = TraceStatus.COMPLETED if success else TraceStatus.FAILED
            if error:
                trace.error = error

    def export_traces(self, filter_criteria: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Export traces for external analysis"""
        traces = list(self.traces.values())

        if filter_criteria:
            filtered_traces = []
            for trace in traces:
                if all(getattr(trace, key, None) == value for key, value in filter_criteria.items()):
                    filtered_traces.append(trace)
            traces = filtered_traces

        return [asdict(trace) for trace in traces]

    async def cleanup(self) -> None:
        """Clean up resources and stop background tasks"""
        for task in self._background_tasks:
            task.cancel()

        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self.logger.info("🧹 Telemetry system cleaned up")

    def __del__(self):
        """Ensure cleanup on garbage collection"""
        if hasattr(self, '_background_tasks'):
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()