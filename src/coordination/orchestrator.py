"""
Multi-Agent Orchestrator: Python Implementation

This orchestrator demonstrates production-grade multi-agent coordination patterns:
- Async/await for proper concurrency
- Type hints for better code clarity
- Dataclasses for clean state management
- Context managers for resource management

Learning Focus: How Python's async ecosystem makes multi-agent
coordination more natural and robust.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import json
from datetime import datetime, timedelta
import uuid


class WorkflowStatus(Enum):
    INITIATED = "initiated"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StageStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Evidence:
    """Evidence collected during workflow execution"""
    type: str
    data: Dict[str, Any]
    timestamp: datetime
    quality_score: float = 0.0
    source_agent: Optional[str] = None


@dataclass
class Decision:
    """Decision made during workflow execution"""
    stage: str
    verdict: str
    rationale: str
    reviewer: str
    timestamp: datetime
    confidence: float = 0.0


@dataclass
class WorkflowMetrics:
    """Performance metrics for a workflow"""
    stages_completed: int = 0
    experiments_run: int = 0
    cost_spent: float = 0.0
    total_duration: float = 0.0
    error_count: int = 0


@dataclass
class Workflow:
    """Complete workflow state"""
    id: str
    type: str
    status: WorkflowStatus
    current_stage: str
    data: Dict[str, Any]
    start_time: datetime
    end_time: Optional[datetime] = None
    last_activity: datetime = field(default_factory=datetime.now)
    evidence: List[Evidence] = field(default_factory=list)
    decisions: List[Decision] = field(default_factory=list)
    metrics: WorkflowMetrics = field(default_factory=WorkflowMetrics)


@dataclass
class StageGate:
    """Stage gate validation rules"""
    required_evidence: List[str]
    thresholds: Dict[str, float]
    validators: List[str]
    timeout: float = 300.0  # 5 minutes default


@dataclass
class AgentCapability:
    """Agent capability definition"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    timeout: float = 30.0


class VentureStudioOrchestrator:
    """
    Advanced Python orchestrator with proper async patterns

    Learning Note: Python's async/await makes coordination much cleaner
    than callback-based systems. Type hints improve debugging and IDE support.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {
            'max_concurrent_workflows': 10,
            'default_timeout': 30.0,
            'retry_attempts': 3,
            'stage_timeout': 600.0,  # 10 minutes
            **(config or {})
        }

        # Core state
        self.agents: Dict[str, Any] = {}
        self.workflows: Dict[str, Workflow] = {}
        self.stage_gates: Dict[str, StageGate] = {}

        # Event system
        self.event_handlers: Dict[str, List[Callable]] = {}

        # Performance tracking
        self.metrics = {
            'workflows_started': 0,
            'workflows_completed': 0,
            'workflows_failed': 0,
            'total_processing_time': 0.0,
            'average_workflow_duration': 0.0
        }

        # Setup logging
        self.logger = logging.getLogger('orchestrator')
        self.logger.setLevel(logging.INFO)

        self.logger.info("🎯 Python Orchestrator initialized")

    async def register_agent(self, agent_id: str, agent: Any, capabilities: List[str] = None) -> None:
        """
        Register an agent with the orchestrator

        Learning Note: Type hints make the interface clear and enable
        better IDE support and runtime validation.
        """
        if agent_id in self.agents:
            raise ValueError(f"Agent {agent_id} already registered")

        # Wrap agent with monitoring
        wrapped_agent = await self._wrap_agent_with_telemetry(agent_id, agent)

        self.agents[agent_id] = {
            'instance': wrapped_agent,
            'capabilities': capabilities or [],
            'registered_at': datetime.now(),
            'status': 'active',
            'metrics': {
                'tasks_completed': 0,
                'tasks_failed': 0,
                'total_duration': 0.0,
                'last_activity': datetime.now()
            }
        }

        self.logger.info(f"🤖 Registered agent: {agent_id}")
        await self._emit_event('agent_registered', {
            'agent_id': agent_id,
            'capabilities': capabilities or []
        })

    def define_stage_gate(self, stage_name: str, gate: StageGate) -> None:
        """Define validation rules for a stage gate"""
        self.stage_gates[stage_name] = gate
        self.logger.info(f"🚪 Defined stage gate: {stage_name}")

    async def start_workflow(self, workflow_type: str, initial_data: Dict[str, Any]) -> str:
        """
        Start a new venture workflow

        Learning Note: Async context managers and proper resource management
        prevent memory leaks and ensure cleanup in distributed systems.
        """
        # Check capacity
        if len(self.workflows) >= self.config['max_concurrent_workflows']:
            raise RuntimeError("Maximum concurrent workflows reached")

        workflow_id = f"{workflow_type}-{int(time.time())}-{uuid.uuid4().hex[:8]}"

        workflow = Workflow(
            id=workflow_id,
            type=workflow_type,
            status=WorkflowStatus.INITIATED,
            current_stage='signals',
            data=initial_data,
            start_time=datetime.now()
        )

        self.workflows[workflow_id] = workflow
        self.metrics['workflows_started'] += 1

        self.logger.info(f"🚀 Starting workflow: {workflow_id} ({workflow_type})")
        await self._emit_event('workflow_started', {
            'workflow_id': workflow_id,
            'type': workflow_type,
            'data': initial_data
        })

        # Begin first stage asynchronously
        asyncio.create_task(self._process_stage(workflow_id, 'signals'))

        return workflow_id

    async def _process_stage(self, workflow_id: str, stage_name: str) -> None:
        """
        Process a specific stage of the workflow

        Learning Note: Proper exception handling and timeout management
        are crucial for resilient multi-agent systems.
        """
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            self.logger.error(f"Workflow {workflow_id} not found")
            return

        self.logger.info(f"📋 Processing stage: {stage_name} for workflow: {workflow_id}")

        try:
            # Update workflow state
            workflow.current_stage = stage_name
            workflow.status = WorkflowStatus.PROCESSING
            workflow.last_activity = datetime.now()

            await self._emit_event('stage_started', {
                'workflow_id': workflow_id,
                'stage_name': stage_name
            })

            # Execute stage with timeout
            stage_timeout = self.config['stage_timeout']

            if stage_name == 'signals':
                await asyncio.wait_for(
                    self._process_signals_stage(workflow_id),
                    timeout=stage_timeout
                )
            elif stage_name == 'validation':
                await asyncio.wait_for(
                    self._process_validation_stage(workflow_id),
                    timeout=stage_timeout
                )
            elif stage_name == 'mvp':
                await asyncio.wait_for(
                    self._process_mvp_stage(workflow_id),
                    timeout=stage_timeout
                )
            elif stage_name == 'scaling':
                await asyncio.wait_for(
                    self._process_scaling_stage(workflow_id),
                    timeout=stage_timeout
                )
            else:
                raise ValueError(f"Unknown stage: {stage_name}")

            # Validate stage gate
            can_proceed = await self._validate_stage_gate(workflow_id, stage_name)

            if can_proceed:
                workflow.metrics.stages_completed += 1
                await self._emit_event('stage_completed', {
                    'workflow_id': workflow_id,
                    'stage_name': stage_name
                })

                # Proceed to next stage
                next_stage = self._get_next_stage(stage_name)
                if next_stage:
                    await self._process_stage(workflow_id, next_stage)
                else:
                    await self._complete_workflow(workflow_id)
            else:
                await self._handle_stage_failure(workflow_id, stage_name)

        except asyncio.TimeoutError:
            self.logger.error(f"Stage {stage_name} timed out for workflow {workflow_id}")
            await self._handle_stage_failure(workflow_id, stage_name, "timeout")

        except Exception as error:
            self.logger.error(f"Error in stage {stage_name}: {str(error)}")
            workflow.metrics.error_count += 1
            await self._emit_event('stage_error', {
                'workflow_id': workflow_id,
                'stage_name': stage_name,
                'error': str(error)
            })
            await self._handle_stage_failure(workflow_id, stage_name, str(error))

    async def _process_signals_stage(self, workflow_id: str) -> None:
        """Process the signals discovery stage"""
        workflow = self.workflows[workflow_id]

        # Create concurrent tasks for signal discovery
        tasks = [
            self._delegate_task('signals-scout', 'discover_signals', {
                'sources': ['reddit', 'twitter', 'news'],
                'timeframe': '24h'
            }),
            self._delegate_task('thesis-synthesizer', 'cluster_signals', {
                'existing_signals': workflow.data.get('signals', [])
            })
        ]

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        signals = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.warning(f"Task failed: {str(result)}")
            elif isinstance(result, list):
                signals.extend(result)

        # Store signals in workflow
        workflow.data['signals'] = signals

        # Add evidence
        evidence = Evidence(
            type='signals_discovered',
            data={'count': len(signals), 'sources': ['reddit', 'twitter', 'news']},
            timestamp=datetime.now(),
            quality_score=self._assess_signal_quality(signals),
            source_agent='signals-scout'
        )
        workflow.evidence.append(evidence)

        self.logger.info(f"📡 Discovered {len(signals)} signals for workflow: {workflow_id}")

    async def _delegate_task(self, agent_id: str, method: str, params: Dict[str, Any],
                           timeout: Optional[float] = None) -> Any:
        """
        Delegate a task to a specific agent with proper error handling

        Learning Note: Async task delegation with proper timeout and retry
        logic prevents hanging workflows and improves system resilience.
        """
        agent_info = self.agents.get(agent_id)
        if not agent_info:
            raise ValueError(f"Agent {agent_id} not found")

        agent = agent_info['instance']
        task_timeout = timeout or self.config['default_timeout']
        retry_attempts = self.config['retry_attempts']

        last_error = None

        for attempt in range(1, retry_attempts + 1):
            try:
                self.logger.info(f"🔄 Delegating {method} to {agent_id} (attempt {attempt}/{retry_attempts})")

                # Get method from agent
                if not hasattr(agent, method):
                    raise AttributeError(f"Agent {agent_id} does not have method {method}")

                agent_method = getattr(agent, method)

                # Execute with timeout
                start_time = time.time()
                result = await asyncio.wait_for(agent_method(**params), timeout=task_timeout)
                duration = time.time() - start_time

                # Update metrics
                agent_info['metrics']['tasks_completed'] += 1
                agent_info['metrics']['total_duration'] += duration
                agent_info['metrics']['last_activity'] = datetime.now()

                self.logger.info(f"✅ Task {method} completed in {duration:.2f}s")
                return result

            except asyncio.TimeoutError:
                last_error = f"Task {method} timed out after {task_timeout}s"
                self.logger.warning(f"⏰ {last_error} (attempt {attempt})")

            except Exception as error:
                last_error = f"Task {method} failed: {str(error)}"
                self.logger.warning(f"⚠️ {last_error} (attempt {attempt})")

            # Exponential backoff before retry
            if attempt < retry_attempts:
                await asyncio.sleep(2 ** attempt)

        # All attempts failed
        agent_info['metrics']['tasks_failed'] += 1
        raise RuntimeError(f"Task {method} failed after {retry_attempts} attempts: {last_error}")

    async def _validate_stage_gate(self, workflow_id: str, stage_name: str) -> bool:
        """
        Validate stage gate requirements

        Learning Note: Evidence-based validation ensures quality control
        and prevents bad ideas from advancing through the pipeline.
        """
        workflow = self.workflows[workflow_id]
        gate = self.stage_gates.get(stage_name)

        if not gate:
            self.logger.warning(f"⚠️ No stage gate defined for {stage_name}, allowing passage")
            return True

        self.logger.info(f"🔍 Validating stage gate: {stage_name}")

        # Check required evidence
        for evidence_type in gate.required_evidence:
            has_evidence = any(e.type == evidence_type for e in workflow.evidence)
            if not has_evidence:
                self.logger.warning(f"❌ Missing required evidence: {evidence_type}")
                return False

        # Check thresholds
        for metric, threshold in gate.thresholds.items():
            actual_value = self._get_workflow_metric(workflow, metric)
            if actual_value < threshold:
                self.logger.warning(f"❌ Metric {metric} ({actual_value}) below threshold ({threshold})")
                return False

        # Run validator agents
        for validator_id in gate.validators:
            try:
                is_valid = await self._delegate_task(validator_id, 'validate', {
                    'workflow': workflow.__dict__,
                    'stage': stage_name
                })

                if not is_valid:
                    self.logger.warning(f"❌ Validator {validator_id} rejected workflow")
                    return False

            except Exception as error:
                self.logger.error(f"Validator {validator_id} failed: {str(error)}")
                return False

        self.logger.info(f"✅ Stage gate passed: {stage_name}")
        return True

    async def _complete_workflow(self, workflow_id: str) -> None:
        """Complete a workflow successfully"""
        workflow = self.workflows[workflow_id]
        workflow.status = WorkflowStatus.COMPLETED
        workflow.end_time = datetime.now()
        workflow.metrics.total_duration = (workflow.end_time - workflow.start_time).total_seconds()

        self.metrics['workflows_completed'] += 1
        self.metrics['total_processing_time'] += workflow.metrics.total_duration
        self.metrics['average_workflow_duration'] = (
            self.metrics['total_processing_time'] / max(1, self.metrics['workflows_completed'])
        )

        self.logger.info(f"🎉 Workflow completed: {workflow_id}")
        await self._emit_event('workflow_completed', {
            'workflow_id': workflow_id,
            'duration': workflow.metrics.total_duration,
            'stages_completed': workflow.metrics.stages_completed
        })

    async def _handle_stage_failure(self, workflow_id: str, stage_name: str, reason: str = "validation_failed") -> None:
        """Handle stage failure with appropriate recovery"""
        workflow = self.workflows[workflow_id]
        workflow.status = WorkflowStatus.FAILED

        decision = Decision(
            stage=stage_name,
            verdict="failed",
            rationale=f"Stage failed: {reason}",
            reviewer="orchestrator",
            timestamp=datetime.now(),
            confidence=1.0
        )
        workflow.decisions.append(decision)

        self.metrics['workflows_failed'] += 1

        self.logger.warning(f"🔄 Handling stage failure: {stage_name} for workflow: {workflow_id}")
        await self._emit_event('workflow_failed', {
            'workflow_id': workflow_id,
            'failed_stage': stage_name,
            'reason': reason
        })

    # Event system
    async def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event to registered handlers"""
        handlers = self.event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as error:
                self.logger.error(f"Event handler failed: {str(error)}")

    def on(self, event_type: str, handler: Callable) -> None:
        """Register an event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)

    # Utility methods
    async def _wrap_agent_with_telemetry(self, agent_id: str, agent: Any) -> Any:
        """Wrap agent methods with telemetry (simplified for demo)"""
        # In production, this would use a more sophisticated proxy
        return agent

    def _get_next_stage(self, current_stage: str) -> Optional[str]:
        """Get the next stage in the workflow"""
        stages = ['signals', 'validation', 'mvp', 'scaling']
        try:
            current_index = stages.index(current_stage)
            return stages[current_index + 1] if current_index < len(stages) - 1 else None
        except ValueError:
            return None

    def _assess_signal_quality(self, signals: List[Any]) -> float:
        """Assess the quality of discovered signals"""
        if not signals:
            return 0.0
        return min(1.0, len(signals) / 10.0)  # Simple quality metric

    def _get_workflow_metric(self, workflow: Workflow, metric: str) -> float:
        """Extract metric value from workflow state"""
        if metric == 'signal_count':
            return len(workflow.data.get('signals', []))
        elif metric == 'evidence_quality':
            if not workflow.evidence:
                return 0.0
            return sum(e.quality_score for e in workflow.evidence) / len(workflow.evidence)
        else:
            return getattr(workflow.metrics, metric, 0.0)

    # Placeholder stage implementations
    async def _process_validation_stage(self, workflow_id: str) -> None:
        """TODO: Implement validation stage"""
        self.logger.info(f"⚠️ Validation stage not yet implemented for {workflow_id}")

    async def _process_mvp_stage(self, workflow_id: str) -> None:
        """TODO: Implement MVP stage"""
        self.logger.info(f"⚠️ MVP stage not yet implemented for {workflow_id}")

    async def _process_scaling_stage(self, workflow_id: str) -> None:
        """TODO: Implement scaling stage"""
        self.logger.info(f"⚠️ Scaling stage not yet implemented for {workflow_id}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        active_workflows = [w for w in self.workflows.values() if w.status == WorkflowStatus.PROCESSING]

        return {
            'agents': {
                'total': len(self.agents),
                'active': len([a for a in self.agents.values() if a['status'] == 'active'])
            },
            'workflows': {
                'total': len(self.workflows),
                'active': len(active_workflows),
                'completed': len([w for w in self.workflows.values() if w.status == WorkflowStatus.COMPLETED]),
                'failed': len([w for w in self.workflows.values() if w.status == WorkflowStatus.FAILED])
            },
            'metrics': self.metrics
        }