# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **production-grade multi-agent venture studio system** built with modern Python async patterns. The system demonstrates autonomous agents that discover market signals, synthesize business theses, and coordinate complex workflows through an orchestrator.

### Core Architecture

The system follows a **hierarchical multi-agent architecture**:

1. **VentureStudioOrchestrator** (`src/coordination/orchestrator.py`) - Central coordinator that manages workflows through stage gates, delegates tasks to agents, and enforces business validation rules
2. **Autonomous Agents** (`src/agents/`) - Specialized agents that handle specific capabilities:
   - `SignalsScoutAgent` - Discovers and scores market signals from multiple sources concurrently
   - `ThesisSynthesizerAgent` - Clusters signals and generates validated business theses
3. **AgentTelemetrySystem** (`src/observability/telemetry.py`) - Distributed tracing and performance monitoring using async context managers
4. **Event-Driven Communication** - Agents communicate through async event handlers and direct method calls

### Key Design Patterns

- **Async-First Design**: All I/O operations use `async/await` with proper resource management
- **Type-Safe State Management**: Dataclasses with type hints for all agent state and communication
- **Context Managers**: Automatic resource cleanup and distributed tracing using `async with`
- **Stage Gate Validation**: Evidence-based decision making with configurable thresholds
- **Circuit Breaker Pattern**: Fault tolerance and graceful degradation under load

## Development Commands

### Running the System
```bash
# Run complete demo with all scenarios
python examples/python_demo.py

# Run specific demo components interactively
python -c "from examples.python_demo import *; asyncio.run(interactive_demo())"

# Run with debug logging
python examples/python_demo.py 2>&1 | grep -E "(ERROR|INFO|WARNING)"
```

### Development Setup
```bash
# No external dependencies required for core functionality
# Optional enhanced features:
pip install -r requirements.txt

# Verify Python version (requires 3.8+)
python -c "import sys; assert sys.version_info >= (3, 8), 'Python 3.8+ required'"
```

### Testing (when implemented)
```bash
# Unit tests for individual agents
python -m pytest tests/unit/ -v

# Integration tests for multi-agent coordination
python -m pytest tests/integration/ -v

# Performance benchmarks
python -m pytest tests/performance/ -v

# Run specific agent tests
python -m pytest tests/unit/test_signals_scout.py -v
```

### Code Quality (when tooling is configured)
```bash
# Type checking
mypy src/ examples/

# Code formatting
black src/ examples/ tests/

# Linting
flake8 src/ examples/ tests/
```

## Agent Development Patterns

### Creating New Agents
1. **Inherit from base patterns** - Follow the async agent structure in existing agents
2. **Define capabilities** - List all async methods the agent can perform in `self.capabilities`
3. **Implement proper typing** - Use dataclasses for state and type hints for all methods
4. **Add telemetry registration** - Register with both orchestrator and telemetry system
5. **Handle errors gracefully** - Use try/except with exponential backoff for external I/O

### Orchestrator Integration
- **Register agents** with `orchestrator.register_agent(agent_id, agent_instance, capabilities)`
- **Define stage gates** using `StageGate` dataclass with evidence requirements and thresholds
- **Delegate tasks** using `orchestrator._delegate_task()` with timeout and retry logic
- **Emit events** for cross-agent communication using the event bus pattern

### Observability Integration
- **Wrap operations** with `async with telemetry.trace()` context managers
- **Record events** using `await telemetry._record_event()` for audit trails
- **Emit metrics** using `await telemetry._emit_metric()` for performance tracking
- **Generate health reports** using `telemetry.generate_health_report()` for system monitoring

## Architecture Extension Points

### Adding New Agent Types
- Create new agent class in `src/agents/` following existing patterns
- Implement required capabilities as async methods
- Register with orchestrator and telemetry system in demo setup
- Add to workflow stages or create new stage logic in orchestrator

### Adding New Workflow Stages
- Define new stage processing method in `VentureStudioOrchestrator`
- Create corresponding `StageGate` with validation rules
- Update `_get_next_stage()` method to include new stage in sequence
- Add stage-specific event handlers for monitoring

### Extending Observability
- Add new metric types in `telemetry.py` using the `Metric` dataclass
- Create custom alert conditions in `_check_alert_conditions()`
- Extend health reporting with additional system metrics
- Add new trace context data for specific agent operations

## Performance Characteristics

The system achieves **~5.3 operations/second** with **26+ signals processed/second** in concurrent load testing. Async concurrency allows 10+ simultaneous agent operations with proper resource management through semaphores and connection pools.