# Tutorial: Building Your First Python Async Agent

## What You'll Learn

By the end of this tutorial, you'll understand:
- How to design async autonomous agents with proper type safety
- Event-driven communication patterns using AsyncIO
- Resource management with async context managers
- Distributed tracing and observability for debugging
- Performance optimization with concurrent processing

## Why Python for Multi-Agent Systems?

Python's async ecosystem provides several advantages over other approaches:

**Native Concurrency**: `async/await` syntax makes concurrent operations natural
**Type Safety**: Type hints catch integration errors at development time
**Clean Resource Management**: Context managers ensure proper cleanup
**Rich Ecosystem**: Easy integration with ML libraries and external APIs
**Production Ready**: Battle-tested frameworks for scaling

## Core Architecture Pattern

```python
@dataclass
class Signal:
    """Type-safe data structure"""
    id: str
    content: str
    timestamp: datetime
    score: float = 0.0

class SignalsScoutAgent:
    """Async autonomous agent"""
    async def discover_signals(self, sources: List[str]) -> List[Signal]:
        async with self.telemetry.trace('discovery', 'scan_sources'):
            tasks = [self._scan_source(s) for s in sources]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return self._process_results(results)
```

## Step 1: Define Your Agent Interface

Start with clear type definitions:

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
import logging

@dataclass
class AgentCapability:
    """Define what the agent can do"""
    name: str
    description: str
    input_type: type
    output_type: type

@dataclass
class AgentConfig:
    """Agent configuration with defaults"""
    timeout: float = 30.0
    max_concurrent_tasks: int = 5
    retry_attempts: int = 3
    enable_caching: bool = True

class BaseAgent:
    """Base class for all agents"""
    def __init__(self, agent_id: str, config: AgentConfig = None):
        self.id = agent_id
        self.config = config or AgentConfig()
        self.capabilities: List[AgentCapability] = []
        self.logger = logging.getLogger(f'agent.{agent_id}')

    async def start(self) -> None:
        """Start the agent"""
        self.logger.info(f"🤖 Agent {self.id} starting")

    async def stop(self) -> None:
        """Stop the agent gracefully"""
        self.logger.info(f"🛑 Agent {self.id} stopping")
```

**Learning Point**: Type hints make your agent interfaces self-documenting and catch integration errors early.

## Step 2: Implement Async I/O Operations

```python
import aiohttp
import asyncio
from typing import AsyncGenerator

class SignalsScoutAgent(BaseAgent):
    def __init__(self, config: AgentConfig = None):
        super().__init__('signals-scout', config)
        self.session: Optional[aiohttp.ClientSession] = None

    async def start(self) -> None:
        await super().start()
        # Initialize HTTP session for external API calls
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )

    async def stop(self) -> None:
        if self.session:
            await self.session.close()
        await super().stop()

    async def discover_signals(self, sources: List[str],
                             timeframe: str = '24h') -> List[Signal]:
        """Main discovery method with proper error handling"""
        if not self.session:
            raise RuntimeError("Agent not started. Call agent.start() first.")

        self.logger.info(f"🔍 Discovering signals from {len(sources)} sources")

        # Create concurrent tasks for each source
        tasks = [
            self._scan_source_with_timeout(source, timeframe)
            for source in sources
        ]

        # Execute concurrently with proper error handling
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle failures gracefully
        all_signals = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.warning(f"Source {sources[i]} failed: {result}")
            else:
                all_signals.extend(result)

        self.logger.info(f"📊 Found {len(all_signals)} total signals")
        return all_signals

    async def _scan_source_with_timeout(self, source: str,
                                       timeframe: str) -> List[Signal]:
        """Scan with timeout and retry logic"""
        for attempt in range(self.config.retry_attempts):
            try:
                return await asyncio.wait_for(
                    self._scan_source(source, timeframe),
                    timeout=self.config.timeout
                )
            except asyncio.TimeoutError:
                self.logger.warning(f"Timeout scanning {source} (attempt {attempt + 1})")
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff

        raise TimeoutError(f"Failed to scan {source} after {self.config.retry_attempts} attempts")

    async def _scan_source(self, source: str, timeframe: str) -> List[Signal]:
        """Actual source scanning implementation"""
        # Simulate API call
        await asyncio.sleep(random.uniform(0.1, 0.5))

        # Mock data generation (replace with real API calls)
        mock_signals = [
            Signal(
                id=f"{source}-{i}",
                content=f"Signal {i} from {source}",
                timestamp=datetime.now(),
                score=random.uniform(0.3, 0.9)
            )
            for i in range(random.randint(1, 5))
        ]

        self.logger.debug(f"📡 {source}: found {len(mock_signals)} signals")
        return mock_signals
```

**Learning Point**: Proper async resource management prevents connection leaks and ensures graceful shutdown.

## Step 3: Add Distributed Tracing

```python
from contextlib import asynccontextmanager
import uuid
import time

@dataclass
class Trace:
    id: str
    operation: str
    start_time: float
    agent_id: str
    context: Dict[str, Any] = field(default_factory=dict)

class TelemetryMixin:
    """Mixin for adding observability to agents"""

    @asynccontextmanager
    async def trace(self, operation: str, context: Dict[str, Any] = None):
        """Async context manager for distributed tracing"""
        trace_id = str(uuid.uuid4())
        trace = Trace(
            id=trace_id,
            operation=operation,
            start_time=time.time(),
            agent_id=self.id,
            context=context or {}
        )

        self.logger.info(f"🔍 Started trace: {trace_id} [{operation}]")

        try:
            yield trace

            duration = time.time() - trace.start_time
            self.logger.info(f"✅ Completed trace: {trace_id} ({duration:.3f}s)")

        except Exception as error:
            duration = time.time() - trace.start_time
            self.logger.error(f"❌ Failed trace: {trace_id} ({duration:.3f}s): {error}")
            raise

# Enhanced agent with observability
class ObservableSignalsScoutAgent(BaseAgent, TelemetryMixin):
    async def discover_signals(self, sources: List[str],
                             timeframe: str = '24h') -> List[Signal]:
        # Wrap entire operation in a trace
        async with self.trace('discover_signals',
                            {'sources': sources, 'timeframe': timeframe}) as trace:

            all_signals = []

            # Trace each source scan
            for source in sources:
                async with self.trace(f'scan_{source}', {'source': source}):
                    signals = await self._scan_source(source, timeframe)
                    all_signals.extend(signals)

            # Add results to trace context
            trace.context['signals_found'] = len(all_signals)

            return all_signals
```

**Learning Point**: Async context managers provide elegant resource management and automatic cleanup for distributed tracing.

## Step 4: Agent Communication and Events

```python
from typing import Callable, Awaitable
from collections import defaultdict

class EventBus:
    """Simple async event bus for agent communication"""

    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = defaultdict(list)

    def subscribe(self, event_type: str, handler: Callable[[Dict], Awaitable[None]]):
        """Subscribe to an event type"""
        self._handlers[event_type].append(handler)

    async def emit(self, event_type: str, data: Dict[str, Any]):
        """Emit an event to all subscribers"""
        handlers = self._handlers.get(event_type, [])

        if handlers:
            # Execute all handlers concurrently
            tasks = [handler(data) for handler in handlers]
            await asyncio.gather(*tasks, return_exceptions=True)

# Event-driven agent
class EventDrivenAgent(ObservableSignalsScoutAgent):
    def __init__(self, event_bus: EventBus, config: AgentConfig = None):
        super().__init__(config)
        self.event_bus = event_bus

        # Subscribe to relevant events
        self.event_bus.subscribe('workflow:started', self._handle_workflow_started)
        self.event_bus.subscribe('signals:requested', self._handle_signals_requested)

    async def discover_signals(self, sources: List[str],
                             timeframe: str = '24h') -> List[Signal]:
        signals = await super().discover_signals(sources, timeframe)

        # Emit event for other agents
        await self.event_bus.emit('signals:discovered', {
            'agent_id': self.id,
            'signals': [signal.__dict__ for signal in signals],
            'count': len(signals),
            'timestamp': datetime.now().isoformat()
        })

        return signals

    async def _handle_workflow_started(self, data: Dict[str, Any]):
        """Handle workflow started event"""
        workflow_id = data.get('workflow_id')
        self.logger.info(f"🏁 Responding to workflow: {workflow_id}")

        # Could automatically start signal discovery
        if data.get('auto_discovery', False):
            await self.discover_signals(['reddit', 'twitter'])

    async def _handle_signals_requested(self, data: Dict[str, Any]):
        """Handle direct signal request"""
        requester = data.get('requester')
        sources = data.get('sources', ['reddit'])

        self.logger.info(f"📨 Signal request from {requester}")
        signals = await self.discover_signals(sources)

        # Send response event
        await self.event_bus.emit('signals:response', {
            'requester': requester,
            'signals': [s.__dict__ for s in signals],
            'provider': self.id
        })
```

**Learning Point**: Event-driven communication creates loose coupling between agents while maintaining coordination.

## Step 5: Error Handling and Resilience

```python
import asyncio
from typing import TypeVar, Generic
from enum import Enum

T = TypeVar('T')

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open" # Testing if service recovered

class CircuitBreaker(Generic[T]):
    """Circuit breaker pattern for resilient agent operations"""

    def __init__(self, failure_threshold: int = 5,
                 recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = CircuitState.CLOSED

    async def call(self, func: Callable[[], Awaitable[T]],
                  *args, **kwargs) -> T:
        """Execute function with circuit breaker protection"""

        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitBreakerOpenError("Circuit breaker is open")

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result

        except Exception as error:
            self._on_failure()
            raise

    def _on_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

# Resilient agent with circuit breaker
class ResilientAgent(EventDrivenAgent):
    def __init__(self, event_bus: EventBus, config: AgentConfig = None):
        super().__init__(event_bus, config)
        self.circuit_breaker = CircuitBreaker(failure_threshold=3)

    async def _scan_source(self, source: str, timeframe: str) -> List[Signal]:
        """Scan source with circuit breaker protection"""
        return await self.circuit_breaker.call(
            super()._scan_source, source, timeframe
        )
```

**Learning Point**: Circuit breakers prevent cascade failures and improve system resilience under load.

## Step 6: Performance Optimization

```python
import asyncio
from asyncio import Semaphore
from typing import Dict, Any
import time

class PerformanceOptimizedAgent(ResilientAgent):
    def __init__(self, event_bus: EventBus, config: AgentConfig = None):
        super().__init__(event_bus, config)

        # Concurrency control
        self.semaphore = Semaphore(config.max_concurrent_tasks)

        # Simple in-memory cache
        self.cache: Dict[str, Any] = {}
        self.cache_ttl = 300  # 5 minutes

    async def discover_signals(self, sources: List[str],
                             timeframe: str = '24h') -> List[Signal]:
        """Optimized discovery with caching and rate limiting"""

        cache_key = f"signals:{':'.join(sources)}:{timeframe}"

        # Check cache first
        if cached := self._get_cached(cache_key):
            self.logger.info(f"📦 Cache hit for {cache_key}")
            return cached

        # Rate limit concurrent operations
        async with self.semaphore:
            signals = await super().discover_signals(sources, timeframe)

        # Cache results
        self._set_cache(cache_key, signals)

        return signals

    def _get_cached(self, key: str) -> Optional[List[Signal]]:
        """Get from cache if not expired"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.cache_ttl:
                return data
            else:
                del self.cache[key]
        return None

    def _set_cache(self, key: str, data: List[Signal]) -> None:
        """Set cache with timestamp"""
        self.cache[key] = (data, time.time())

    async def batch_process_signals(self, signal_batches: List[List[Signal]],
                                  batch_size: int = 10) -> List[Signal]:
        """Process signals in batches for better throughput"""

        async def process_batch(batch: List[Signal]) -> List[Signal]:
            async with self.semaphore:
                # Simulate processing
                await asyncio.sleep(0.1)
                return [s for s in batch if s.score > 0.5]

        # Create batches
        batches = [
            signal_batches[i:i + batch_size]
            for i in range(0, len(signal_batches), batch_size)
        ]

        # Process batches concurrently
        batch_tasks = [process_batch(batch) for batch in batches]
        results = await asyncio.gather(*batch_tasks)

        # Flatten results
        return [signal for batch_result in results for signal in batch_result]
```

**Learning Point**: Semaphores control concurrency, caching reduces redundant work, and batching improves throughput.

## Running Your Agent

```python
async def main():
    """Complete example of running an async agent"""

    # Setup
    event_bus = EventBus()
    config = AgentConfig(max_concurrent_tasks=3, timeout=10.0)

    # Create and start agent
    agent = PerformanceOptimizedAgent(event_bus, config)
    await agent.start()

    try:
        # Discover signals
        signals = await agent.discover_signals(['reddit', 'twitter', 'news'])
        print(f"Found {len(signals)} signals")

        # Show results
        for signal in signals[:3]:
            print(f"  {signal.id}: {signal.score:.2f}")

    finally:
        # Always cleanup
        await agent.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

## 🎯 Key Takeaways

1. **Type Safety First**: Use dataclasses and type hints for clear interfaces
2. **Async Context Managers**: Elegant resource management and observability
3. **Graceful Error Handling**: Circuit breakers and retries for resilience
4. **Event-Driven Communication**: Loose coupling with high coordination
5. **Performance Optimization**: Semaphores, caching, and batching for scale

**The Golden Rule**: Design for async from the start—it's much easier than retrofitting sync code.

Your agent is now production-ready with proper async patterns, observability, and resilience! 🚀