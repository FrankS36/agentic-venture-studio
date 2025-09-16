# Observability & Debugging: Python Async Multi-Agent Systems

## Why Observability Matters

Multi-agent systems are **inherently complex**. Unlike single-threaded applications, you have:

- **Multiple autonomous entities** making decisions independently
- **Asynchronous communication** across agent boundaries
- **Emergent behaviors** that arise from agent interactions
- **Distributed state** spread across multiple components

Without proper observability, debugging feels like trying to understand a conversation by hearing only one side.

## The Three Pillars of Observability

### 1. Metrics: What is happening?

**Quantitative measurements** that answer "How much?" and "How fast?"

```javascript
// System metrics
system.memory.heap_used: 245MB
system.active_agents: 5
system.active_traces: 23

// Business metrics
ideas.processed_per_hour: 47
experiments.success_rate: 0.15
cost.per_validated_idea: $23.50

// Performance metrics
agent.response_time.avg: 1.2s
workflow.completion_rate: 0.85
error.rate: 0.03
```

**Learning Point**: Metrics reveal **trends and patterns** over time. A single metric is data; a metric over time is insight.

### 2. Traces: What happened and why?

**Distributed tracing** follows requests/workflows across multiple agents.

```
Trace: validate-idea-12345
├─ SignalsScout.discoverSignals (230ms)
│  ├─ scanSource("reddit") (120ms)
│  ├─ scanSource("twitter") (95ms)
│  └─ scoreSignals() (15ms)
├─ ThesisSynthesizer.clusterSignals (340ms)
│  ├─ semanticClustering() (200ms)
│  ├─ synthesizeTheses() (100ms)
│  └─ validateTheses() (40ms)
└─ ProblemValidator.runSurveys (2.1s)
   ├─ createSurvey() (200ms)
   ├─ distributeToPanel() (1.8s) ← SLOW!
   └─ analyzeResponses() (100ms)
```

**Learning Point**: Traces reveal **causality and bottlenecks**. They show not just what happened, but the sequence and dependencies.

### 3. Logs: What was the context?

**Structured events** with rich context for detailed investigation.

```javascript
{
  "timestamp": "2025-01-15T10:30:45Z",
  "level": "INFO",
  "event": "agent.task.started",
  "agent_id": "signals-scout",
  "trace_id": "trace-abc123",
  "data": {
    "method": "discoverSignals",
    "params": {"sources": ["reddit", "twitter"]},
    "workflow_id": "validate-idea-12345"
  }
}
```

**Learning Point**: Logs provide **detailed context** for understanding specific events and debugging edge cases.

## Instrumentation Patterns

### Python Async Context Managers for Observability

**Automatic with Decorators**: Clean instrumentation without changing agent logic

```python
# Before: Regular agent method
async def discover_signals(self, sources: List[str]) -> List[Signal]:
    signals = await self._scan_sources(sources)
    return self._filter_signals(signals)

# After: Automatically wrapped with telemetry via registration
# (No code changes needed - instrumentation happens during agent registration)
```

**Context Manager Pattern**: Elegant resource management with automatic cleanup

```python
async def discover_signals(self, sources: List[str]) -> List[Signal]:
    async with self.telemetry.trace('signals-scout', 'discover_signals',
                                   {'sources': sources}) as trace:

        # Nested spans for detailed timing
        async with self.telemetry.span(trace.id, 'scan_sources') as span:
            signals = await self._scan_sources(sources)
            span.tags['signals_found'] = len(signals)

        async with self.telemetry.span(trace.id, 'filter_signals') as span:
            filtered = self._filter_signals(signals)
            span.tags['signals_filtered'] = len(filtered)

        return filtered
```

**Manual with Type Safety**: Explicit observability with proper typing

```python
async def validate_idea(self, idea: BusinessIdea) -> ValidationResult:
    await self._record_event('validation.started', {
        'idea_id': idea.id,
        'domain': idea.domain
    })

    market_data = await self._fetch_market_data(idea.domain)

    await self._record_event('market.data.fetched', {
        'domain': idea.domain,
        'data_points': len(market_data),
        'quality_score': self._assess_data_quality(market_data)
    })

    if len(market_data) < 10:
        await self._record_event('market.data.insufficient', {
            'required': 10,
            'actual': len(market_data)
        })
        raise InsufficientDataError("Need at least 10 market data points")

    return await self._analyze_market(market_data)
```
```

### Correlation IDs and Context

**The Problem**: In distributed systems, related events happen across different agents at different times.

**The Solution**: Correlation IDs that flow through the entire workflow.

```javascript
// Start workflow with correlation ID
const workflowId = 'validate-idea-12345'
const context = {
  workflow_id: workflowId,
  user_id: 'user789',
  experiment_budget: 500
}

// Every event includes this context
telemetry.recordEvent('workflow.started', context)
telemetry.recordEvent('signals.discovered', { ...context, signal_count: 23 })
telemetry.recordEvent('thesis.generated', { ...context, thesis_id: 'thesis-abc' })
```

**Result**: You can query for all events related to a specific workflow, user, or experiment.

## Debugging Multi-Agent Workflows

### Common Issues and Detection

**1. Deadlocks**
```javascript
// Symptom: Workflows stall indefinitely
const stalled = telemetry.queryTraces({
  status: 'active',
  older_than: '10m'
})

// Investigation: Look for circular dependencies
stalled.forEach(trace => {
  console.log(`Stalled trace: ${trace.id}`)
  console.log(`Waiting on: ${trace.waiting_for}`)
})
```

**2. Message Loss**
```javascript
// Symptom: Events sent but never received
const sent = telemetry.queryEvents({ type: 'message.sent' })
const received = telemetry.queryEvents({ type: 'message.received' })

sent.forEach(sentEvent => {
  const messageId = sentEvent.data.message_id
  const wasReceived = received.some(r => r.data.message_id === messageId)

  if (!wasReceived) {
    console.log(`Lost message: ${messageId}`)
  }
})
```

**3. Resource Contention**
```javascript
// Symptom: Performance degrades under load
const concurrentTasks = telemetry.queryMetrics('concurrent.tasks.active')
const responseTime = telemetry.queryMetrics('agent.response_time.avg')

// Look for correlation between load and performance
if (concurrentTasks.avg > 10 && responseTime.avg > 5000) {
  console.log('Performance degradation detected under high load')
}
```

**4. State Inconsistency**
```javascript
// Symptom: Agents have different views of shared state
telemetry.recordEvent('state.snapshot', {
  agent_id: 'agent-1',
  idea_count: localState.ideas.length,
  last_update: localState.lastModified
})

// Compare snapshots across agents to detect inconsistencies
```

### Debug Workflow Example

**Scenario**: Ideas are being validated but no theses are generated.

**Step 1**: Check the pipeline health
```javascript
const healthReport = telemetry.generateHealthReport()
console.log('Agent Health:', healthReport.agents.health)

// Output shows ThesisSynthesizer has 0 tasks completed
```

**Step 2**: Look for errors in ThesisSynthesizer
```javascript
const errors = telemetry.queryEvents({
  type: 'agent.error',
  agent_id: 'thesis-synthesizer',
  last: '1h'
})

// Found: "No signals provided for clustering"
```

**Step 3**: Trace the communication flow
```javascript
const traces = telemetry.queryTraces({
  operation: 'clusterSignals',
  status: 'failed'
})

traces.forEach(trace => {
  console.log(`Failed trace: ${trace.id}`)
  console.log(`Input signals: ${trace.context.signals?.length || 0}`)
})

// Discovered: SignalsScout is finding signals but not passing them to ThesisSynthesizer
```

**Step 4**: Fix the integration
```javascript
// Found the bug: Event name mismatch
// SignalsScout emits: 'signals:discovered'
// ThesisSynthesizer listens for: 'signals.discovered'

// Fixed by standardizing event names
```

## Performance Optimization

### Identifying Bottlenecks

**Response Time Analysis**
```javascript
const slowTraces = telemetry.queryTraces({
  duration_gt: 10000,  // > 10 seconds
  status: 'completed'
})

slowTraces.forEach(trace => {
  const bottleneck = trace.spans
    .sort((a, b) => b.duration - a.duration)[0]

  console.log(`Slowest span in ${trace.id}: ${bottleneck.name} (${bottleneck.duration}ms)`)
})
```

**Resource Utilization**
```javascript
const memoryTrend = telemetry.queryMetrics('system.memory.heap_used', '1h')
const cpuTrend = telemetry.queryMetrics('system.cpu.usage', '1h')

if (memoryTrend.trend === 'increasing') {
  console.log('Potential memory leak detected')
}
```

### Optimization Strategies

**1. Parallel Execution**
```javascript
// Before: Sequential execution
const signals = await signalsScout.discover()
const theses = await thesisSynthesizer.cluster(signals)
const validation = await validator.validate(theses)

// After: Pipeline parallelization
const [signals, marketData] = await Promise.all([
  signalsScout.discover(),
  marketAgent.fetchData()
])

const theses = await thesisSynthesizer.cluster(signals, marketData)
```

**2. Caching**
```javascript
telemetry.recordEvent('cache.check', { key: cacheKey })

const cached = await cache.get(cacheKey)
if (cached) {
  telemetry.recordEvent('cache.hit', { key: cacheKey })
  return cached
}

telemetry.recordEvent('cache.miss', { key: cacheKey })
const result = await expensiveOperation()
await cache.set(cacheKey, result)
```

**3. Load Balancing**
```javascript
const agentLoad = telemetry.queryMetrics('agent.active_tasks')
const leastLoaded = Object.entries(agentLoad)
  .sort(([,a], [,b]) => a - b)[0][0]

telemetry.recordEvent('task.delegated', {
  selected_agent: leastLoaded,
  reason: 'load_balancing'
})
```

## Alerting and Monitoring

### Alert Rules

```javascript
const alertRules = [
  {
    name: 'High Error Rate',
    condition: 'agent.error_rate > 0.05',
    severity: 'critical'
  },
  {
    name: 'Slow Response Time',
    condition: 'workflow.avg_duration > 30000',
    severity: 'warning'
  },
  {
    name: 'Agent Unresponsive',
    condition: 'agent.last_activity > 300000',
    severity: 'critical'
  }
]
```

### Health Checks

```javascript
class HealthCheck {
  async checkSystemHealth() {
    const checks = await Promise.all([
      this.checkAgentHealth(),
      this.checkWorkflowHealth(),
      this.checkResourceHealth()
    ])

    return {
      status: checks.every(c => c.healthy) ? 'healthy' : 'unhealthy',
      checks
    }
  }
}
```

---

## 🎯 Key Takeaways

1. **Instrument Early** → Add observability from day one, not after problems arise
2. **Correlate Everything** → Use trace IDs to connect related events across agents
3. **Monitor Business Metrics** → Technical metrics are important, but business outcomes matter most
4. **Automate Detection** → Don't wait for humans to notice problems
5. **Practice Debugging** → Regularly simulate failures to test your observability tools

**The Golden Rule**: If you can't observe it, you can't debug it. If you can't debug it, you can't optimize it.