# Tutorial: Build Your First Multi-Agent Orchestrator

## What You'll Learn

By the end of this tutorial, you'll understand:
- How to coordinate multiple autonomous agents
- Event-driven communication patterns
- Task delegation with error handling
- Stage-gate validation for business processes
- Performance monitoring and debugging

## The Problem We're Solving

You have multiple agents (Signals Scout, Problem Validator, MVP Builder) that need to work together to validate business ideas. Without coordination, you get:

- **Race Conditions** → Agents step on each other
- **Duplicate Work** → Multiple agents doing the same task
- **No Quality Control** → Bad ideas make it through
- **Poor Visibility** → Can't debug or optimize

The orchestrator solves these problems with **centralized coordination** and **distributed execution**.

## Core Architecture

```
┌─────────────────────────┐
│      Orchestrator       │  ← Central coordinator
├─────────────────────────┤
│ • Workflow Management   │
│ • Task Delegation       │
│ • Stage Gate Validation │
│ • Error Recovery        │
└─────────┬───────────────┘
          │
    ┌─────┴─────┐
    │ Event Bus │  ← Communication backbone
    └─────┬─────┘
          │
  ┌───────┼───────┐
  ▼       ▼       ▼
Agent1  Agent2  Agent3  ← Specialized workers
```

## Step 1: Understanding the Orchestrator Class

```javascript
class VentureStudioOrchestrator extends EventEmitter {
  constructor(config = {}) {
    super()  // Inherit event capabilities

    // Core state - what the orchestrator tracks
    this.agents = new Map()      // Who can do work?
    this.workflows = new Map()   // What work is in progress?
    this.stageGates = new Map()  // What are the quality rules?
    this.metrics = {}            // How are we performing?
  }
}
```

**Learning Point**: The orchestrator inherits from `EventEmitter` because **event-driven coordination** is more scalable than direct method calls.

## Step 2: Agent Registration Pattern

```javascript
registerAgent(agentId, agent) {
  // Validation
  if (this.agents.has(agentId)) {
    throw new Error(`Agent ${agentId} already registered`)
  }

  // Telemetry wrapper - adds monitoring without changing agent code
  const wrappedAgent = this.wrapAgentWithTelemetry(agentId, agent)
  this.agents.set(agentId, wrappedAgent)

  // Event notification
  this.emit('agent:registered', { agentId, capabilities: agent.capabilities })
}
```

**Learning Point**: The **decorator pattern** (`wrapAgentWithTelemetry`) adds cross-cutting concerns (logging, metrics, error handling) without modifying agent implementations.

## Step 3: Workflow State Management

```javascript
const workflow = {
  id: workflowId,           // Unique identifier
  type: workflowType,       // "idea-validation", "mvp-build", etc.
  status: 'initiated',      // Lifecycle state
  currentStage: 'signals',  // Where in the process
  data: initialData,        // Business data being processed
  evidence: [],             // Accumulated validation evidence
  decisions: [],            // Decision audit trail
  metrics: {}              // Performance tracking
}
```

**Learning Point**: Rich workflow state enables **debuggability** and **auditability**. You can reconstruct exactly what happened and why.

## Step 4: Task Delegation with Resilience

```javascript
async delegateTask(agentId, method, params, options = {}) {
  const agent = this.agents.get(agentId)
  const timeout = options.timeout || this.config.defaultTimeout
  const retries = options.retries || this.config.retryAttempts

  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      // Race between task completion and timeout
      const taskPromise = agent[method](params)
      const timeoutPromise = new Promise((_, reject) =>
        setTimeout(() => reject(new Error('Task timeout')), timeout)
      )

      return await Promise.race([taskPromise, timeoutPromise])

    } catch (error) {
      if (attempt < retries) {
        await this.sleep(Math.pow(2, attempt) * 1000)  // Exponential backoff
      }
    }
  }

  throw new Error(`Task failed after ${retries} attempts`)
}
```

**Learning Points**:
- **Promise.race()** implements timeout behavior
- **Exponential backoff** prevents overwhelming failed services
- **Circuit breaker pattern** could be added here for system protection

## Step 5: Stage Gate Validation

```javascript
async validateStageGate(workflowId, stageName) {
  const workflow = this.workflows.get(workflowId)
  const gate = this.stageGates.get(stageName)

  // Check required evidence
  for (const evidenceType of gate.requiredEvidence) {
    const hasEvidence = workflow.evidence.some(e => e.type === evidenceType)
    if (!hasEvidence) return false
  }

  // Check metric thresholds
  for (const [metric, threshold] of Object.entries(gate.thresholds)) {
    const actualValue = this.getWorkflowMetric(workflow, metric)
    if (actualValue < threshold) return false
  }

  return true
}
```

**Learning Point**: **Evidence-based decision making** prevents gut-feel decisions. Every workflow advancement requires objective evidence.

## Step 6: Event-Driven Communication

```javascript
// Broadcasting events
this.emit('workflow:started', { workflowId, workflow })
this.emit('stage:completed', { workflowId, stageName })
this.emit('agent:error', { agentId, error })

// Listening for events
this.on('agent:error', ({ agentId, error }) => {
  console.error(`Agent ${agentId} error:`, error.message)
  // Could restart agent, escalate to human, etc.
})
```

**Learning Point**: Events create **loose coupling**. Agents don't need to know about each other directly—they just emit/listen for relevant events.

## Running Your First Orchestrator

```javascript
// Create orchestrator
const orchestrator = new VentureStudioOrchestrator({
  maxConcurrentWorkflows: 5,
  defaultTimeout: 30000
})

// Define stage gates
orchestrator.defineStageGate('signals', {
  requiredEvidence: ['signals_discovered'],
  thresholds: { signal_count: 10 }
})

// Register agents (we'll build these next)
orchestrator.registerAgent('signals-scout', new SignalsScoutAgent())
orchestrator.registerAgent('thesis-synthesizer', new ThesisSynthesizerAgent())

// Start a workflow
const workflowId = await orchestrator.startWorkflow('idea-validation', {
  sources: ['reddit', 'twitter'],
  target_market: 'SaaS'
})

console.log(`Started workflow: ${workflowId}`)
```

## Debugging and Monitoring

The orchestrator emits detailed events for observability:

```javascript
// Performance monitoring
orchestrator.on('agent:task:completed', ({ agentId, method, duration }) => {
  console.log(`${agentId}.${method}() took ${duration}ms`)
})

// Error tracking
orchestrator.on('agent:error', ({ agentId, error }) => {
  console.error(`Agent ${agentId} failed: ${error.message}`)
})

// Workflow progress
orchestrator.on('stage:completed', ({ workflowId, stageName }) => {
  console.log(`Workflow ${workflowId} completed stage: ${stageName}`)
})
```

## Common Pitfalls and Solutions

### 1. **Tight Coupling**
❌ **Bad**: `agent1.processData(agent2.getData())`
✅ **Good**: Event-driven communication via orchestrator

### 2. **Blocking Operations**
❌ **Bad**: Waiting for slow agents blocks entire workflow
✅ **Good**: Timeouts and circuit breakers

### 3. **Poor Error Handling**
❌ **Bad**: One agent failure kills entire workflow
✅ **Good**: Isolated failures with retry logic

### 4. **No Observability**
❌ **Bad**: Black box—can't debug when things go wrong
✅ **Good**: Rich event streams and metrics

## Next Steps

1. **Build Your First Agent** → Create a simple Signals Scout
2. **Add Communication** → Connect agents via events
3. **Implement Stage Gates** → Add business validation rules
4. **Scale and Optimize** → Add more agents and monitor performance

The orchestrator is your foundation. Everything else builds on these patterns!