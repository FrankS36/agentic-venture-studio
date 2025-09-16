#!/usr/bin/env python3
"""
Complete Multi-Agent System Demo: Python Implementation

This demo showcases a fully functional multi-agent system with:
- Async orchestration and coordination
- Intelligent agents with proper typing
- Comprehensive observability and monitoring
- Real-time performance tracking

Run this to see Python multi-agent coordination in action!
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from coordination.orchestrator import VentureStudioOrchestrator, StageGate
from agents.signals_scout import SignalsScoutAgent
from agents.thesis_synthesizer import ThesisSynthesizerAgent
from observability.telemetry import AgentTelemetrySystem, TelemetryConfig


async def setup_demo_system() -> Dict[str, Any]:
    """
    Create and configure the complete multi-agent system

    Learning Note: Python's async/await makes system setup much cleaner
    than callback-based approaches. Type hints catch integration errors early.
    """
    print("🚀 Setting up Multi-Agent Venture Studio Demo (Python)\n")

    # 1. Initialize telemetry with custom config
    telemetry_config = TelemetryConfig()
    telemetry_config.enable_tracing = True
    telemetry_config.enable_metrics = True
    telemetry_config.log_level = logging.INFO

    telemetry = AgentTelemetrySystem(telemetry_config)

    # 2. Create orchestrator
    orchestrator = VentureStudioOrchestrator({
        'max_concurrent_workflows': 5,
        'default_timeout': 30.0,
        'stage_timeout': 120.0
    })

    # 3. Create agents with custom configurations
    signals_scout = SignalsScoutAgent({
        'sources': ['reddit', 'twitter', 'news'],
        'min_signal_score': 0.4,
        'parallel_sources': True,
        'cache_ttl': 300
    })

    thesis_synthesizer = ThesisSynthesizerAgent({
        'min_cluster_size': 2,
        'similarity_threshold': 0.5,
        'validation_threshold': 0.6,
        'max_theses_per_cluster': 3
    })

    # 4. Register agents with orchestrator
    await orchestrator.register_agent('signals-scout', signals_scout, signals_scout.capabilities)
    await orchestrator.register_agent('thesis-synthesizer', thesis_synthesizer, thesis_synthesizer.capabilities)

    # 5. Register agents with telemetry
    await telemetry.register_agent('signals-scout', signals_scout, signals_scout.capabilities)
    await telemetry.register_agent('thesis-synthesizer', thesis_synthesizer, thesis_synthesizer.capabilities)

    # 6. Define stage gates with evidence requirements
    signals_gate = StageGate(
        required_evidence=['signals_discovered'],
        thresholds={'signal_count': 3},  # Need at least 3 signals
        validators=[]  # No validator agents for demo
    )
    orchestrator.define_stage_gate('signals', signals_gate)

    # 7. Setup cross-agent communication
    await setup_agent_communication(orchestrator, signals_scout, thesis_synthesizer, telemetry)

    print("✅ Multi-agent system initialized successfully\n")

    return {
        'orchestrator': orchestrator,
        'signals_scout': signals_scout,
        'thesis_synthesizer': thesis_synthesizer,
        'telemetry': telemetry
    }


async def setup_agent_communication(orchestrator, signals_scout, thesis_synthesizer, telemetry):
    """
    Setup inter-agent communication patterns

    Learning Note: Python's event system with async handlers creates
    clean, non-blocking communication between agents.
    """

    async def handle_workflow_started(data):
        print(f"🏁 Workflow started: {data['workflow_id']}")

    async def handle_stage_completed(data):
        print(f"✅ Stage completed: {data['stage_name']} for workflow: {data['workflow_id']}")

    async def handle_workflow_completed(data):
        print(f"🎉 Workflow completed: {data['workflow_id']} in {data['duration']:.2f}s")

    async def handle_agent_error(data):
        print(f"🚨 Agent error: {data}")

    # Register orchestrator event handlers
    orchestrator.on('workflow_started', handle_workflow_started)
    orchestrator.on('stage_completed', handle_stage_completed)
    orchestrator.on('workflow_completed', handle_workflow_completed)

    # Register telemetry event handlers
    telemetry.on('method.error', handle_agent_error)

    print("🔗 Agent communication setup complete")


async def demonstrate_basic_workflow():
    """Demonstrate a complete workflow execution"""
    print("🎭 Running Basic Workflow Demo\n")

    system = await setup_demo_system()
    orchestrator = system['orchestrator']
    telemetry = system['telemetry']

    try:
        # Start a venture validation workflow
        workflow_id = await orchestrator.start_workflow('idea-validation', {
            'domain': 'AI/ML',
            'target_market': 'Enterprise',
            'budget': 5000
        })

        print(f"📊 Workflow started: {workflow_id}")

        # Wait for workflow to progress
        await asyncio.sleep(5.0)

        # Show system status
        status = orchestrator.get_system_status()
        print(f"\n📈 System Status:")
        print(f"   Active Workflows: {status['workflows']['active']}")
        print(f"   Completed Workflows: {status['workflows']['completed']}")
        print(f"   Active Agents: {status['agents']['active']}")

        return True

    except Exception as error:
        print(f"❌ Basic workflow failed: {str(error)}")
        return False

    finally:
        await telemetry.cleanup()


async def demonstrate_direct_agent_interaction():
    """Demonstrate direct agent-to-agent communication"""
    print("🔗 Testing Direct Agent Communication\n")

    system = await setup_demo_system()
    signals_scout = system['signals_scout']
    thesis_synthesizer = system['thesis_synthesizer']
    telemetry = system['telemetry']

    try:
        # Direct signal discovery
        print("🔍 Discovering signals directly...")
        signals = await signals_scout.discover_signals(
            sources=['reddit', 'twitter'],
            timeframe='24h',
            min_quality=0.3
        )

        print(f"📡 Discovered {len(signals)} signals")

        if signals:
            # Show signal details
            for i, signal in enumerate(signals[:3], 1):
                print(f"\n📋 Signal {i}:")
                print(f"   Title: {signal.title}")
                print(f"   Source: {signal.source}")
                print(f"   Score: {signal.score:.2f}")
                print(f"   Category: {signal.category}")
                print(f"   Tags: {', '.join(signal.tags)}")

            # Direct thesis synthesis
            print("\n🧠 Synthesizing theses...")
            theses = await thesis_synthesizer.cluster_signals(signals=signals)

            print(f"💡 Generated {len(theses)} business theses")

            # Show thesis details
            for i, thesis in enumerate(theses, 1):
                print(f"\n📊 Thesis {i}:")
                print(f"   Title: {thesis.title}")
                print(f"   Type: {thesis.type.value}")
                print(f"   Target Market: {thesis.target_market}")
                print(f"   Validation Score: {thesis.validation_score:.2f}")
                print(f"   Confidence: {thesis.confidence:.2f}")
                print(f"   Priority Rank: {thesis.priority_rank}")

        return len(signals), len(theses) if signals else 0

    except Exception as error:
        print(f"❌ Direct interaction failed: {str(error)}")
        return 0, 0

    finally:
        await telemetry.cleanup()


async def demonstrate_observability():
    """Demonstrate comprehensive observability features"""
    print("📊 Demonstrating Observability Features\n")

    system = await setup_demo_system()
    telemetry = system['telemetry']
    signals_scout = system['signals_scout']

    try:
        # Generate some activity for observability
        print("🔧 Generating system activity...")

        # Multiple concurrent operations
        tasks = [
            signals_scout.discover_signals(sources=['reddit'], timeframe='12h'),
            signals_scout.discover_signals(sources=['twitter'], timeframe='6h'),
            signals_scout.discover_signals(sources=['news'], timeframe='24h')
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        successful_results = [r for r in results if not isinstance(r, Exception)]

        print(f"✅ Completed {len(successful_results)} operations")

        # Wait for metrics to be collected
        await asyncio.sleep(2.0)

        # Generate comprehensive health report
        print("\n📋 System Health Report:")
        health_report = await telemetry.generate_health_report()

        print(f"   Uptime: {health_report['uptime_seconds']:.1f}s")
        print(f"   Traces Created: {health_report['system']['traces_created']}")
        print(f"   Events Recorded: {health_report['system']['events_recorded']}")
        print(f"   Active Agents: {health_report['agents']['active']}/{health_report['agents']['total']}")

        # Show agent performance
        print(f"\n🤖 Agent Performance:")
        for agent_health in health_report['agents']['health']:
            print(f"   {agent_health['agent_id']}:")
            print(f"      Tasks Completed: {agent_health['tasks_completed']}")
            print(f"      Error Rate: {agent_health['error_rate']:.1%}")
            print(f"      Avg Duration: {agent_health['avg_duration']:.3f}s")

        # Query specific metrics
        print(f"\n📈 Metrics Analysis:")
        trace_metrics = await telemetry.query_metrics('trace.completed', 300.0)
        print(f"   Traces Completed: {trace_metrics['count']}")
        if trace_metrics['count'] > 0:
            print(f"   Avg Duration: {trace_metrics['avg']:.3f}s")
            print(f"   Min Duration: {trace_metrics['min']:.3f}s")
            print(f"   Max Duration: {trace_metrics['max']:.3f}s")

        # Export traces for analysis
        traces = telemetry.export_traces({'status': 'completed'})
        print(f"   Exportable Traces: {len(traces)}")

        return health_report

    except Exception as error:
        print(f"❌ Observability demo failed: {str(error)}")
        return None

    finally:
        await telemetry.cleanup()


async def interactive_demo():
    """Interactive demo mode for learning exploration"""
    print("🎓 Interactive Learning Mode\n")

    system = await setup_demo_system()
    signals_scout = system['signals_scout']
    thesis_synthesizer = system['thesis_synthesizer']
    orchestrator = system['orchestrator']
    telemetry = system['telemetry']

    print("Available commands:")
    print("1. discover - Run signal discovery")
    print("2. synthesize - Generate theses from cached signals")
    print("3. workflow - Start full orchestrated workflow")
    print("4. health - Show detailed system health")
    print("5. agents - Show agent status")
    print("6. metrics - Show performance metrics")
    print("7. traces - Show trace analysis")

    try:
        # Simulate running commands (in a real implementation, you'd use input())
        commands = ['discover', 'synthesize', 'health', 'metrics']

        for cmd in commands:
            print(f"\n> {cmd}")

            if cmd == 'discover':
                signals = await signals_scout.discover_signals(
                    sources=['reddit', 'twitter', 'news'],
                    timeframe='24h'
                )
                print(f"Discovered {len(signals)} signals")

                # Cache signals for synthesis
                cached_signals = signals

            elif cmd == 'synthesize':
                if 'cached_signals' in locals():
                    theses = await thesis_synthesizer.cluster_signals(signals=cached_signals)
                    print(f"Synthesized {len(theses)} theses")
                else:
                    print("No signals available. Run 'discover' first.")

            elif cmd == 'health':
                health = await telemetry.generate_health_report()
                print(f"System Health: {health['agents']['active']} active agents, "
                      f"{health['traces']['active']} active traces")

            elif cmd == 'metrics':
                # Show key metrics
                trace_metrics = await telemetry.query_metrics('trace.completed')
                agent_metrics = await telemetry.query_metrics('agent.tasks_completed')
                print(f"Traces: {trace_metrics['count']}, Agent Tasks: {agent_metrics['sum']}")

            # Pause between commands
            await asyncio.sleep(1.0)

        print("\n🎓 Interactive demo completed!")

    except Exception as error:
        print(f"❌ Interactive demo failed: {str(error)}")

    finally:
        await telemetry.cleanup()


async def performance_benchmark():
    """Benchmark system performance under load"""
    print("⚡ Performance Benchmark\n")

    system = await setup_demo_system()
    signals_scout = system['signals_scout']
    telemetry = system['telemetry']

    try:
        print("🔥 Running concurrent load test...")

        # Create many concurrent tasks
        start_time = time.time()
        tasks = []

        for i in range(10):  # 10 concurrent discovery operations
            task = signals_scout.discover_signals(
                sources=['reddit', 'twitter'],
                timeframe='12h',
                min_quality=0.2
            )
            tasks.append(task)

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()
        duration = end_time - start_time

        # Analyze results
        successful = [r for r in results if not isinstance(r, Exception)]
        failed = [r for r in results if isinstance(r, Exception)]

        print(f"⚡ Performance Results:")
        print(f"   Total Duration: {duration:.2f}s")
        print(f"   Successful Operations: {len(successful)}")
        print(f"   Failed Operations: {len(failed)}")
        print(f"   Operations/Second: {len(successful) / duration:.2f}")

        if successful:
            total_signals = sum(len(signals) for signals in successful)
            print(f"   Total Signals Processed: {total_signals}")
            print(f"   Signals/Second: {total_signals / duration:.2f}")

        # Show system metrics after load test
        health = await telemetry.generate_health_report()
        print(f"   System Events Generated: {health['system']['events_recorded']}")
        print(f"   Traces Created: {health['system']['traces_created']}")

        return {
            'duration': duration,
            'successful': len(successful),
            'failed': len(failed),
            'throughput': len(successful) / duration
        }

    except Exception as error:
        print(f"❌ Performance benchmark failed: {str(error)}")
        return None

    finally:
        await telemetry.cleanup()


async def main():
    """Main demo entry point"""
    print("🚀 Multi-Agent Venture Studio Demo (Python)\n")

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        print("🎯 Demo Scenarios:\n")

        # 1. Basic workflow demonstration
        print("=" * 60)
        success = await demonstrate_basic_workflow()
        if success:
            print("✅ Basic workflow demo completed successfully")
        else:
            print("❌ Basic workflow demo failed")

        await asyncio.sleep(2.0)

        # 2. Direct agent interaction
        print("\n" + "=" * 60)
        signals_count, theses_count = await demonstrate_direct_agent_interaction()
        print(f"✅ Direct interaction demo: {signals_count} signals → {theses_count} theses")

        await asyncio.sleep(2.0)

        # 3. Observability demonstration
        print("\n" + "=" * 60)
        health_report = await demonstrate_observability()
        if health_report:
            print("✅ Observability demo completed successfully")

        await asyncio.sleep(2.0)

        # 4. Interactive learning
        print("\n" + "=" * 60)
        await interactive_demo()

        await asyncio.sleep(2.0)

        # 5. Performance benchmark
        print("\n" + "=" * 60)
        perf_results = await performance_benchmark()
        if perf_results:
            print(f"✅ Performance benchmark: {perf_results['throughput']:.2f} ops/sec")

        print("\n" + "=" * 60)
        print("🎉 All demos completed successfully!")

        print("\n🎯 Next Steps:")
        print("1. Explore the source code to understand the patterns")
        print("2. Modify agent behaviors and configurations")
        print("3. Add new agents with custom capabilities")
        print("4. Implement real external APIs and data sources")
        print("5. Scale the system with distributed coordination")
        print("6. Add ML-powered clustering and analysis")

    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as error:
        print(f"\n❌ Demo failed with error: {str(error)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the async demo
    asyncio.run(main())