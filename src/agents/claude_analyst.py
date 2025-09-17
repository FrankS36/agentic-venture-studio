"""
Claude AI Analyst Agent

This agent uses Claude AI to provide intelligent analysis of signals and business opportunities:
- Deep analysis of signal content for business potential
- Market opportunity assessment with reasoning
- Competitive landscape analysis
- Investment thesis generation
- Risk factor identification

Features:
- Uses your Claude API key from environment variables
- Async operations for performance
- Intelligent prompt engineering for business analysis
- Type-safe responses with confidence scoring
- Integration with existing signal database
"""

import asyncio
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import anthropic
except ImportError:
    anthropic = None

# Handle imports for different execution contexts
try:
    from ..persistence.repositories import get_signal_repository
    from ..persistence.models import Signal
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from persistence.repositories import get_signal_repository
    from persistence.models import Signal

logger = logging.getLogger(__name__)


@dataclass
class ClaudeAnalysis:
    """Claude AI analysis result"""
    signal_id: str
    business_potential: float  # 0-1 score
    market_size_estimate: str
    target_audience: str
    competitive_landscape: str
    investment_thesis: str
    risk_factors: List[str]
    opportunity_type: str  # problem-solution, market-gap, trend-acceleration
    confidence: float  # 0-1 confidence in analysis
    reasoning: str
    timestamp: datetime


class ClaudeAnalystAgent:
    """
    Claude AI-powered business analyst agent

    This agent provides deep, intelligent analysis of business signals using
    Claude's advanced reasoning capabilities for market assessment.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.api_key = os.getenv('CLAUDE_API_KEY')
        self.model = os.getenv('CLAUDE_MODEL', 'claude-3-5-sonnet-20241022')
        self.max_tokens = int(os.getenv('CLAUDE_MAX_TOKENS', '4096'))

        # Initialize Claude client
        self.client = None
        if anthropic and self.api_key:
            try:
                self.client = anthropic.Anthropic(api_key=self.api_key)
                logger.info("✅ Claude AI client initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Claude client: {e}")
        else:
            if not anthropic:
                logger.warning("⚠️  Anthropic library not installed. Install with: pip install anthropic")
            if not self.api_key:
                logger.warning("⚠️  CLAUDE_API_KEY not found in environment variables")

    async def analyze_signal(self, signal: Signal) -> Optional[ClaudeAnalysis]:
        """
        Analyze a single signal for business potential using Claude AI

        Args:
            signal: Signal object to analyze

        Returns:
            ClaudeAnalysis object with detailed business assessment
        """

        if not self.client:
            logger.warning(f"Claude client not available for signal {signal.id}")
            return None

        try:
            # Prepare the analysis prompt
            prompt = self._create_analysis_prompt(signal)

            # Call Claude API
            response = await self._call_claude_api(prompt)

            if response:
                # Parse Claude's response into structured analysis
                analysis = self._parse_claude_response(signal.id, response)
                return analysis

        except Exception as e:
            logger.error(f"Error analyzing signal {signal.id} with Claude: {e}")

        return None

    async def analyze_signals_batch(self, signals: List[Signal], max_concurrent: int = 3) -> List[ClaudeAnalysis]:
        """
        Analyze multiple signals concurrently with rate limiting

        Args:
            signals: List of signals to analyze
            max_concurrent: Maximum concurrent API calls

        Returns:
            List of ClaudeAnalysis objects
        """

        if not self.client:
            logger.warning("Claude client not available for batch analysis")
            return []

        # Process in batches to respect rate limits
        semaphore = asyncio.Semaphore(max_concurrent)

        async def analyze_with_limit(signal):
            async with semaphore:
                return await self.analyze_signal(signal)

        # Execute batch analysis
        tasks = [analyze_with_limit(signal) for signal in signals]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter successful analyses
        analyses = []
        for result in results:
            if isinstance(result, ClaudeAnalysis):
                analyses.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Batch analysis error: {result}")

        logger.info(f"✅ Completed Claude analysis for {len(analyses)}/{len(signals)} signals")
        return analyses

    async def generate_market_thesis(self, signals: List[Signal]) -> Optional[str]:
        """
        Generate a comprehensive market thesis from multiple signals

        Args:
            signals: List of related signals

        Returns:
            Comprehensive market thesis string
        """

        if not self.client or not signals:
            return None

        try:
            # Create market thesis prompt
            prompt = self._create_thesis_prompt(signals)

            # Call Claude API
            response = await self._call_claude_api(prompt)
            return response

        except Exception as e:
            logger.error(f"Error generating market thesis: {e}")
            return None

    def _create_analysis_prompt(self, signal: Signal) -> str:
        """Create a prompt for analyzing a single signal"""

        return f"""
You are a seasoned business analyst and venture capitalist evaluating market opportunities.

Analyze this business signal for its commercial potential:

**Signal Source:** {signal.source}
**Title:** {signal.title}
**Content:** {signal.content}
**Original Score:** {signal.final_score}
**Keywords Found:** {', '.join(signal.keywords or [])}
**Opportunity Signals:** {', '.join(signal.signals_found or [])}

Provide a comprehensive business analysis in the following JSON format:

{{
    "business_potential": 0.8,
    "market_size_estimate": "Large ($1B+ TAM) / Medium ($100M-1B TAM) / Small (<$100M TAM)",
    "target_audience": "Specific description of target customers",
    "competitive_landscape": "Analysis of existing solutions and competitors",
    "investment_thesis": "3-sentence investment rationale",
    "risk_factors": ["Risk 1", "Risk 2", "Risk 3"],
    "opportunity_type": "problem-solution / market-gap / trend-acceleration",
    "confidence": 0.7,
    "reasoning": "Detailed explanation of your analysis and scoring rationale"
}}

Focus on:
1. Real market demand and size
2. Competitive differentiation potential
3. Implementation feasibility
4. Revenue model viability
5. Scalability factors

Be specific, data-driven, and honest about limitations.
"""

    def _create_thesis_prompt(self, signals: List[Signal]) -> str:
        """Create a prompt for generating market thesis from multiple signals"""

        signals_summary = []
        for i, signal in enumerate(signals[:10], 1):  # Limit to top 10 for context
            signals_summary.append(f"""
Signal {i}:
- Source: {signal.source}
- Title: {signal.title}
- Score: {signal.final_score:.2f}
- Content: {signal.content[:200]}...
""")

        return f"""
You are a venture capital partner writing an investment thesis based on market signals.

Analyze these {len(signals)} related business signals and generate a comprehensive market thesis:

{chr(10).join(signals_summary)}

Create a detailed market thesis covering:

1. **Market Opportunity**: What specific problem/need is emerging?
2. **Market Size**: Estimated TAM, SAM, SOM with reasoning
3. **Timing**: Why now? What catalysts make this opportunity timely?
4. **Target Segments**: Primary and secondary customer segments
5. **Competitive Landscape**: Current solutions and gaps
6. **Business Model**: Potential revenue streams and unit economics
7. **Key Success Factors**: What would make a startup successful here?
8. **Risk Assessment**: Main risks and mitigation strategies
9. **Investment Rationale**: Why this is an attractive investment opportunity

Format as a well-structured markdown document suitable for presenting to investors.
Be specific, cite signal evidence, and provide actionable insights.
"""

    async def _call_claude_api(self, prompt: str) -> Optional[str]:
        """Make an API call to Claude"""

        try:
            # Use asyncio to run the synchronous anthropic client
            loop = asyncio.get_event_loop()

            response = await loop.run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )
            )

            if response.content and len(response.content) > 0:
                return response.content[0].text

        except Exception as e:
            logger.error(f"Claude API call failed: {e}")

        return None

    def _parse_claude_response(self, signal_id: str, response: str) -> Optional[ClaudeAnalysis]:
        """Parse Claude's JSON response into ClaudeAnalysis object"""

        try:
            # Try to extract JSON from Claude's response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)

                return ClaudeAnalysis(
                    signal_id=signal_id,
                    business_potential=float(data.get('business_potential', 0.5)),
                    market_size_estimate=data.get('market_size_estimate', 'Unknown'),
                    target_audience=data.get('target_audience', 'Not specified'),
                    competitive_landscape=data.get('competitive_landscape', 'Unknown'),
                    investment_thesis=data.get('investment_thesis', 'No thesis generated'),
                    risk_factors=data.get('risk_factors', []),
                    opportunity_type=data.get('opportunity_type', 'unknown'),
                    confidence=float(data.get('confidence', 0.5)),
                    reasoning=data.get('reasoning', 'No reasoning provided'),
                    timestamp=datetime.utcnow()
                )

        except Exception as e:
            logger.error(f"Failed to parse Claude response: {e}")

        return None

    async def get_signal_recommendations(self, limit: int = 10) -> List[Tuple[Signal, ClaudeAnalysis]]:
        """
        Get top signals from database and analyze them with Claude

        Returns:
            List of (Signal, ClaudeAnalysis) tuples sorted by business potential
        """

        try:
            # Get top signals from database
            signal_repo = get_signal_repository()
            signals = await signal_repo.get_top_signals(limit=limit, min_score=0.3)

            if not signals:
                logger.info("No signals found for Claude analysis")
                return []

            # Analyze with Claude
            analyses = await self.analyze_signals_batch(signals)

            # Pair signals with analyses
            recommendations = []
            analysis_dict = {a.signal_id: a for a in analyses}

            for signal in signals:
                if signal.id in analysis_dict:
                    recommendations.append((signal, analysis_dict[signal.id]))

            # Sort by business potential
            recommendations.sort(key=lambda x: x[1].business_potential, reverse=True)

            logger.info(f"✅ Generated {len(recommendations)} Claude-powered recommendations")
            return recommendations

        except Exception as e:
            logger.error(f"Error getting signal recommendations: {e}")
            return []


# Example usage and testing
async def main():
    """Demo the Claude analyst agent"""

    print("🤖 Claude AI Analyst Agent Demo")
    print("=" * 40)

    # Check if API key is available
    api_key = os.getenv('CLAUDE_API_KEY')
    if not api_key or api_key == 'your_claude_api_key_here':
        print("❌ Claude API key not configured")
        print("📝 Please set your CLAUDE_API_KEY in the .env file")
        print("🔗 Get your key from: https://console.anthropic.com/")
        return

    analyst = ClaudeAnalystAgent()

    if not analyst.client:
        print("❌ Claude client not available")
        return

    try:
        # Get recommendations
        print("🔍 Getting Claude-powered signal recommendations...")
        recommendations = await analyst.get_signal_recommendations(limit=5)

        if recommendations:
            print(f"\n📊 Claude Analysis Results ({len(recommendations)} signals):")
            print("-" * 60)

            for i, (signal, analysis) in enumerate(recommendations, 1):
                print(f"\n{i}. {signal.title}")
                print(f"   Business Potential: {analysis.business_potential:.2f}")
                print(f"   Market Size: {analysis.market_size_estimate}")
                print(f"   Opportunity Type: {analysis.opportunity_type}")
                print(f"   Confidence: {analysis.confidence:.2f}")
                print(f"   Investment Thesis: {analysis.investment_thesis}")

                if analysis.risk_factors:
                    print(f"   Key Risks: {', '.join(analysis.risk_factors[:2])}")

        else:
            print("ℹ️  No signals available for analysis")
            print("💡 Run the Reddit discovery first to populate signals")

    except Exception as e:
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())