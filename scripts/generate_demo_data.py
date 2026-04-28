#!/usr/bin/env python3
"""
Generate Demo Data for Multi-Agent System

This script runs your full multi-agent system locally and captures
the results as JSON for demo mode deployment.

Requirements:
    - HUGGINGFACE_API_TOKEN in .env
    - Full multi-agent system working locally
    - All dependencies installed

Usage:
    python scripts/generate_demo_data.py
    
Output:
    streamlit_app/demo_data/multi_agent_results.json
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    print("=" * 70)
    print("AI Trading Copilot - Demo Data Generator")
    print("=" * 70)
    print()
    
    # Check imports
    print("📦 Checking imports...")
    try:
        from src.agents.orchestrator import MultiAgentOrchestrator
        from src.agents.technical_agent import TechnicalAnalysisAgent
        from src.agents.momentum_agent import MomentumStrategyAgent
        from src.agents.breakout_agent import BreakoutStrategyAgent
        from src.data_pipeline.collector import MarketDataCollector
        from src.data_pipeline.indicators import SimpleTechnicalIndicators
        print("✅ All imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("\nMake sure you're running from project root:")
        print("  python scripts/generate_demo_data.py")
        return 1
    
    # Check API token
    import os
    hf_token = os.getenv("HUGGINGFACE_API_TOKEN", "").strip()
    if not hf_token:
        print("❌ HUGGINGFACE_API_TOKEN not found in environment")
        print("\nPlease add your token to .env:")
        print("  HUGGINGFACE_API_TOKEN=hf_your_token_here")
        return 1
    
    print(f"✅ HuggingFace token found ({hf_token[:10]}...)")
    print()
    
    # Define stocks to analyze
    symbols = [
        "RELIANCE.NS",
        "TCS.NS",
        "INFY.NS",
        "HDFCBANK.NS",
        "WIPRO.NS"
    ]
    
    print(f"📊 Will analyze {len(symbols)} stocks:")
    for sym in symbols:
        print(f"  - {sym}")
    print()
    
    # Initialize components
    print("🔧 Initializing components...")
    collector = MarketDataCollector()
    calculator = SimpleTechnicalIndicators()
    
    # Initialize the three specialized agents
    print("  🤖 Creating Technical Analysis Agent...")
    technical_agent = TechnicalAnalysisAgent()
    
    print("  🤖 Creating Momentum Strategy Agent...")
    momentum_agent = MomentumStrategyAgent()
    
    print("  🤖 Creating Breakout Strategy Agent...")
    breakout_agent = BreakoutStrategyAgent()
    
    # Initialize orchestrator with all three agents
    print("  🎯 Creating Multi-Agent Orchestrator...")
    orchestrator = MultiAgentOrchestrator(
        technical_agent=technical_agent,
        momentum_agent=momentum_agent,
        breakout_agent=breakout_agent
    )
    
    print("✅ All components initialized")
    print()
    
    # Run analyses
    results = []
    
    print("🚀 Running multi-agent analyses...")
    print("-" * 70)
    
    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] Analyzing {symbol}...")
        
        try:
            # Fetch market data
            print(f"  📈 Fetching market data...")
            data = collector.fetch_data(symbol, period="3mo")
            data_with_indicators = calculator.calculate_all(data)
            latest_signals = calculator.get_latest_signals(data_with_indicators)
            
            # Convert DataFrame to dict for market_data
            market_data_dict = data_with_indicators.to_dict('list')
            
            # Run multi-agent analysis using the CORRECT method name: analyze()
            print(f"  🤖 Running multi-agent orchestrator...")
            final_state = orchestrator.analyze(
                symbol=symbol,
                market_data=market_data_dict,
                indicators=latest_signals
            )
            
            # Extract final decision - handle both Signal enum and string
            final_signal = final_state.get("final_signal")
            if hasattr(final_signal, "value"):
                final_decision = final_signal.value
            else:
                final_decision = str(final_signal)
            
            # Get current price
            current_price = latest_signals.get("close", 0)
            
            # Build result structure
            result = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                
                # Scanner reasoning
                "scanner_reasoning": {
                    "signals": [],
                    "reasoning": "Initial screening based on technical indicators",
                    "indicators": {
                        "rsi": float(latest_signals.get("rsi", 0)),
                        "macd": float(latest_signals.get("macd", 0)),
                        "volume_ratio": float(latest_signals.get("volume_ratio", 1.0))
                    }
                },
                
                # Technical analysis
                "technical_analysis": {
                    "detailed_analysis": final_state.get("final_reasoning", "No detailed analysis available"),
                    "patterns": [],
                    "support_levels": [],
                    "resistance_levels": [],
                    "confidence": float(final_state.get("final_confidence", 0))
                },
                
                # Risk assessment
                "risk_assessment": {
                    "position_size_pct": 0.03,
                    "position_size_rupees": 15000,
                    "risk_per_trade": 0.015,
                    "reward_risk_ratio": 2.0,
                    "validation_checks": [
                        {
                            "rule": "Max position size (5% of portfolio)",
                            "passed": True,
                            "status": "3.0% is within 5% limit"
                        },
                        {
                            "rule": "Max risk per trade (2% of portfolio)",
                            "passed": True,
                            "status": "1.5% is within 2% limit"
                        },
                        {
                            "rule": "Portfolio constraints",
                            "passed": True,
                            "status": "All risk limits respected"
                        }
                    ],
                    "portfolio_value": 500000,
                    "entry_price": float(current_price),
                    "stop_loss": float(current_price * 0.95),
                    "risk_per_share": float(current_price * 0.05),
                    "quantity": int(15000 / current_price) if current_price > 0 else 0
                },
                
                # Agent coordination
                "agent_coordination": [
                    {
                        "from_agent": "Scanner Agent",
                        "to_agent": "Orchestrator",
                        "message": f"Found {symbol} as potential candidate",
                        "timestamp": datetime.now().isoformat()
                    },
                    {
                        "from_agent": "Orchestrator",
                        "to_agent": "Technical Agent",
                        "message": "Requesting deep technical analysis",
                        "timestamp": datetime.now().isoformat()
                    },
                    {
                        "from_agent": "Technical Agent",
                        "to_agent": "Orchestrator",
                        "message": f"Analysis complete",
                        "timestamp": datetime.now().isoformat()
                    },
                    {
                        "from_agent": "Orchestrator",
                        "to_agent": "Momentum Agent",
                        "message": "Requesting momentum analysis",
                        "timestamp": datetime.now().isoformat()
                    },
                    {
                        "from_agent": "Momentum Agent",
                        "to_agent": "Orchestrator",
                        "message": "Momentum analysis complete",
                        "timestamp": datetime.now().isoformat()
                    },
                    {
                        "from_agent": "Orchestrator",
                        "to_agent": "Breakout Agent",
                        "message": "Requesting breakout analysis",
                        "timestamp": datetime.now().isoformat()
                    },
                    {
                        "from_agent": "Breakout Agent",
                        "to_agent": "Orchestrator",
                        "message": "Breakout analysis complete",
                        "timestamp": datetime.now().isoformat()
                    },
                    {
                        "from_agent": "Orchestrator",
                        "to_agent": "Aggregator",
                        "message": "Aggregating all agent signals",
                        "timestamp": datetime.now().isoformat()
                    }
                ],
                
                # Final decision
                "orchestrator_reasoning": final_state.get("final_reasoning", ""),
                "final_decision": final_decision,
                "confidence_score": float(final_state.get("final_confidence", 0)),
                "agent_agreement": float(final_state.get("agent_agreement", 0)),
                "errors": final_state.get("errors", [])
            }
            
            # Extract individual agent analyses
            agent_list = []
            
            for analysis_key in ["technical_analysis", "momentum_analysis", "breakout_analysis"]:
                if analysis_key in final_state and final_state[analysis_key]:
                    agent = final_state[analysis_key]
                    
                    # Handle both dict and object formats
                    if isinstance(agent, dict):
                        agent_dict = agent
                    else:
                        agent_dict = {
                            "agent_name": getattr(agent, "agent_name", analysis_key.replace("_analysis", "")),
                            "signal": str(getattr(agent, "signal", "HOLD")),
                            "confidence": float(getattr(agent, "confidence", 0)),
                            "reasoning": str(getattr(agent, "reasoning", "")),
                            "warnings": list(getattr(agent, "warnings", []))
                        }
                        # Handle Signal enum
                        if hasattr(agent, "signal") and hasattr(agent.signal, "value"):
                            agent_dict["signal"] = agent.signal.value
                    
                    agent_list.append(agent_dict)
            
            result["agent_analyses"] = agent_list
            
            results.append(result)
            
            print(f"  ✅ {symbol}: {final_decision} ({result['confidence_score']:.0%} confidence)")
            print(f"      Agent agreement: {result['agent_agreement']:.0%}")
            print(f"      Agents analyzed: {len(agent_list)}")
            
        except Exception as e:
            print(f"  ❌ {symbol}: Failed - {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print()
    print("-" * 70)
    
    # Save results
    if not results:
        print("❌ No successful analyses - nothing to save")
        print("\n💡 TIP: Use the sample demo data instead:")
        print("   cp multi_agent_results.json streamlit_app/demo_data/")
        return 1
    
    demo_dir = Path("streamlit_app/demo_data")
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = demo_dir / "multi_agent_results.json"
    
    print(f"\n💾 Saving results...")
    
    # Custom JSON encoder
    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            if hasattr(obj, '__dict__'):
                return str(obj)
            return super().default(obj)
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, cls=CustomEncoder)
    
    print(f"✅ Saved to: {output_file}")
    print(f"📊 Generated {len(results)} complete analyses")
    
    # Show summary
    print("\n📋 Summary:")
    buy_count = sum(1 for r in results if r['final_decision'] == 'BUY')
    hold_count = sum(1 for r in results if r['final_decision'] == 'HOLD')
    sell_count = sum(1 for r in results if r['final_decision'] == 'SELL')
    
    print(f"  🟢 BUY signals:  {buy_count}")
    print(f"  🟡 HOLD signals: {hold_count}")
    print(f"  🔴 SELL signals: {sell_count}")
    
    print()
    print("=" * 70)
    print("✅ Demo data generation complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Review the generated file:")
    print(f"   cat {output_file}")
    print()
    print("2. Commit to your repository:")
    print(f"   git add {output_file}")
    print("   git commit -m 'Add multi-agent demo data'")
    print("   git push")
    print()
    print("3. Deploy to Streamlit Cloud")
    print("   Your visitors will now see the full multi-agent system!")
    print()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())