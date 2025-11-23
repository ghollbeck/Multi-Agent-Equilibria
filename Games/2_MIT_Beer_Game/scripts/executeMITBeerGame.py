#!/usr/bin/env python3
"""
executeMRIPtbrGame.py

This script runs the MIT Beer Game simulation with configurable parameters.
"""
import argparse
import asyncio
import os
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass  # print("Warning: nest_asyncio not available, running without it")  # Commented out

# Import modules
import llm_calls_mitb_game
from MIT_Beer_Game import run_beer_game_simulation
from models_mitb_game import SimulationData


def parse_args():
    parser = argparse.ArgumentParser(
        description="Execute the MIT Beer Game simulation with custom parameters"
    )
    parser.add_argument(
        "--num_rounds", type=int, default=10,
        help="Number of rounds to simulate (canonical MIT Beer Game uses 36-50 rounds)"
    )
    parser.add_argument(
        "--holding_cost_per_unit", type=float, default=0.0,
        help="Holding cost per unit per round"
    )
    parser.add_argument(
        "--backlog_cost_per_unit", type=float, default=2.5,
        help="Backlog cost per unit per round"
    )
    parser.add_argument(
        "--temperature", type=float, default=0,
        help="Sampling temperature for the LLM"
    )
    parser.add_argument(
        "--model_name", type=str, default=llm_calls_mitb_game.MODEL_NAME,
        help="Name of the LLM model to use (for LiteLLM provider)"
    )
    parser.add_argument(
        "--provider", type=str, choices=["litellm", "anthropic"], default="litellm",
        help="Which LLM provider to use: 'litellm' (default) or 'anthropic'"
    )
    parser.add_argument(
        "--anthropic_model", type=str, default="claude-3-5-sonnet-latest",
        help="Claude model name to use when --provider anthropic (default: claude-3-5-sonnet-latest)"
    )
    parser.add_argument(
        "--enable_communication", action="store_true", default=False,
        help="Enable agent communication before each round"
    )
    parser.add_argument(
        "--communication_rounds", type=int, default=2,
        help="Number of communication rounds per game round"
    )
    parser.add_argument(
        "--enable_memory", action="store_true",
        help="Enable agent memory storage for strategies and reasoning"
    )
    parser.add_argument(
        "--memory_retention_rounds", type=int, default=5,
        help="Number of previous rounds to retain in agent memory (default: 5)"
    )
    parser.add_argument(
        "--enable_shared_memory", action="store_true",
        help="Enable shared memory pool accessible by all agents"
    )
    parser.add_argument(
        "--langsmith_project", type=str, default="MIT_beer_game_Langsmith",
        help="LangSmith project name for tracing (default: MIT_beer_game_Langsmith)"
    )
    # New arguments for initial agent values
    parser.add_argument(
        "--initial_inventory", type=int, default=100,
        help="Initial inventory for all agents (default: 100)"
    )
    parser.add_argument(
        "--initial_backlog", type=int, default=0,
        help="Initial backlog for all agents (default: 0)"
    )
    parser.add_argument(
        "--sale_price", type=float, default=5.0,
        help="Sale price per unit shipped downstream (default: 5.0)"
    )
    parser.add_argument(
        "--purchase_cost", type=float, default=2.5,
        help="Purchase cost per unit ordered upstream (default: 2.5)"
    )
    parser.add_argument(
        "--production_cost", type=float, default=1.5,
        help="Factory production cost per unit (default: 1.5)"
    )
    parser.add_argument(
        "--initial_balance", type=float, default=1000.0,
        help="Initial bank account balance for all agents (default: 1000.0)"
    )
    # Orchestrator options
    parser.add_argument(
        "--enable_orchestrator", action="store_true",
        help="Enable chain-level LLM orchestrator that gives order recommendations"
    )
    parser.add_argument(
        "--orchestrator_history", type=int, default=3,
        help="How many past rounds the orchestrator sees in its prompt (default: 3)"
    )
    parser.add_argument(
        "--orchestrator_override", action="store_true",
        help="If set, orchestrator recommendations override agent order quantities"
    )
    parser.add_argument(
        "--longtermplanning_boolean", action="store_true",
        help="Enable collaborative long-term planning mode. If enabled, agents optimize for collective supply chain success. If disabled (default), agents focus on individual profit maximization."
    )
    # New hyperparameter arguments for inventory management
    parser.add_argument(
        "--safety_stock_target", type=float, default=60.0,
        help="Target safety stock level S_s for all agents (default: 10.0 units)"
    )
    parser.add_argument(
        "--backlog_clearance_rate", type=float, default=0.5,
        help="Backlog clearance rate γ ∈ [0,1] for inventory management (default: 0.5)"
    )
    parser.add_argument(
        "--demand_smoothing_factor", type=float, default=0.3,
        help="Demand smoothing parameter δ for order quantity adjustments (default: 0.3)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Select provider and set up client/model
    if args.provider == "anthropic":
        from llm_calls_mitb_game import AnthropicLLMClient
        llm_calls_mitb_game.lite_client = AnthropicLLMClient()
        llm_calls_mitb_game.MODEL_NAME = args.anthropic_model
        print(f"🔄 Using Anthropic Claude model: {args.anthropic_model}")
    else:
        # Default to LiteLLM provider (OpenAI-compatible)
        from llm_calls_mitb_game import LiteLLMClient
        llm_calls_mitb_game.lite_client = LiteLLMClient()
        llm_calls_mitb_game.MODEL_NAME = args.model_name
        print(f"🔄 Using LiteLLM/OpenAI model: {args.model_name}")

    if args.langsmith_project:
        import os
        os.environ["LANGSMITH_PROJECT"] = args.langsmith_project
        # Set Pydantic serialization mode to handle type conversion
        os.environ["LANGSMITH_SERIALIZATION_MODE"] = "python"

    # nest_asyncio already applied in import block if available

    # Run simulation
    sim_data: SimulationData = asyncio.run(
        run_beer_game_simulation(
            num_rounds=args.num_rounds,
            temperature=args.temperature,
            enable_communication=args.enable_communication,
            communication_rounds=args.communication_rounds,
            enable_memory=args.enable_memory,
            memory_retention_rounds=args.memory_retention_rounds,
            enable_shared_memory=args.enable_shared_memory,
            initial_inventory=args.initial_inventory,
            initial_backlog=args.initial_backlog,
            sale_price_per_unit=args.sale_price,
            purchase_cost_per_unit=args.purchase_cost,
            production_cost_per_unit=args.production_cost,
            initial_balance=args.initial_balance,
            holding_cost_per_unit=args.holding_cost_per_unit,
            backlog_cost_per_unit=args.backlog_cost_per_unit,
            enable_orchestrator=args.enable_orchestrator,
            orchestrator_history=args.orchestrator_history,
            orchestrator_override=args.orchestrator_override,
            longtermplanning_boolean=args.longtermplanning_boolean,
            safety_stock_target=args.safety_stock_target,
            backlog_clearance_rate=args.backlog_clearance_rate,
            demand_smoothing_factor=args.demand_smoothing_factor,
        )
    )

    # Output summary
    results = sim_data.to_dict()
    print("\n✅ Simulation complete!")
    # print("Simulation complete. Summary:")  # Commented out
    # print(results)  # Commented out


if __name__ == "__main__":
    main()            



# Example usage:
# python Games/2_MIT_Beer_Game/scripts/executeMITBeerGame.py \
#   --num_rounds 30 \
#   --temperature 0.8 \
#   --enable_memory \
#   --memory_retention_rounds 7 \
#   --communication_rounds 3 \
#   --initial_inventory 150 \
#   --initial_backlog 10

# python Games/2_MIT_Beer_Game/scripts/executeMITBeerGame.py \
#   --num_rounds 50 \
#   --holding_cost_per_unit 0.75 \
#   --backlog_cost_per_unit 2.0 \
#   --profit_per_unit_sold 6.0 \
#   --temperature 0.9 \
#   --model_name "gpt-4o-mini" \
#   --enable_communication \
#   --communication_rounds 4 \
#   --enable_memory \
#   --memory_retention_rounds 10 \
#   --enable_shared_memory \
#   --langsmith_project "my_beer_game_experiment" \
#   --initial_inventory 80 \
#   --initial_backlog 20


# or like this 
# python executeMITBeerGame.py --provider anthropic --anthropic_model claude-3-haiku-20240307  --num_rounds 5