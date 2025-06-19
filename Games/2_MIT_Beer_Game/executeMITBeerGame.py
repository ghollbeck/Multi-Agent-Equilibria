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
    print("Warning: nest_asyncio not available, running without it")

# Import modules
import llm_calls_mitb_game
from MIT_Beer_Game import run_beer_game_simulation
from models_mitb_game import SimulationData


def parse_args():
    parser = argparse.ArgumentParser(
        description="Execute the MIT Beer Game simulation with custom parameters"
    )
    parser.add_argument(
        "--num_generations", type=int, default=1,
        help="Number of generations to simulate"
    )
    parser.add_argument(
        "--num_rounds_per_generation", type=int, default=10,
        help="Number of rounds per generation"
    )
    parser.add_argument(
        "--holding_cost_per_unit", type=float, default=0.5,
        help="Holding cost per unit per round"
    )
    parser.add_argument(
        "--backlog_cost_per_unit", type=float, default=1.5,
        help="Backlog cost per unit per round"
    )
    parser.add_argument(
        "--profit_per_unit_sold", type=float, default=5.0,
        help="Profit earned per unit sold"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature for the LLM"
    )
    parser.add_argument(
        "--model_name", type=str, default=llm_calls_mitb_game.MODEL_NAME,
        help="Name of the LLM model to use"
    )
    parser.add_argument(
        "--enable_communication", action="store_true", default=True,
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
    return parser.parse_args()


def main():
    args = parse_args()

    # Override the default MODEL_NAME in llm_calls module
    llm_calls_mitb_game.MODEL_NAME = args.model_name

    if args.langsmith_project:
        import os
        os.environ["LANGSMITH_PROJECT"] = args.langsmith_project

    # nest_asyncio already applied in import block if available

    # Run simulation
    loop = asyncio.get_event_loop()
    sim_data: SimulationData = loop.run_until_complete(
        run_beer_game_simulation(
            num_generations=args.num_generations,
            num_rounds_per_generation=args.num_rounds_per_generation,
            temperature=args.temperature,
            enable_communication=args.enable_communication,
            communication_rounds=args.communication_rounds,
            enable_memory=args.enable_memory,
            memory_retention_rounds=args.memory_retention_rounds,
            enable_shared_memory=args.enable_shared_memory
        )
    )

    # Output summary
    results = sim_data.to_dict()
    print("Simulation complete. Summary:")
    print(results)


if __name__ == "__main__":
    main()            