#!/usr/bin/env python3
"""
simulate_chain.py - Run a Chinese Whispers chain of LLM agents using LangGraph
"""
import json
import asyncio
from typing import Dict, Any, List
from datetime import datetime
import os

import yaml
from langgraph.graph import StateGraph, END
from langsmith import traceable
from litellm import acompletion

# Import the centralized LiteLLM client
from llm_client import lite_client

# Import story definitions
from story_definitions import get_default_story, get_story_by_name, get_story_for_config, list_stories

# Import prompt definitions
from prompt_definitions import PROMPTS

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'  # End color
    BOLD = '\033[1m'

def print_success(message: str):
    print(f"{Colors.GREEN}✅{Colors.ENDC} {message}")

def print_error(message: str):
    print(f"{Colors.RED}❌{Colors.ENDC} {message}")

def print_info(message: str):
    print(f"{Colors.BLUE}ℹ️{Colors.ENDC} {message}")

def print_warning(message: str):
    print(f"{Colors.YELLOW}⚠️{Colors.ENDC} {message}")


def create_run_folder() -> str:
    """Create a timestamped run folder in the results directory"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    run_folder = os.path.join(results_dir, f"run_{timestamp}")
    
    # Create the results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    # Create the run folder
    os.makedirs(run_folder, exist_ok=True)
    
    return run_folder


class ChainState(Dict[str, Any]):
    """State for the Chinese Whispers chain"""
    story: str
    history: List[Dict[str, Any]]
    current_step: int
    total_steps: int


class ChineseWhispersChain:
    """Orchestrates the Chinese Whispers chain using LangGraph"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = config["model"]
        self.temperature = config["temperature"]
        self.max_tokens = config["max_tokens"]
        self.num_agents = config["num_agents"]
        
    @traceable(name="rewrite_story")
    async def rewrite_story(self, story: str, step: int):
        """Have an agent rewrite the story. Returns dict with new story and prompts."""
        print_info(f"Agent {step}/{self.num_agents} is rewriting the story...")
        
        prompt_idx = (step - 1) % len(PROMPTS)
        prompt_entry = PROMPTS[prompt_idx]

        # Determine system & user prompts depending on structure
        if isinstance(prompt_entry, dict):
            system_template = prompt_entry.get("system", "You are a helpful assistant.")
            user_template = prompt_entry.get("user", "{story}")

            try:
                system_prompt = system_template.format(i=step, N=self.num_agents, story=story)
            except KeyError:
                system_prompt = system_template

            try:
                user_prompt = user_template.format(i=step, N=self.num_agents, story=story)
            except KeyError:
                user_prompt = user_template
        else:
            # Backwards-compatibility: prompt_entry is a plain string
            try:
                user_prompt = prompt_entry.format(story=story, i=step, N=self.num_agents)
            except KeyError:
                user_prompt = prompt_entry
            system_prompt = "You are a helpful assistant that rewrites stories to make them more interesting while keeping their essence."

        try:
            response_str = await lite_client.chat_completion(
                model=self.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                task_type="story_rewrite",
                step=step
            )
            
            new_story = response_str.strip()
            print_success(f"Agent {step} completed successfully")

            return {
                "story": new_story,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt
            }
            
        except Exception as e:
            print_error(f"Agent {step} failed: {str(e)}")
            # Print more specific error information
            if "authentication" in str(e).lower() or "api key" in str(e).lower():
                print_error("API Key issue detected. Please check your .env file:")
                print_error("- Ensure LITELLM_API_KEY is set correctly")
                print_error("- Check for any extra spaces or formatting issues")
            raise e
    
    async def agent_node(self, state: ChainState) -> ChainState:
        """Process one agent in the chain"""
        step = state["current_step"]
        story = state["story"]
        
        # Rewrite the story and get prompts used
        rewrite_result = await self.rewrite_story(story, step)
        new_story = rewrite_result["story"]
        system_prompt = rewrite_result["system_prompt"]
        user_prompt = rewrite_result["user_prompt"]
        
        # Record in history
        state["history"].append({
            "step": step,
            "story": new_story,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Update state
        state["story"] = new_story
        state["current_step"] = step + 1
        
        return state
    
    def should_continue(self, state: ChainState) -> str:
        """Determine if we should continue to the next agent"""
        if state["current_step"] > state["total_steps"]:
            return END
        return "agent"
    
    def build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(ChainState)
        
        # Add the agent node
        workflow.add_node("agent", self.agent_node)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add conditional edge
        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                "agent": "agent",
                END: END
            }
        )
        
        return workflow.compile()
    
    async def run(self, initial_story: str) -> List[Dict[str, Any]]:
        """Run the Chinese Whispers chain"""
        graph = self.build_graph()
        
        initial_state = ChainState(
            story=initial_story,
            history=[{
                "step": 0,
                "story": initial_story,
                "timestamp": datetime.utcnow().isoformat()
            }],
            current_step=1,
            total_steps=self.num_agents
        )
        
        # Set recursion limit to accommodate any number of agents
        # Default is 25, but we want to support more agents
        config = {"recursion_limit": max(100, self.num_agents * 2)}
        final_state = await graph.ainvoke(initial_state, config=config)
        
        return final_state["history"]


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Chinese Whispers chain")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--story", default=None, help="Story name from story_definitions.py (overrides config)")
    parser.add_argument("--list-stories", action="store_true", help="List available stories and exit")
    parser.add_argument("--num-agents", type=int, default=None, help="Number of agents (overrides config)")
    args = parser.parse_args()
    
    if args.list_stories:
        list_stories()
        return
    
    print_info("=" * 50)
    print_info("Chinese Whispers Chain Simulation")
    print_info("=" * 50)
    
    # Create run folder at the start
    run_folder = create_run_folder()
    print_info(f"Created run folder: {run_folder}")
    
    # Load config
    try:
        print_info(f"Loading configuration from: {args.config}")
        with open(args.config) as f:
            config = yaml.safe_load(f)
        print_success("Configuration loaded successfully")
        
        # Override story if specified
        if args.story:
            try:
                story_data = get_story_for_config(args.story)
                config["initial_story"] = story_data["initial_story"]
                print_success(f"Using story: {args.story}")
                print_info(f"Story description: {story_data['story_description']}")
                print_info(f"Difficulty: {story_data['story_difficulty']}")
                print_info(f"Tags: {', '.join(story_data['story_tags'])}")
                
                # Add story metadata to config for saving
                config.update(story_data)
            except ValueError as e:
                print_error(f"Story error: {e}")
                return
        
        # Override number of agents if specified
        if args.num_agents is not None:
            config["num_agents"] = args.num_agents
            print_success(f"Using {args.num_agents} agents (overriding config)")
        
        print_info(f"Model: {config['model']}")
        print_info(f"Agents: {config['num_agents']}")
        print_info(f"Temperature: {config['temperature']}")
    except Exception as e:
        print_error(f"Failed to load config: {e}")
        return
    
    # Initialize chain
    try:
        print_info("Initializing Chinese Whispers chain...")
        chain = ChineseWhispersChain(config)
        print_success("Chain initialized successfully")
    except Exception as e:
        print_error(f"Failed to initialize chain: {e}")
        return
    
    # Run the chain
    try:
        history = await chain.run(config["initial_story"])
        
        print_info("-" * 50)
        print_success("Chain execution completed!")
        
        # Save history to run folder
        output_file = os.path.join(run_folder, "history.jsonl")
        with open(output_file, "w") as f:
            for entry in history:
                f.write(json.dumps(entry) + "\n")
        
        print_success(f"History saved to {output_file}")
        print_info(f"Final story: {history[-1]['story'][:150]}...")
        
        # Save session summary to run folder
        summary = lite_client.get_session_summary()
        if isinstance(summary, dict):
            summary_file = os.path.join(run_folder, "llm_inference_metrics.json")
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)
            
            print_info("-" * 50)
            print_success("LLM Session Summary:")
            print_info(f"  Total Calls: {summary['total_calls']}")
            print_info(f"  Total Cost: ${summary['total_cost_usd']}")
            print_info(f"  Total Tokens: {summary['total_tokens']}")
            print_info(f"  Total Time: {summary['total_inference_time_seconds']}s")
            print_success(f"Session summary saved to {summary_file}")
        
        # Create a run_info.json file with metadata
        run_info = {
            "timestamp": datetime.utcnow().isoformat(),
            "config": config,
            "num_steps": len(history),
            "run_folder": run_folder,
            "story_used": args.story if args.story else "from_config"
        }
        run_info_file = os.path.join(run_folder, "run_info.json")
        with open(run_info_file, "w") as f:
            json.dump(run_info, f, indent=2)
        print_success(f"Run info saved to {run_info_file}")
        
        print_success("Simulation completed successfully!")
        print_info(f"All files saved to: {run_folder}")
        
    except Exception as e:
        print_error(f"Chain execution failed: {e}")
        print_error("Please check your API keys and network connection")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 




#     # Run a complete new simulation
# python run_complete_pipeline.py

# # Or run steps individually
# python simulate_chain.py
# python generate_sql.py
# python evaluate_sql.py

# python simulate_chain.py --num-agents 25