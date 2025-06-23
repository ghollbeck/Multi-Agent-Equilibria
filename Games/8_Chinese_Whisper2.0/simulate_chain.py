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
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langsmith import Client, traceable
from litellm import acompletion

load_dotenv()

# Configure LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "chinese_whispers_sql"


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
    async def rewrite_story(self, story: str, step: int) -> str:
        """Have an agent rewrite the story"""
        if step == 1:
            # First agent uses base template
            prompt = self.config["prompts"]["base_agent"].format(
                story=story,
                i=step,
                N=self.num_agents
            )
        elif step == self.num_agents:
            # Last agent uses final template
            prompt = self.config["prompts"]["final_agent"].format(
                story=story,
                i=step,
                N=self.num_agents
            )
        else:
            # Middle agents use base template
            prompt = self.config["prompts"]["base_agent"].format(
                story=story,
                i=step,
                N=self.num_agents
            )
        
        response = await acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return response.choices[0].message.content.strip()
    
    async def agent_node(self, state: ChainState) -> ChainState:
        """Process one agent in the chain"""
        step = state["current_step"]
        story = state["story"]
        
        # Rewrite the story
        new_story = await self.rewrite_story(story, step)
        
        # Record in history
        state["history"].append({
            "step": step,
            "story": new_story,
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
        
        final_state = await graph.ainvoke(initial_state)
        return final_state["history"]


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Chinese Whispers chain")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Initialize chain
    chain = ChineseWhispersChain(config)
    
    # Run the chain
    print(f"Running Chinese Whispers chain with {config['num_agents']} agents...")
    history = await chain.run(config["initial_story"])
    
    # Save history
    output_file = "history.jsonl"
    with open(output_file, "w") as f:
        for entry in history:
            f.write(json.dumps(entry) + "\n")
    
    print(f"Chain complete! History saved to {output_file}")
    print(f"Final story: {history[-1]['story'][:100]}...")


if __name__ == "__main__":
    asyncio.run(main()) 