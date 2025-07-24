"""
Simple workflow for MIT Beer Game simulation without LangGraph dependencies.
Coordinates agent communication, memory retrieval, decision making, and memory updates.
"""
import os
import json
import asyncio
import warnings
from typing import Dict, List, Any, Optional, TypedDict
from datetime import datetime
from dataclasses import dataclass, field

# Suppress Pydantic serialization warnings
warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.*")

# Force disable LangSmith to avoid rate limit issues
LANGSMITH_AVAILABLE = False

def traceable(name=None, **kwargs):
    def decorator(func):
        return func
    return decorator

from models_mitb_game import BeerGameAgent, SimulationData, RoundData
from memory_storage import AgentMemory, SharedMemory, MemoryManager


class BeerGameState(TypedDict):
    """State schema for the Beer Game workflow"""
    round_index: int
    generation: int
    total_rounds: int
    agents: List[BeerGameAgent]
    agent_states: Dict[str, Dict[str, Any]]
    communication_history: List[Dict[str, Any]]
    orders_placed: Dict[str, int]
    shipments_received: Dict[str, int]
    shipments_sent: Dict[str, int]
    external_demand: int
    simulation_data: SimulationData
    memory_manager: Optional[MemoryManager]
    enable_memory: bool
    enable_shared_memory: bool
    enable_communication: bool
    communication_rounds: int
    temperature: float
    profit_per_unit_sold: float


class BeerGameWorkflow:
    """Simple workflow for MIT Beer Game simulation with memory and communication."""
    
    def __init__(self, agents: List[BeerGameAgent], simulation_data: SimulationData,
                 memory_manager: Optional[MemoryManager] = None,
                 enable_memory: bool = False, enable_shared_memory: bool = False,
                 enable_communication: bool = False, communication_rounds: int = 2):
        self.agents = agents
        self.simulation_data = simulation_data
        self.memory_manager = memory_manager
        self.enable_memory = enable_memory
        self.enable_shared_memory = enable_shared_memory
        self.enable_communication = enable_communication
        self.communication_rounds = communication_rounds
        
        if self.enable_memory and self.memory_manager:
            for agent in self.agents:
                agent.memory = self.memory_manager.initialize_agent_memory(agent.role_name)
        
        if self.enable_shared_memory and self.memory_manager:
            self.memory_manager.initialize_shared_memory()
    
    @traceable
    async def _retrieve_memory_node(self, state: BeerGameState) -> BeerGameState:
        """Node: Retrieve relevant memories for each agent."""
        if not self.enable_memory or not self.memory_manager:
            return state
        
        for agent in state["agents"]:
            if agent.memory:
                memory_context = agent.load_memory_context()
                if agent.role_name not in state["agent_states"]:
                    state["agent_states"][agent.role_name] = {}
                state["agent_states"][agent.role_name]["memory_context"] = memory_context
        
        if self.enable_shared_memory:
            shared_memory = self.memory_manager.get_shared_memory()
            if shared_memory:
                shared_context = shared_memory.get_shared_context("system")
                state["agent_states"]["shared_context"] = shared_context
        
        return state
    
    @traceable
    async def _communication_phase_node(self, state: BeerGameState) -> BeerGameState:
        """Node: Run the communication phase where agents exchange messages."""
        if not state["enable_communication"]:
            return state
        
        round_messages = []
        
        for comm_round in range(state["communication_rounds"]):
            comm_round_messages = []
            
            for agent in state["agents"]:
                try:
                    message_response = await agent.generate_communication_message(
                        round_index=state["round_index"],
                        other_agents=[a for a in state["agents"] if a != agent],
                        message_history=state["communication_history"],
                        temperature=state["temperature"]
                    )
                    
                    message_entry = {
                        "round": state["round_index"],
                        "communication_round": comm_round + 1,
                        "sender": agent.role_name,
                        "message": message_response.get("message", ""),
                        "strategy_hint": message_response.get("strategy_hint", ""),
                        "collaboration_proposal": message_response.get("collaboration_proposal", ""),
                        "information_shared": message_response.get("information_shared", ""),
                        "confidence": message_response.get("confidence", 0.5)
                    }
                    
                    comm_round_messages.append(message_entry)
                    round_messages.append(message_entry)
                    
                except Exception as e:
                    # Skip this agent's communication on error
                    pass
            
            for agent in state["agents"]:
                agent.message_history.extend(comm_round_messages)

        state["communication_history"].extend(round_messages)
        
        for msg in round_messages:
            state["simulation_data"].add_communication_entry(msg)
        
        return state
    
    @traceable
    async def _decision_making_node(self, state: BeerGameState) -> BeerGameState:
        """Node: Agents make order decisions based on current state and communication."""
        decision_tasks = []
        recent_communications = []
        
        if state["enable_communication"]:
            recent_communications = [
                msg for msg in state["communication_history"] 
                if msg.get("round") == state["round_index"]
            ]
        
        for agent in state["agents"]:
            if state["enable_communication"] and recent_communications:
                task = agent.llm_decision(
                    "decision",
                    comm_history=recent_communications,
                    history_limit=10,
                    temperature=state["temperature"],
                    profit_per_unit_sold=state["profit_per_unit_sold"],
                    total_chain_inventory=sum(a.inventory for a in state["agents"]),
                    total_chain_backlog=sum(a.backlog for a in state["agents"]),
                )
            else:
                task = agent.llm_decision(
                    "decision",
                    comm_history=None,
                    history_limit=10,
                    temperature=state["temperature"],
                    profit_per_unit_sold=state["profit_per_unit_sold"],
                    total_chain_inventory=sum(a.inventory for a in state["agents"]),
                    total_chain_backlog=sum(a.backlog for a in state["agents"]),
                )
            decision_tasks.append(task)
        
        try:
            decisions = await asyncio.gather(*decision_tasks)
        except Exception as e:
            # Return safe defaults
            decisions = [{"order_quantity": 10, "confidence": 0.5, "rationale": "Default fallback"} 
                        for _ in state["agents"]]
        
        orders_placed = {}
        for i, (agent, decision) in enumerate(zip(state["agents"], decisions)):
            order_quantity = max(0, int(decision.get("order_quantity", 10)))
            orders_placed[agent.role_name] = order_quantity
            agent.last_order_placed = order_quantity
            
            if agent.role_name not in state["agent_states"]:
                state["agent_states"][agent.role_name] = {}
            state["agent_states"][agent.role_name]["decision_output"] = decision
        
        state["orders_placed"] = orders_placed
        return state
    
    @traceable
    async def _process_orders_node(self, state: BeerGameState) -> BeerGameState:
        """Node: Process orders and update agent states (inventory, backlog, profits)."""
        agents = state["agents"]
        orders_placed = state["orders_placed"]
        
        shipments_received = {}
        for i, agent in enumerate(agents):
            if i == len(agents) - 1:  # Factory
                shipments_received[agent.role_name] = orders_placed[agent.role_name]
            else:
                upstream_agent = agents[i + 1]
                available_inventory = upstream_agent.inventory
                requested = orders_placed[agent.role_name] if i < len(orders_placed) else 0
                shipments_received[agent.role_name] = min(available_inventory, requested)
        
        shipments_sent = {}
        for i, agent in enumerate(agents):
            if i == 0:  # Retailer
                demand = state["external_demand"]
                available = agent.inventory + shipments_received[agent.role_name]
                shipments_sent[agent.role_name] = min(available, demand)
            else:
                downstream_agent = agents[i - 1]
                downstream_order = orders_placed.get(downstream_agent.role_name, 0)
                available = agent.inventory + shipments_received[agent.role_name]
                shipments_sent[agent.role_name] = min(available, downstream_order)
        
        for i, agent in enumerate(agents):
            received = shipments_received[agent.role_name]
            sent = shipments_sent[agent.role_name]
            
            agent.inventory += received - sent
            
            if i == 0:  # Retailer
                unmet_demand = state["external_demand"] - sent
                agent.backlog += max(0, unmet_demand)
                agent.backlog = max(0, agent.backlog - sent)
            else:
                downstream_agent = agents[i - 1]
                downstream_order = orders_placed.get(downstream_agent.role_name, 0)
                unmet_order = downstream_order - sent
                agent.backlog += max(0, unmet_order)
                agent.backlog = max(0, agent.backlog - sent)
            
            holding_cost = agent.inventory * 0.5  # $0.5 per unit per round
            backlog_cost = agent.backlog * 1.0    # $1.0 per unit per round
            revenue = sent * state["profit_per_unit_sold"]
            # NEW: include purchase/production cost based on role
            if agent.role_name == "Factory":
                unit_cost = 1.5  # production cost per unit (could be parameterised)
            else:
                unit_cost = 2.5  # purchase cost per unit (could be parameterised)
            purchase_cost = state["orders_placed"].get(agent.role_name, 0) * unit_cost
            round_profit = revenue - holding_cost - backlog_cost - purchase_cost

            # Deduct purchase/production cost immediately from balance
            agent.balance -= purchase_cost

            agent.profit_accumulated += round_profit
            agent.last_profit = round_profit
            # NEW: Update profit and balance history
            agent.update_profit_history(round_profit, agent.profit_accumulated)
            
            if agent.role_name not in state["agent_states"]:
                state["agent_states"][agent.role_name] = {}
            state["agent_states"][agent.role_name]["performance_data"] = {
                "profit": round_profit,
                "units_sold": sent,
                "holding_cost": holding_cost,
                "backlog_cost": backlog_cost,
                "purchase_cost": purchase_cost
            }
            
            round_data = RoundData(
                generation=state["generation"],
                round_index=state["round_index"],
                role_name=agent.role_name,
                inventory=agent.inventory,
                backlog=agent.backlog,
                order_placed=orders_placed[agent.role_name],
                shipment_received=received,
                shipment_sent_downstream=sent,
                profit=agent.profit_accumulated
            )
            state["simulation_data"].add_round_entry(round_data)
        
        state["shipments_received"] = shipments_received
        state["shipments_sent"] = shipments_sent
        
        return state
    
    @traceable
    async def _update_memory_node(self, state: BeerGameState) -> BeerGameState:
        """Node: Update agent memories with new experiences."""
        if not state["enable_memory"] or not self.memory_manager:
            return state
        
        for agent in state["agents"]:
            if agent.memory:
                agent_state = state["agent_states"].get(agent.role_name, {})
                decision_output = agent_state.get("decision_output")
                performance_data = agent_state.get("performance_data")
                
                communication_output = None
                agent_communications = [
                    msg for msg in state["communication_history"]
                    if msg.get("sender") == agent.role_name and msg.get("round") == state["round_index"]
                ]
                if agent_communications:
                    communication_output = agent_communications[-1]  # Most recent message
                
                agent.update_memory(
                    round_number=state["round_index"],
                    decision_output=decision_output,
                    communication_output=communication_output,
                    performance_data=performance_data
                )
        
        if state["enable_shared_memory"]:
            shared_memory = self.memory_manager.get_shared_memory()
            if shared_memory:
                total_inventory = sum(agent.inventory for agent in state["agents"])
                total_backlog = sum(agent.backlog for agent in state["agents"])
                total_orders = sum(state["orders_placed"].values())
                
                market_data = {
                    "total_inventory": total_inventory,
                    "total_backlog": total_backlog,
                    "total_orders": total_orders,
                    "external_demand": state["external_demand"]
                }
                
                observation = f"Market state: {total_inventory} inventory, {total_backlog} backlog, {total_orders} orders"
                shared_memory.add_market_observation(
                    round_number=state["round_index"],
                    observation=observation,
                    market_data=market_data
                )
        
        return state
    
    async def run_round(self, initial_state: BeerGameState) -> BeerGameState:
        """Run a single round of the Beer Game workflow."""
        # Simple sequential execution without LangGraph
        state = initial_state
        state = await self._retrieve_memory_node(state)
        state = await self._communication_phase_node(state)
        state = await self._decision_making_node(state)
        state = await self._process_orders_node(state)
        state = await self._update_memory_node(state)
        
        return state
    
    def create_initial_state(self, round_index: int, generation: int, total_rounds: int,
                           external_demand: int, temperature: float = 0.7,
                           profit_per_unit_sold: float = 5.0) -> BeerGameState:
        """Create initial state for a round."""
        return BeerGameState(
            round_index=round_index,
            generation=generation,
            total_rounds=total_rounds,
            agents=self.agents,
            agent_states={},
            communication_history=[],
            orders_placed={},
            shipments_received={},
            shipments_sent={},
            external_demand=external_demand,
            simulation_data=self.simulation_data,
            memory_manager=self.memory_manager,
            enable_memory=self.enable_memory,
            enable_shared_memory=self.enable_shared_memory,
            enable_communication=self.enable_communication,
            communication_rounds=self.communication_rounds,
            temperature=temperature,
            profit_per_unit_sold=profit_per_unit_sold
        )
