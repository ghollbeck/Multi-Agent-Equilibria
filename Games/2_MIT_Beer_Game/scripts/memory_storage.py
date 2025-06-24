"""
Memory storage system for MIT Beer Game agents with LangGraph integration.
Provides individual and shared memory capabilities for agent strategy persistence.
"""
import json
import os
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import copy


@dataclass
class MemoryEntry:
    """Single memory entry containing strategic information"""
    timestamp: str
    round_number: int
    entry_type: str  # 'decision', 'communication', 'performance', 'strategy'
    agent_role: str
    data: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        return cls(**data)


class AgentMemory:
    """Individual agent memory storage for strategies, communication patterns, and performance metrics"""
    
    def __init__(self, agent_role: str, retention_rounds: int = 5):
        self.agent_role = agent_role
        self.retention_rounds = retention_rounds
        self.memories: List[MemoryEntry] = []
        self._current_round = 0
    
    def add_decision_memory(self, round_number: int, order_quantity: int, 
                          inventory: int, backlog: int, reasoning: str, 
                          confidence: float = 0.5) -> None:
        """Add a decision-making memory entry"""
        entry = MemoryEntry(
            timestamp=datetime.now().isoformat(),
            round_number=round_number,
            entry_type='decision',
            agent_role=self.agent_role,
            data={
                'order_quantity': order_quantity,
                'inventory': inventory,
                'backlog': backlog,
                'reasoning': reasoning,
                'confidence': confidence
            }
        )
        self._add_memory(entry)
    
    def add_communication_memory(self, round_number: int, message: str,
                               strategy_hint: str, collaboration_proposal: str,
                               information_shared: str) -> None:
        """Add a communication memory entry"""
        entry = MemoryEntry(
            timestamp=datetime.now().isoformat(),
            round_number=round_number,
            entry_type='communication',
            agent_role=self.agent_role,
            data={
                'message': message,
                'strategy_hint': strategy_hint,
                'collaboration_proposal': collaboration_proposal,
                'information_shared': information_shared
            }
        )
        self._add_memory(entry)
    
    def add_performance_memory(self, round_number: int, profit: float,
                             inventory: int, backlog: int, units_sold: int,
                             holding_cost: float, backlog_cost: float) -> None:
        """Add a performance outcome memory entry"""
        entry = MemoryEntry(
            timestamp=datetime.now().isoformat(),
            round_number=round_number,
            entry_type='performance',
            agent_role=self.agent_role,
            data={
                'profit': profit,
                'inventory': inventory,
                'backlog': backlog,
                'units_sold': units_sold,
                'holding_cost': holding_cost,
                'backlog_cost': backlog_cost
            }
        )
        self._add_memory(entry)
    
    def add_strategy_memory(self, round_number: int, strategy_description: str,
                          strategy_rationale: str, expected_outcome: str) -> None:
        """Add a strategic thinking memory entry"""
        entry = MemoryEntry(
            timestamp=datetime.now().isoformat(),
            round_number=round_number,
            entry_type='strategy',
            agent_role=self.agent_role,
            data={
                'strategy_description': strategy_description,
                'strategy_rationale': strategy_rationale,
                'expected_outcome': expected_outcome
            }
        )
        self._add_memory(entry)
    
    def _add_memory(self, entry: MemoryEntry) -> None:
        """Add memory entry and enforce retention policy"""
        self.memories.append(entry)
        self._current_round = max(self._current_round, entry.round_number)
        self._enforce_retention()
    
    def _enforce_retention(self) -> None:
        """Remove memories older than retention_rounds"""
        if self.retention_rounds <= 0:
            return
        
        cutoff_round = self._current_round - self.retention_rounds
        self.memories = [m for m in self.memories if m.round_number > cutoff_round]
    
    def get_recent_memories(self, entry_types: Optional[List[str]] = None,
                          max_entries: Optional[int] = None) -> List[MemoryEntry]:
        """Get recent memories, optionally filtered by type"""
        memories = self.memories
        
        if entry_types:
            memories = [m for m in memories if m.entry_type in entry_types]
        
        memories = sorted(memories, key=lambda x: x.round_number, reverse=True)
        
        if max_entries:
            memories = memories[:max_entries]
        
        return memories
    
    def get_memory_context_for_decision(self) -> str:
        """Generate formatted memory context for decision-making prompts"""
        recent_decisions = self.get_recent_memories(['decision'], max_entries=3)
        recent_performance = self.get_recent_memories(['performance'], max_entries=3)
        recent_strategies = self.get_recent_memories(['strategy'], max_entries=2)
        
        context_parts = []
        
        if recent_decisions:
            context_parts.append("Recent Decisions:")
            for mem in recent_decisions:
                data = mem.data
                context_parts.append(f"  Round {mem.round_number}: Ordered {data['order_quantity']} units")
                context_parts.append(f"    Inventory: {data['inventory']}, Backlog: {data['backlog']}")
                context_parts.append(f"    Reasoning: {data['reasoning']}")
        
        if recent_performance:
            context_parts.append("\nRecent Performance:")
            for mem in recent_performance:
                data = mem.data
                context_parts.append(f"  Round {mem.round_number}: Profit ${data['profit']:.2f}")
                context_parts.append(f"    Sold {data['units_sold']} units, Inventory: {data['inventory']}, Backlog: {data['backlog']}")
        
        if recent_strategies:
            context_parts.append("\nPast Strategies:")
            for mem in recent_strategies:
                data = mem.data
                context_parts.append(f"  Round {mem.round_number}: {data['strategy_description']}")
                context_parts.append(f"    Rationale: {data['strategy_rationale']}")
        
        return "\n".join(context_parts) if context_parts else "No relevant memory context available."
    
    def get_memory_context_for_communication(self) -> str:
        """Generate formatted memory context for communication prompts"""
        recent_communications = self.get_recent_memories(['communication'], max_entries=2)
        recent_decisions = self.get_recent_memories(['decision'], max_entries=2)
        
        context_parts = []
        
        if recent_communications:
            context_parts.append("Recent Communications:")
            for mem in recent_communications:
                data = mem.data
                context_parts.append(f"  Round {mem.round_number}: {data['message']}")
                context_parts.append(f"    Strategy Hint: {data['strategy_hint']}")
        
        if recent_decisions:
            context_parts.append("\nRecent Decision Context:")
            for mem in recent_decisions:
                data = mem.data
                context_parts.append(f"  Round {mem.round_number}: Ordered {data['order_quantity']}, Inventory: {data['inventory']}")
        
        return "\n".join(context_parts) if context_parts else "No relevant communication context available."
    
    def clear_memory(self) -> None:
        """Clear all memories"""
        self.memories.clear()
        self._current_round = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Export memory to dictionary"""
        return {
            'agent_role': self.agent_role,
            'retention_rounds': self.retention_rounds,
            'current_round': self._current_round,
            'memories': [mem.to_dict() for mem in self.memories]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMemory':
        """Import memory from dictionary"""
        memory = cls(data['agent_role'], data['retention_rounds'])
        memory._current_round = data['current_round']
        memory.memories = [MemoryEntry.from_dict(mem_data) for mem_data in data['memories']]
        return memory


class SharedMemory:
    """Optional shared memory pool accessible by all agents"""
    
    def __init__(self, retention_rounds: int = 5):
        self.retention_rounds = retention_rounds
        self.shared_memories: List[MemoryEntry] = []
        self._current_round = 0
    
    def add_shared_insight(self, round_number: int, insight_type: str,
                          description: str, contributing_agent: str,
                          data: Dict[str, Any]) -> None:
        """Add a shared insight from any agent"""
        entry = MemoryEntry(
            timestamp=datetime.now().isoformat(),
            round_number=round_number,
            entry_type=f'shared_{insight_type}',
            agent_role=contributing_agent,
            data={
                'description': description,
                'contributing_agent': contributing_agent,
                **data
            }
        )
        self._add_shared_memory(entry)
    
    def add_market_observation(self, round_number: int, observation: str,
                             market_data: Dict[str, Any]) -> None:
        """Add market-level observation visible to all agents"""
        entry = MemoryEntry(
            timestamp=datetime.now().isoformat(),
            round_number=round_number,
            entry_type='market_observation',
            agent_role='system',
            data={
                'observation': observation,
                'market_data': market_data
            }
        )
        self._add_shared_memory(entry)
    
    def _add_shared_memory(self, entry: MemoryEntry) -> None:
        """Add shared memory entry and enforce retention policy"""
        self.shared_memories.append(entry)
        self._current_round = max(self._current_round, entry.round_number)
        self._enforce_retention()
    
    def _enforce_retention(self) -> None:
        """Remove shared memories older than retention_rounds"""
        if self.retention_rounds <= 0:
            return
        
        cutoff_round = self._current_round - self.retention_rounds
        self.shared_memories = [m for m in self.shared_memories if m.round_number > cutoff_round]
    
    def get_shared_context(self, requesting_agent: str) -> str:
        """Get shared memory context for a specific agent"""
        recent_shared = sorted(self.shared_memories, key=lambda x: x.round_number, reverse=True)[:5]
        
        if not recent_shared:
            return "No shared insights available."
        
        context_parts = ["Shared Market Insights:"]
        for mem in recent_shared:
            data = mem.data
            if mem.entry_type == 'market_observation':
                context_parts.append(f"  Round {mem.round_number}: {data['observation']}")
            else:
                context_parts.append(f"  Round {mem.round_number}: {data['description']} (from {data['contributing_agent']})")
        
        return "\n".join(context_parts)
    
    def clear_shared_memory(self) -> None:
        """Clear all shared memories"""
        self.shared_memories.clear()
        self._current_round = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Export shared memory to dictionary"""
        return {
            'retention_rounds': self.retention_rounds,
            'current_round': self._current_round,
            'shared_memories': [mem.to_dict() for mem in self.shared_memories]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SharedMemory':
        """Import shared memory from dictionary"""
        memory = cls(data['retention_rounds'])
        memory._current_round = data['current_round']
        memory.shared_memories = [MemoryEntry.from_dict(mem_data) for mem_data in data['shared_memories']]
        return memory


class MemoryManager:
    """Handles memory persistence, retrieval, and retention policies"""
    
    def __init__(self, base_path: str = ".", retention_rounds: int = 5, enable_shared_memory: bool = False):
        self.base_path = base_path
        self.retention_rounds = retention_rounds
        self.enable_shared_memory = enable_shared_memory
        self.agent_memories: Dict[str, AgentMemory] = {}
        self.shared_memory: Optional[SharedMemory] = None
        
        if self.enable_shared_memory:
            self.shared_memory = SharedMemory(retention_rounds)
    
    def initialize_agent_memory(self, agent_role: str) -> AgentMemory:
        """Initialize memory for a specific agent"""
        if agent_role not in self.agent_memories:
            self.agent_memories[agent_role] = AgentMemory(agent_role, self.retention_rounds)
        return self.agent_memories[agent_role]
    
    def create_agent_memory(self, agent_role: str) -> AgentMemory:
        """Create and return memory for a specific agent (alias for initialize_agent_memory)"""
        return self.initialize_agent_memory(agent_role)
    
    def initialize_shared_memory(self) -> SharedMemory:
        """Initialize shared memory pool"""
        if self.shared_memory is None:
            self.shared_memory = SharedMemory(self.retention_rounds)
        return self.shared_memory
    
    def get_agent_memory(self, agent_role: str) -> Optional[AgentMemory]:
        """Get memory for a specific agent"""
        return self.agent_memories.get(agent_role)
    
    def get_shared_memory(self) -> Optional[SharedMemory]:
        """Get shared memory pool"""
        return self.shared_memory
    
    def get_shared_memory_context(self, requesting_agent: str = "system") -> str:
        """Get shared memory context for agent decision making"""
        if self.shared_memory:
            return self.shared_memory.get_shared_context(requesting_agent)
        return "No shared memory available."
    
    def save_memories_to_file(self, filepath: str) -> None:
        """Save all memories to a JSON file"""
        memory_data = {
            'timestamp': datetime.now().isoformat(),
            'retention_rounds': self.retention_rounds,
            'agent_memories': {role: mem.to_dict() for role, mem in self.agent_memories.items()},
            'shared_memory': self.shared_memory.to_dict() if self.shared_memory else None
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(memory_data, f, indent=2)
    
    def load_memories_from_file(self, filepath: str) -> bool:
        """Load memories from a JSON file"""
        try:
            if not os.path.exists(filepath):
                return False
            
            with open(filepath, 'r') as f:
                memory_data = json.load(f)
            
            self.retention_rounds = memory_data.get('retention_rounds', 5)
            
            agent_data = memory_data.get('agent_memories', {})
            for role, mem_data in agent_data.items():
                self.agent_memories[role] = AgentMemory.from_dict(mem_data)
            
            shared_data = memory_data.get('shared_memory')
            if shared_data:
                self.shared_memory = SharedMemory.from_dict(shared_data)
            
            return True
        except Exception as e:
            # print(f"Error loading memories from {filepath}: {e}")  # Commented out
            return False
    
    def clear_all_memories(self) -> None:
        """Clear all agent and shared memories"""
        for memory in self.agent_memories.values():
            memory.clear_memory()
        if self.shared_memory:
            self.shared_memory.clear_shared_memory()
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary statistics of current memory usage"""
        summary = {
            'retention_rounds': self.retention_rounds,
            'agent_count': len(self.agent_memories),
            'agents': {},
            'shared_memory_enabled': self.shared_memory is not None
        }
        
        for role, memory in self.agent_memories.items():
            summary['agents'][role] = {
                'total_memories': len(memory.memories),
                'memory_types': {}
            }
            
            for mem in memory.memories:
                mem_type = mem.entry_type
                if mem_type not in summary['agents'][role]['memory_types']:
                    summary['agents'][role]['memory_types'][mem_type] = 0
                summary['agents'][role]['memory_types'][mem_type] += 1
        
        if self.shared_memory:
            summary['shared_memory'] = {
                'total_memories': len(self.shared_memory.shared_memories),
                'memory_types': {}
            }
            
            for mem in self.shared_memory.shared_memories:
                mem_type = mem.entry_type
                if mem_type not in summary['shared_memory']['memory_types']:
                    summary['shared_memory']['memory_types'][mem_type] = 0
                summary['shared_memory']['memory_types'][mem_type] += 1
        
        return summary
