# MIT Beer Game: Multi-Agent Strategic Behavior Research Framework

## Overview

The MIT Beer Game is a sophisticated multi-agent simulation framework designed to investigate strategic behavior, emergent dynamics, and coordination mechanisms in Large Language Model (LLM) driven agents within a supply chain management context. This implementation serves as a core component of the Multi-Agent Equilibria research project, enabling systematic study of how AI agents develop strategic behaviors, memory formation, and collective intelligence patterns.

## Research Objectives

### Primary Research Questions

1. **Strategic Memory Formation**: How do LLM agents develop and retain strategic insights across multiple rounds of interaction?
2. **Collective Intelligence Emergence**: What patterns emerge when agents share memory and coordinate strategies?
3. **Multi-Agent Coordination**: How do memory-enhanced agents coordinate to minimize the bullwhip effect while maximizing individual profits?
4. **Behavioral Evolution**: How do agent strategies adapt and evolve over extended simulation periods?
5. **Nash Equilibrium Convergence**: Do LLM agents converge to Nash equilibrium solutions, and how does memory affect this convergence?

### Research Hypotheses

- **H1**: Agents with memory will demonstrate improved strategic decision-making compared to memoryless agents
- **H2**: Shared memory pools will lead to better supply chain coordination and reduced bullwhip effect
- **H3**: Communication-enabled agents will develop more sophisticated coordination strategies
- **H4**: Memory retention length will correlate with strategic sophistication and performance

## Game Mechanics and Rules

### Supply Chain Structure

The Beer Game simulates a four-tier supply chain:

```
Customer Demand → Retailer → Wholesaler → Distributor → Factory → Supplier
```

### Agent Roles and Responsibilities

1. **Retailer**: 
   - Receives customer demand
   - Orders from Wholesaler
   - Manages inventory and backlog
   - Profit: $5 per unit sold, -$1 per unit inventory, -$2 per unit backlog

2. **Wholesaler**:
   - Receives orders from Retailer
   - Orders from Distributor  
   - Manages inventory and backlog
   - Profit: $3 per unit sold, -$1 per unit inventory, -$2 per unit backlog

3. **Distributor**:
   - Receives orders from Wholesaler
   - Orders from Factory
   - Manages inventory and backlog
   - Profit: $2 per unit sold, -$1 per unit inventory, -$2 per unit backlog

4. **Factory**:
   - Receives orders from Distributor
   - Produces goods (2-round production delay)
   - Manages inventory and backlog
   - Profit: $1 per unit sold, -$1 per unit inventory, -$2 per unit backlog

### Game Flow

1. **Demand Generation**: Customer demand follows configurable patterns (constant, random, step function)
2. **Communication Phase** (optional): Agents exchange strategic information
3. **Decision Phase**: Each agent decides order quantity based on:
   - Current inventory level
   - Current backlog
   - Recent demand/order patterns
   - Memory context (if enabled)
   - Communication from other agents
4. **Fulfillment Phase**: Orders are processed, inventory updated, profits calculated
5. **Memory Update** (if enabled): Agents store decision context and outcomes

### Key Metrics

- **Individual Performance**: Profit, inventory levels, backlog, order quantities
- **System Performance**: Total supply chain cost, bullwhip effect magnitude
- **Strategic Metrics**: Strategy diversity, cooperation patterns, memory utilization
- **Equilibrium Analysis**: Nash deviation, Pareto efficiency, convergence rates

## Technical Implementation

### Core Architecture

```
MIT_Beer_Game.py           # Main simulation orchestrator
├── models_mitb_game.py    # Agent classes and behavior
├── prompts_mitb_game.py   # LLM prompt generation
├── llm_calls_mitb_game.py # LLM integration and cost tracking
├── memory_storage.py      # Memory management system
├── langraph_workflow.py   # LangGraph workflow coordination
└── analysis_mitb_game.py  # Results analysis and visualization
```

### Agent Implementation (`BeerGameAgent`)

```python
class BeerGameAgent:
    def __init__(self, role_name: str, logger: BeerGameLogger):
        self.role_name = role_name          # Retailer/Wholesaler/Distributor/Factory
        self.inventory = 100                # Starting inventory
        self.backlog = 0                   # Unfulfilled orders
        self.profit_accumulated = 0.0      # Total profit
        self.memory = None                 # AgentMemory instance (if enabled)
        self.strategy = {}                 # Current strategy parameters
        
    async def decide_order_quantity(self, temperature: float = 0.1) -> dict:
        """Make ordering decision using LLM with optional memory context"""
        
    async def communicate(self, message_history: List[Dict], 
                         other_agents: List[str]) -> dict:
        """Generate communication message to other agents"""
        
    def update_memory(self, round_number: int, decision_output: dict, 
                     performance_data: dict):
        """Store decision context and outcomes in memory"""
```

### Memory System Architecture

#### AgentMemory Class
```python
class AgentMemory:
    def __init__(self, agent_id: str, retention_rounds: int = 5):
        self.agent_id = agent_id
        self.retention_rounds = retention_rounds
        self.decision_memory = []          # Past decisions and reasoning
        self.communication_memory = []     # Past communications
        self.performance_memory = []       # Past performance metrics
        
    def add_decision_memory(self, round_number: int, order_quantity: int,
                           inventory: int, backlog: int, reasoning: str,
                           confidence: float):
        """Store decision context with automatic retention management"""
        
    def get_memory_context_for_decision(self) -> str:
        """Format memory for injection into decision prompts"""
        
    def get_memory_context_for_communication(self) -> str:
        """Format memory for injection into communication prompts"""
```

#### MemoryManager Class
```python
class MemoryManager:
    def __init__(self, retention_rounds: int = 5, enable_shared_memory: bool = False):
        self.retention_rounds = retention_rounds
        self.enable_shared_memory = enable_shared_memory
        self.agent_memories = {}           # Individual agent memories
        self.shared_memory = []            # Shared memory pool
        
    def create_agent_memory(self, agent_id: str) -> AgentMemory:
        """Create and register agent memory instance"""
        
    def get_shared_memory_context(self) -> str:
        """Get formatted shared memory context for all agents"""
```

### LLM Integration and Prompt Engineering

#### Decision Prompts
Decision prompts include:
- Current state (inventory, backlog, recent demand)
- Strategy context
- Memory context (if enabled)
- Communication context (if available)
- Performance feedback

#### Communication Prompts
Communication prompts include:
- Current state and strategy
- Message history
- Memory context of past communications
- Collaboration opportunities

#### Memory Context Injection
Memory is injected into user prompts (not system prompts) with format:
```
## Your Past Experiences and Learning

**Past Decision Patterns:**
- Round X: Ordered Y units (inventory: Z, backlog: W) - Reasoning: "..."
- Performance outcome: Profit change of $X.XX

**Past Communication Patterns:**  
- Round X: Shared information about demand trends
- Collaboration outcome: Improved coordination with Wholesaler

Use these past experiences to inform your current decision, but adapt to changing conditions.
```

### LangGraph Workflow Integration

```python
class BeerGameWorkflow:
    def __init__(self, agents: List[BeerGameAgent], memory_manager: MemoryManager):
        self.agents = agents
        self.memory_manager = memory_manager
        
    async def execute_round(self, round_number: int, customer_demand: int) -> dict:
        """Execute complete round with memory and communication coordination"""
        
        # Phase 1: Memory retrieval and context loading
        for agent in self.agents:
            agent.load_memory_context()
            
        # Phase 2: Communication (if enabled)
        if self.enable_communication:
            communication_results = await self.communication_phase()
            
        # Phase 3: Decision making with full context
        decisions = await self.decision_phase()
        
        # Phase 4: Order processing and fulfillment
        results = self.process_orders(decisions, customer_demand)
        
        # Phase 5: Memory updates
        for agent in self.agents:
            agent.update_memory(round_number, decisions[agent.role_name], results)
            
        return results
```

### Configuration Parameters

#### Command Line Arguments
```bash
python executeMITBeerGame.py \
  --num_generations 1 \                    # Number of simulation runs
  --num_rounds_per_generation 20 \         # Rounds per simulation
  --temperature 0.1 \                      # LLM temperature
  --enable_communication \                 # Enable agent communication
  --communication_rounds 2 \               # Communication rounds per game round
  --enable_memory \                        # Enable agent memory
  --memory_retention_rounds 10 \           # Memory retention length
  --enable_shared_memory \                 # Enable shared memory pool
  --langsmith_project MIT_beer_game_Langsmith  # LangSmith tracing project
```

#### Memory Configuration
- `enable_memory`: Boolean flag to enable/disable memory features
- `memory_retention_rounds`: Integer (5-20) controlling memory length
- `enable_shared_memory`: Boolean flag for shared memory pool
- Memory storage: Local only, cleared between simulation executions

#### Communication Configuration  
- `enable_communication`: Boolean flag for agent communication
- `communication_rounds`: Number of communication exchanges per round
- Communication includes: strategy hints, collaboration proposals, information sharing

## Data Collection and Logging

### CSV Output Format (`beer_game_detailed_log.csv`)
```csv
generation,round,agent_role,inventory,backlog,order_quantity,units_sold,profit_this_round,
profit_accumulated,customer_demand,llm_reported_inventory,llm_confidence,llm_rationale,
memory_context_length,communication_sent,communication_received
```

### JSON Output Format (`beer_game_detailed_log.json`)
```json
{
  "generation": 1,
  "round": 5,
  "agent_role": "Retailer", 
  "inventory": 85,
  "backlog": 12,
  "order_quantity": 25,
  "decision_reasoning": "Based on increasing demand trend...",
  "memory_context": "Past 3 rounds showed consistent demand increase...",
  "communication_log": [
    {"sender": "Wholesaler", "message": "Expecting supply constraints..."}
  ]
}
```

### LLM Metrics (`llm_inference_metrics.json`)
```json
{
  "timestamp": "2025-06-20T06:00:00Z",
  "model": "gpt-4o",
  "input_tokens": 1250,
  "output_tokens": 180,
  "cost": 0.0234,
  "inference_time": 2.3,
  "agent_role": "Retailer",
  "round_index": 5,
  "decision_type": "ordering",
  "memory_enabled": true,
  "communication_enabled": true
}
```

## Validation Framework

### Test Categories

#### 1. Memory System Validation
```python
def test_memory_retention():
    """Verify memory retention policies work correctly"""
    # Create agent with 5-round retention
    # Add 10 rounds of memory
    # Verify only last 5 rounds retained
    
def test_memory_context_injection():
    """Verify memory context appears in prompts"""
    # Add decision memory
    # Generate decision prompt
    # Verify memory context included in user prompt
    
def test_shared_memory_functionality():
    """Verify shared memory accessible by all agents"""
    # Agent A adds to shared memory
    # Agent B retrieves shared memory
    # Verify Agent B sees Agent A's information
```

#### 2. Strategic Behavior Validation
```python
def test_memory_improves_performance():
    """Verify memory-enabled agents outperform memoryless agents"""
    # Run simulation with memory disabled
    # Run simulation with memory enabled
    # Compare profit, bullwhip effect, coordination metrics
    
def test_communication_coordination():
    """Verify communication improves coordination"""
    # Run simulation without communication
    # Run simulation with communication
    # Measure coordination improvements
```

#### 3. LLM Integration Validation
```python
def test_prompt_memory_injection():
    """Verify memory context properly injected into prompts"""
    # Create agent with memory
    # Generate decision prompt
    # Verify memory section present and formatted correctly
    
def test_langsmith_tracing():
    """Verify LangSmith traces capture agent metadata"""
    # Run simulation with LangSmith enabled
    # Verify traces include agent_role, decision_type, round_index
```

#### 4. Data Integrity Validation
```python
def test_csv_format_preservation():
    """Verify all required CSV fields present and correctly formatted"""
    # Run simulation
    # Load CSV output
    # Verify all expected columns present with correct data types
    
def test_json_logging_completeness():
    """Verify JSON logs capture all decision context"""
    # Run simulation with memory and communication
    # Verify JSON includes memory_context and communication_log fields
```

### Performance Benchmarks

#### Memory Impact Metrics
- **Decision Quality**: Compare profit accumulation with/without memory
- **Strategic Sophistication**: Measure strategy diversity and adaptation
- **Coordination Efficiency**: Bullwhip effect reduction with memory

#### Communication Impact Metrics  
- **Information Sharing**: Frequency and quality of strategic information exchange
- **Collaboration Success**: Joint strategy development and execution
- **System Optimization**: Overall supply chain performance improvement

#### Computational Metrics
- **LLM Call Efficiency**: Token usage and cost per decision
- **Memory Overhead**: Storage and retrieval performance
- **Simulation Runtime**: End-to-end execution time

## Research Applications

### Experimental Designs

#### 1. Memory Length Impact Study
```python
# Test different memory retention lengths
memory_lengths = [3, 5, 10, 15, 20]
for length in memory_lengths:
    run_simulation(memory_retention_rounds=length)
    analyze_performance_correlation()
```

#### 2. Shared vs Individual Memory Comparison
```python
# Compare individual vs shared memory strategies
run_simulation(enable_memory=True, enable_shared_memory=False)   # Individual
run_simulation(enable_memory=True, enable_shared_memory=True)    # Shared
compare_coordination_patterns()
```

#### 3. Communication Strategy Evolution
```python
# Study how communication strategies evolve with memory
run_simulation(enable_communication=True, enable_memory=True, 
               num_rounds_per_generation=50)
analyze_communication_evolution()
```

### Expected Research Outcomes

1. **Memory Formation Patterns**: Documentation of how agents develop strategic memory
2. **Coordination Mechanisms**: Identification of effective multi-agent coordination strategies  
3. **Equilibrium Dynamics**: Analysis of Nash equilibrium convergence with memory
4. **Emergent Behaviors**: Discovery of unexpected strategic behaviors and adaptations
5. **Performance Optimization**: Guidelines for optimal memory and communication configurations

## Usage Examples

### Basic Memory-Enabled Simulation
```bash
python executeMITBeerGame.py \
  --num_generations 1 \
  --num_rounds_per_generation 20 \
  --enable_memory \
  --memory_retention_rounds 10 \
  --temperature 0.1
```

### Advanced Multi-Feature Simulation
```bash
python executeMITBeerGame.py \
  --num_generations 3 \
  --num_rounds_per_generation 30 \
  --enable_communication \
  --communication_rounds 2 \
  --enable_memory \
  --enable_shared_memory \
  --memory_retention_rounds 15 \
  --langsmith_project MIT_beer_game_research \
  --temperature 0.1
```

### Research Batch Execution
```bash
# Memory length study
for length in 5 10 15 20; do
  python executeMITBeerGame.py \
    --num_generations 5 \
    --num_rounds_per_generation 25 \
    --enable_memory \
    --memory_retention_rounds $length \
    --output_prefix "memory_${length}_"
done
```

## File Structure and Dependencies

### Core Files
- `MIT_Beer_Game.py` - Main simulation orchestrator
- `models_mitb_game.py` - Agent classes and behavior logic
- `prompts_mitb_game.py` - LLM prompt generation and memory integration
- `llm_calls_mitb_game.py` - LLM API integration with cost tracking
- `memory_storage.py` - Memory management system
- `langraph_workflow.py` - LangGraph workflow coordination
- `executeMITBeerGame.py` - Command-line interface
- `analysis_mitb_game.py` - Results analysis and visualization

### Test Files
- `test_langraph_integration.py` - Comprehensive integration tests
- `test_memory_system.py` - Memory system validation
- `test_communication_feature.py` - Communication feature tests

### Dependencies
```python
# Core dependencies
openai>=1.0.0
litellm>=1.0.0
langchain-core>=0.1.0
langgraph>=0.1.0
langsmith>=0.1.0

# Data processing
pandas>=1.5.0
numpy>=1.20.0
matplotlib>=3.5.0

# Async support
asyncio
nest_asyncio
```

## Troubleshooting and Common Issues

### Memory System Issues
- **Memory not persisting**: Verify `--enable_memory` flag is set
- **Memory context not in prompts**: Check memory retention rounds > 0
- **Shared memory not working**: Ensure `--enable_shared_memory` flag is set

### LangSmith Integration Issues
- **Traces not appearing**: Verify LANGCHAIN_API_KEY environment variable
- **Project not found**: Check LANGCHAIN_PROJECT matches `--langsmith_project`
- **Graceful fallback**: System continues working without LangSmith if unavailable

### Performance Issues
- **Slow execution**: Reduce number of rounds or disable communication
- **High costs**: Monitor LLM token usage in `llm_inference_metrics.json`
- **Memory overhead**: Reduce memory retention rounds for large simulations

This comprehensive framework enables sophisticated research into multi-agent strategic behavior, memory formation, and collective intelligence while maintaining rigorous experimental controls and detailed data collection for validation and analysis.
