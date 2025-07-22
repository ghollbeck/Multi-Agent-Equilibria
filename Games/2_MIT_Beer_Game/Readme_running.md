# MIT Beer Game - Comprehensive Development Log & Documentation

## 📚 WHAT IS THE MIT BEER GAME?

### **Educational Background**
The MIT Beer Game is a classic **supply chain management simulation** developed at the MIT Sloan School of Management in the 1960s by Jay Forrester and others. It demonstrates the **Bullwhip Effect** - how small changes in consumer demand can cause increasingly large swings in orders upstream in the supply chain.

### **Educational Objectives**
- **Bullwhip Effect**: Understanding how demand amplification occurs in supply chains
- **Information Asymmetry**: How lack of communication causes suboptimal decisions
- **Lead Time Impact**: Understanding delays and their effect on ordering behavior
- **Coordination Challenges**: Why local optimization often leads to global suboptimization
- **Cost Management**: Balancing holding costs, backlog costs, and service levels

### **Traditional Game Setup**
- **4 Roles**: Retailer → Wholesaler → Distributor → Factory
- **Information Constraints**: Each player only sees orders from downstream (not end customer demand)
- **Lead Times**: 1-2 rounds between ordering and receiving inventory
- **Cost Structure**: Holding costs for excess inventory, backlog costs for unfilled orders
- **Goal**: Minimize individual total cost (holding + backlog costs)

### **Common Outcomes in Traditional Game**
- **Demand Amplification**: Small customer demand changes create huge upstream swings
- **Bullwhip Pattern**: Orders oscillate wildly as players over-correct
- **High Costs**: Poor coordination leads to either excess inventory or stockouts
- **Learning Effect**: Subsequent rounds often show improvement with better communication

---

## 🤖 LLM-ENHANCED MIT BEER GAME

### **Our Innovation: AI-Powered Agents**
This simulation replaces human players with **Large Language Model (LLM) agents** that:
- **Learn and Adapt**: Develop and update ordering strategies based on performance
- **Communicate**: Exchange information and coordinate (when enabled)
- **Remember**: Utilize memory systems to learn from past experiences
- **Optimize**: Maximize individual profit while considering supply chain dynamics

### **Key Simulation Features**
- **Role-Specific Prompts**: Each agent (Retailer, Wholesaler, Distributor, Factory) receives tailored context
- **Realistic Constraints**: Shipment constraints prevent unrealistic oversupplying
- **Economic Modeling**: Holding costs ($0.50/unit/round), backlog costs ($1.50/unit/round), profits ($5.00/unit sold)
- **Dynamic Strategy**: Agents can update strategies between generations
- **Communication System**: Agents can coordinate via structured messaging
- **Memory Integration**: Past experiences inform future decisions
- **Full State Access**: Agents have complete access to their current balance, inventory, and backlog during both communication and decision phases

### **Supply Chain Flow**
```
External Demand → [Retailer] → [Wholesaler] → [Distributor] → [Factory] → Production
                      ↑           ↑            ↑           ↑
                   Orders      Orders       Orders      Orders
                      ↓           ↓            ↓           ↓
                  Shipments   Shipments   Shipments   Shipments
```

### **Agent Information Access During Communication**
**CONFIRMED**: Agents have **full access** to their current state information during communication rounds, including:

- ✅ **Current Inventory**: Exact inventory level (units)
- ✅ **Current Backlog**: Unfilled orders requiring fulfillment  
- ✅ **Total Profit/Balance**: Complete financial status (${profit_accumulated:.2f})
- ✅ **Recent Orders**: History of downstream demand/orders
- ✅ **Last Order Placed**: Most recent ordering decision
- ✅ **Current Strategy**: Strategic approach being followed
- ✅ **Communication History**: Previous messages from other agents

**Implementation Location**: `models_mitb_game.py` - `generate_communication_message()` method
**Prompt Structure**: Same comprehensive state information provided during decision-making phases

This ensures agents can make **informed communications** about their situation and propose **realistic coordination strategies** based on their actual financial and inventory status.

### **Agent Decision Process**
1. **Observe State**: Inventory, backlog, recent orders, incoming shipments
2. **Communicate**: Exchange information with other agents (if enabled)
3. **Decide Order**: Use LLM reasoning to determine optimal order quantity
4. **Execute**: Place orders and fulfill downstream demand
5. **Learn**: Update strategies based on performance and outcomes

---

## 🔄 RECENT DEVELOPMENT CHANGES

### **2024-01-XX - Major System Architecture Updates**

#### **1. ROLE-SPECIFIC SYSTEM PROMPTS IMPLEMENTATION**
**Files Modified:** `prompts_mitb_game.py`, `models_mitb_game.py`

**Change Description:**
- **BEFORE**: Single generic system prompt: `"You are an expert supply chain manager. Return valid JSON only."`
- **AFTER**: Role-specific system prompts via `BeerGamePrompts.get_system_prompt(role_name, enable_communication=True)`

**New System Prompt Features:**
- Explains MIT Beer Game context and 4-stage supply chain (Retailer → Wholesaler → Distributor → Factory)
- Sets agent's sole objective: **maximize YOUR cumulative profit**
- Lists agent capabilities: observe state, decide order_quantity, communicate (if enabled)
- Explains information provided in USER messages: state, costs, strategy, communications, memory
- Enforces JSON-only response requirement

**Implementation Details:**
- `BeerGamePrompts.get_system_prompt()` generates role-aware prompts
- All LLM calls in `BeerGameAgent` now use role-specific system prompts
- Communication-enabled calls include communication context in system prompt

**Impact:** Each agent (Retailer, Wholesaler, Distributor, Factory) now receives tailored context about their role and objectives.

---

#### **2. COMPREHENSIVE SYSTEM PROMPT LOGGING**
**Files Modified:** `models_mitb_game.py`, `MIT_Beer_Game.py`

**Change Description:**
- **BEFORE**: Only user prompts logged to human_readable_log.txt
- **AFTER**: Both system AND user prompts logged to terminal and human_readable_log.txt

**New Logging Fields Added:**
```python
# New storage fields in BeerGameAgent
last_decision_system_prompt: str = ""
last_update_system_prompt: str = ""  
last_init_system_prompt: str = ""
last_communication_system_prompt: str = ""
```

**Logging Output Format in human_readable_log.txt:**
```
LLM Decision System Prompt: [role-specific system prompt]
LLM Decision Prompt: [context-specific user prompt]
LLM Decision Output: [JSON response]

LLM Communication System Prompt: [role-specific system prompt with communication enabled]
LLM Communication Prompt: [communication context user prompt]  
LLM Communication Output: [JSON response]

LLM Update System Prompt: [role-specific system prompt]
LLM Update Prompt: [strategy update user prompt]
LLM Update Output: [JSON response]

LLM Init System Prompt: [role-specific system prompt]
LLM Init Prompt: [initial strategy user prompt]
LLM Init Output: [JSON response]
```

**Impact:** Complete visibility into all LLM interactions for debugging and analysis.

---

#### **3. SHIPMENT CONSTRAINT IMPLEMENTATION**
**Files Modified:** `MIT_Beer_Game.py`, `prompts_mitb_game.py`

**Change Description:**
- **BEFORE**: Agents could ship entire inventory regardless of orders received
- **AFTER**: **CRITICAL SHIPMENT CONSTRAINT**: Agents can only ship up to `(downstream_order + current_backlog)` units

**New Constraint Logic:**
```python
# For each agent in supply chain
max_allowed_shipment = downstream_order + agent.backlog
available_to_ship = min(agent.inventory, max_allowed_shipment)

# Apply constraint with logging
if logger and agent.inventory > max_allowed_shipment:
    logger.log(f"📦 [{agent.role_name}] Shipment constraint applied: inventory={agent.inventory}, max_allowed={max_allowed_shipment}")
```

**Constraint Applied To:**
- **Retailer**: Can ship max `(external_demand + retailer.backlog)`
- **Wholesaler**: Can ship max `(retailer_order + wholesaler.backlog)`  
- **Distributor**: Can ship max `(wholesaler_order + distributor.backlog)`
- **Factory**: Can ship max `(distributor_order + factory.backlog)`

**Prompt Updates:**
- **System Prompt**: Added constraint explanation to all role-specific prompts
- **Decision Prompts**: Added "SHIPMENT CONSTRAINT" section explaining the rule
- **Strategy Prompts**: Included constraint in strategy generation and updates
- **Communication Prompts**: Added shipment constraints as discussion topic

**Documentation Updates:**
- Added constraint explanation to file header documentation
- Added inline code comments explaining constraint logic
- Added constraint logging for debugging and transparency

**Impact:** 
- Realistic supply chain behavior (no oversupplying)
- Forces strategic inventory management
- Reduces bullwhip effect
- Agents understand and plan around constraint

---

#### **4. WARNING SUPPRESSION FOR LANGSMITH TRACING**
**Files Modified:** `langraph_workflow.py`

**Change Description:**
- **BEFORE**: `PydanticSerializationUnexpectedValue` errors cluttered terminal output
- **AFTER**: Clean terminal output with suppressed tracing warnings

**Implementation:**
```python
import warnings
warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.*")
```

**Impact:** Cleaner terminal output without affecting core functionality.

---

## 🏗️ CURRENT SYSTEM ARCHITECTURE

### **Prompt Structure:**
- **System Prompt**: Role-specific context, objectives, and rules (generated by `get_system_prompt()`)
- **User Prompt**: Dynamic task-specific instructions and data (generated by various prompt methods)

### **Agent Workflow:**
1. **Strategy Initialization**: Agent receives role-specific system prompt + strategy generation user prompt
2. **Order Decisions**: Agent receives role-specific system prompt + decision context user prompt  
3. **Communication**: Agent receives communication-enabled system prompt + communication user prompt
4. **Strategy Updates**: Agent receives role-specific system prompt + performance-based update prompt

### **Economic Model:**
- **Holding Cost**: $0.50 per unit per round (inventory storage cost)
- **Backlog Cost**: $1.50 per unit per round (penalty for unfilled orders)
- **Revenue**: $5.00 per unit sold (profit from fulfilling demand)
- **Lead Time**: 1 round delay between ordering and receiving inventory
- **Objective**: Maximize individual cumulative profit

### **Constraint Enforcement:**
- **Shipment Constraint**: `max_shipment = downstream_order + current_backlog`
- **Applied During**: Inventory fulfillment phase each round
- **Logged When**: Constraint prevents oversupplying (inventory > max_allowed)
- **Purpose**: Realistic supply chain behavior, prevents demand amplification

### **Logging Capabilities:**
- **Terminal Logging**: All LLM calls with system/user prompts
- **Human Readable Log**: Complete LLM interaction history with prompts and responses
- **CSV/JSON Logs**: Structured simulation data for analysis
- **Constraint Logging**: When shipment constraints are applied
- **Performance Metrics**: Costs, profits, inventory levels, order quantities

### **Communication System:**
- **Pre-Decision Communication**: Agents exchange messages before ordering decisions
- **Structured Messages**: Includes strategy hints, collaboration proposals, information sharing
- **Communication Rounds**: Multiple rounds of messaging per game round
- **Memory Integration**: Past communications inform future decisions

### **Memory System:**
- **Agent Memory**: Individual learning from past decisions and outcomes
- **Shared Memory**: Market-wide observations accessible to all agents
- **Memory Retention**: Configurable number of rounds to remember
- **Context Integration**: Memory incorporated into decision prompts

---

## 🔍 SEARCHABLE KEYWORDS FOR LLMS

**Core Game Concepts**: MIT Beer Game, Bullwhip Effect, Supply Chain Management, Multi-Agent Simulation, Inventory Management, Backlog Management, Order Amplification, Demand Forecasting, Strategic Planning  
**LLM Integration**: OpenAI GPT-4, Adaptive Strategies, Role-Specific Prompts, System Prompts, Temperature Control, JSON Response Validation, Strategy Evolution, Real-time Decision Making  
**Technical Implementation**: LangGraph Workflow, Agent Communication, Memory Storage, Shipment Constraints, Realistic Supply Chain Rules, Configurable Initial Values, Command Line Interface, Order Received Tracking, Demand Flow Analysis  
**Research Applications**: Nash Equilibrium Analysis, Strategy Convergence, Communication Impact, Collaborative Behavior, Supply Chain Optimization, Economic Game Theory, Multi-Agent Learning, Order Flow Patterns  
**Simulation Features**: CSV/JSON Logging, Visualization Plots, Performance Metrics, Cost Tracking, Human-Readable Logs, Parameter Configuration, Initial State Customization  
**Economic Focus**: Profit Maximization, Market Collapse Prevention, Cost-Benefit Analysis, Holding Cost Optimization, Backlog Cost Management, Profit Margin Analysis, Supply Chain Stability, Order-to-Fulfillment Tracking  
**Communication Strategy**: Profit Transparency, Inventory Sharing, Collaborative Planning, Risk Communication, Lead Time Awareness, Economic Coordination, Market Stability Warnings, Order Visibility

**Game Concepts**: MIT Beer Game, bullwhip effect, supply chain simulation, demand amplification, lead time, inventory management, backlog, holding cost

**System Prompts**: `get_system_prompt()`, role-specific prompts, Retailer prompt, Wholesaler prompt, Distributor prompt, Factory prompt, communication-enabled prompts

**Logging**: `human_readable_log.txt`, system prompt logging, `last_decision_system_prompt`, `last_communication_system_prompt`, LLM interaction logging

**Shipment Constraints**: `max_allowed_shipment`, `downstream_order + backlog`, shipment constraint enforcement, realistic supply chain, oversupplying prevention

**Prompt Types**: strategy generation, strategy update, order decision, communication prompt, decision with communication, memory-enhanced prompts

**Agent Workflow**: BeerGameAgent, LLM calls, temperature parameter, JSON response validation, prompt-response cycle

**Architecture**: modular prompts, system vs user prompts, role-aware agents, MIT Beer Game simulation, supply chain management

**Economics**: holding cost, backlog cost, profit maximization, cost minimization, inventory optimization, demand fulfillment

**Features**: agent communication, memory system, strategy adaptation, performance logging, constraint enforcement

**File Structure**: `prompts_mitb_game.py`, `models_mitb_game.py`, `MIT_Beer_Game.py`, `langraph_workflow.py`, simulation results

---

## 📋 SIMULATION EXECUTION

### **Running the Simulation:**
```bash
# Basic simulation
python Games/2_MIT_Beer_Game/scripts/executeMITBeerGame.py

# With advanced features
python Games/2_MIT_Beer_Game/scripts/executeMITBeerGame.py \
  --num_rounds 30 \
  --temperature 0.8 \
  --enable_communication \
  --enable_memory \
  --communication_rounds 3
```

### **Key Parameters:**
- `--num_rounds`: Number of simulation rounds (default: 10)
- `--temperature`: LLM sampling temperature for decision variability (default: 0)
- `--enable_communication`: Allow agents to exchange messages (default: True)
- `--enable_memory`: Enable agent learning and memory (default: False)
- `--communication_rounds`: Messages exchanged per round (default: 2)

### **Output Files:**
- `human_readable_log.txt`: Complete simulation narrative with all prompts/responses
- `beer_game_detailed_log.csv`: Structured data for analysis
- `beer_game_detailed_log.json`: Nested simulation data with agent states
- `llm_session_summary.json`: LLM usage statistics and costs
- **Plots**: Inventory, backlog, orders, and cost visualizations

---

## 📊 RESEARCH APPLICATIONS

### **Research Questions Addressable:**
- How do LLM agents compare to human players in supply chain coordination?
- What communication strategies emerge for reducing bullwhip effect?
- How does memory/learning affect long-term supply chain performance?
- Can LLM agents discover optimal supply chain policies?
- What role does information sharing play in supply chain efficiency?

### **Experimental Variables:**
- **Agent Configuration**: Different LLM models, temperatures, prompting strategies
- **Communication Settings**: Enabled/disabled, frequency, structured vs. free-form
- **Memory Systems**: Individual vs. shared memory, retention periods
- **Economic Parameters**: Cost structures, profit margins, demand patterns
- **Constraint Variations**: Different shipment rules, capacity limits

### **Performance Metrics:**
- **Individual Performance**: Profit, costs, inventory efficiency
- **System Performance**: Total costs, bullwhip effect magnitude, coordination level
- **Learning Metrics**: Strategy evolution, communication effectiveness
- **Emergent Behaviors**: Coordination patterns, information sharing strategies

---

## 🚀 FUTURE IMPROVEMENTS

### **Planned Enhancements:**
- [ ] Add constraint violation penalties/costs
- [ ] Implement dynamic constraint adjustment based on market conditions  
- [ ] Add constraint awareness to memory system
- [ ] Create constraint-specific communication strategies
- [ ] Add constraint impact analysis to post-simulation reports
- [ ] Multi-generation strategy evolution
- [ ] Advanced economic models (variable costs, capacity constraints)
- [ ] Real-time demand patterns and market shocks
- [ ] Integration with reinforcement learning approaches
- [ ] Comparative analysis with human player data

### **Research Extensions:**
- [ ] Multi-product supply chains
- [ ] Network topologies beyond linear chains
- [ ] Collaborative vs. competitive agent objectives
- [ ] Information sharing mechanisms and their effects
- [ ] Dynamic market conditions and adaptation strategies

---

## 📖 EDUCATIONAL USE

### **Learning Objectives:**
Students and researchers using this simulation can learn about:
- **Supply Chain Dynamics**: Real-world complexities of multi-stage inventory management
- **AI Decision Making**: How LLMs approach strategic planning and coordination
- **Prompt Engineering**: Designing effective prompts for complex decision tasks
- **System Design**: Building multi-agent simulations with realistic constraints
- **Performance Analysis**: Measuring and optimizing supply chain metrics

### **Use Cases:**
- **Academic Research**: Supply chain management, AI/ML, operations research
- **Educational Tool**: Business school courses, supply chain training
- **Industry Applications**: Testing coordination strategies, policy evaluation
- **AI Research**: Multi-agent systems, strategic reasoning, communication protocols

---

## 📊 Latest Changes & Implementation Log

### 🎯 Implementation 8: Order Received Tracking (Latest)
**Date**: 2025-01-20  
**Summary**: Added explicit tracking of new orders received from downstream customers each round

**Changes Made**:
1. **Enhanced RoundData Structure** (`models_mitb_game.py`):
   - Added `order_received: int` parameter to track new orders from downstream customers
   - Now logs both what agents ordered upstream AND what they received from downstream

2. **Order Flow Tracking** (`MIT_Beer_Game.py`):
   - Added `orders_received_from_downstream` array to track new customer orders
   - Retailer: tracks `retailer_demand` (external customer demand)
   - Wholesaler: tracks `retailer_order` (orders from Retailer)
   - Distributor: tracks `wh_order` (orders from Wholesaler)
   - Factory: tracks `dist_order` (orders from Distributor)

3. **Enhanced Logging**:
   - Human-readable logs now show "Orders received per agent" alongside shipments
   - CSV/JSON exports include the new `order_received` field
   - Logger output includes order tracking for debugging

**Business Value**: 
- Agents can now see exactly what new demand pressure came in each round
- Better understanding of order flow vs shipment flow
- Clearer visibility into supply chain demand propagation
- Enhanced analysis of bullwhip effect patterns

**Key Distinction**:
- `order_received` = New orders from downstream customers (demand to fulfill)
- `shipment_received` = Goods received from upstream suppliers (inventory replenishment)
- `order_placed` = Orders sent to upstream suppliers
- `shipment_sent_downstream` = Goods shipped to downstream customers

### 🎯 Implementation 7: Enhanced Communication & Profit Optimization
**Date**: 2025-01-20  
**Summary**: Enhanced communication prompts for profit-focused collaboration and updated default profit values

**Changes Made**:
1. **Enhanced Communication Prompts** (`prompts_mitb_game.py`):
   - Added "CRITICAL COMMUNICATION OBJECTIVES" section emphasizing:
     - 📊 PROFIT FOCUS: Share current profit status and threats
     - 📦 INVENTORY STATUS: Transparently share inventory/backlog situation
     - 🤝 COLLABORATIVE PROFIT: Propose total supply chain profit maximization
     - ⚠️ PREVENT COLLAPSE: Warn about market instability risks
   - Added "CRITICAL SUPPLY CHAIN CONTEXT" section with:
     - Lead time information (1 round for all orders)
     - Economics (holding $0.5, backlog $1.5, profit $2.5 per unit)
     - Shipment constraints and factory production details
   - Enhanced elaboration requirements for timing, costs, and stability

2. **Updated Default Profit Values**:
   - Changed profit_per_unit_sold from $5.00 to $2.50 throughout:
     - `prompts_mitb_game.py`: All prompt methods
     - `models_mitb_game.py`: All agent methods
     - `MIT_Beer_Game.py`: Simulation functions and hyperparameters
     - `executeMITBeerGame.py`: Command line default
   - Makes profit margins tighter, encouraging better coordination

**Communication Enhancement Example**:
```python
# Before: Generic sharing suggestions
"Consider sharing: demand patterns, coordination suggestions..."

# After: Specific profit-focused objectives
"CRITICAL COMMUNICATION OBJECTIVES:
📊 PROFIT FOCUS: Emphasize YOUR current profit ($X) and how decisions affect it
📦 INVENTORY STATUS: Share your inventory (X units) and backlog (Y units)
🤝 COLLABORATIVE PROFIT: Propose strategies to maximize TOTAL supply chain profits
⚠️ PREVENT COLLAPSE: Warn about risks of stockouts, oversupply, or market instability"
```

**Research Impact**:
- Agents now explicitly discuss profit trajectories and threats
- Enhanced awareness of supply chain timing (1-round lead time)
- Tighter profit margins ($2.50 vs $5.00) create more challenging optimization
- Communication focuses on preventing total market collapse
- Better alignment between individual and collective profit goals

---

### 🎯 Implementation 6: Configurable Initial Agent Values
**Date**: 2025-01-20  
**Summary**: Made agent starting conditions configurable via command line

**Changes Made**:
1. **Modified BeerGameAgent class** (`models_mitb_game.py`):
   - Changed `shipments_in_transit` default from `{0:10, 1:10}` to `{0:0, 1:0}` 
   - Added `create_agent()` class method for configurable initialization
   - Agents now start with zero shipments in transit for clean initial state

2. **Enhanced executeMITBeerGame.py**:
   - Added `--initial_inventory` parameter (default: 100)
   - Added `--initial_backlog` parameter (default: 0)
   - Parameters passed to simulation function

3. **Updated MIT_Beer_Game.py**:
   - Added initial value parameters to `run_beer_game_simulation()`
   - Used `BeerGameAgent.create_agent()` for consistent initialization
   - Added initial values to hyperparameters logging

**Before/After Comparison**:
```python
# BEFORE: Fixed values, agents received 10 units in first round
agents = [BeerGameAgent(role_name=role, logger=logger) for role in roles]
shipments_in_transit = {0:10, 1:10}  # Hard-coded initial shipments

# AFTER: Configurable values, clean start
agents = [BeerGameAgent.create_agent(role_name=role, 
                                    initial_inventory=initial_inventory,
                                    initial_backlog=initial_backlog, 
                                    logger=logger) for role in roles]
shipments_in_transit = {0:0, 1:0}  # Zero initial shipments
```

**Usage Examples**:
```bash
# Start with default values (inventory=100, backlog=0)
python executeMITBeerGame.py --num_rounds 20

# Start with higher inventory and some backlog
python executeMITBeerGame.py --initial_inventory 150 --initial_backlog 10

# Test extreme conditions
python executeMITBeerGame.py --initial_inventory 50 --initial_backlog 25 --num_rounds 30
```

**Research Applications**:
- Test different market conditions (high/low initial stock)
- Simulate supply chain disruptions (agents start with backlog)
- Compare agent behavior under resource constraints vs abundance
- Study equilibrium convergence from different starting points

---

### 🎯 Implementation 5: Comprehensive Documentation System
**Date**: 2025-01-19  
**Summary**: Created searchable documentation with complete MIT Beer Game background

*Last Updated: January 2024*
*Document Version: 2.0*
*Maintainer: MIT Beer Game Development Team* 

---

## 🏛️ Background: The MIT Beer Game 

## Latest Changes Log

### 2025-06-27: LangSmith Integration Disabled to Avoid Rate Limits

**Issue**: The MIT Beer Game simulation was encountering LangSmith rate limit errors when trying to trace LLM calls, causing error messages to flood the output:
```
Failed to multipart ingest runs: langsmith.utils.LangSmithRateLimitError: Rate limit exceeded for https://api.smith.langchain.com/runs/multipart
```

**Solution Applied**:
1. **Disabled LangSmith in `llm_calls_mitb_game.py`**: 
   - Set `LANGSMITH_AVAILABLE = False` at the top of the file
   - Commented out the LangSmith import to prevent any tracing attempts
   - Created a no-op `traceable` decorator function

2. **Simplified `langraph_workflow.py`**:
   - Removed all LangGraph dependencies (StateGraph, END, etc.)
   - Created a simple sequential workflow that doesn't require LangGraph
   - Removed complex graph building and just runs nodes sequentially

**Result**: 
- Simulation now runs cleanly without LangSmith rate limit errors
- All functionality preserved: communication, decision making, memory, logging
- Performance improved with cleaner output
- Successfully completed 10-round simulation with 84 LLM calls costing $0.87

**Key Files Modified**:
- `Games/2_MIT_Beer_Game/scripts/llm_calls_mitb_game.py` - Disabled LangSmith tracing
- `Games/2_MIT_Beer_Game/scripts/langraph_workflow.py` - Simplified to remove LangGraph dependencies

**Command to Run Without LangSmith**:
```bash
python Games/2_MIT_Beer_Game/scripts/executeMITBeerGame.py --num_rounds 10 --initial_inventory 50 --initial_backlog 0 --temperature 0 --communication_rounds 1 --profit_per_unit_sold 1.5 --backlog_cost_per_unit 1
```

The simulation logic remains intact and the Beer Game mechanics work properly with the fixed order flow from previous iterations. 

## Latest Implementation - Bank Account Balance System (2025-01-XX)

### Major Feature: Bank Account Balance System
- **Replaced profit tracking with bank account balance system**
- **Initial balance**: $1000 per agent (configurable via CLI)
- **Bankruptcy detection**: Simulation terminates if any agent balance ≤ 0

#### Changes Made:
1. **Models (`models_mitb_game.py`)**:
   - Added `balance: float = 1000.0` field to `BeerGameAgent`
   - Added compatibility alias `profit_accumulated` → `balance`
   - Updated `RoundData` to track: `starting_balance`, `revenue`, `purchase_cost`, `holding_cost`, `backlog_cost`, `ending_balance`
   - Modified `create_agent()` to accept `initial_balance` parameter

2. **Core Game Logic (`MIT_Beer_Game.py`)**:
   - **Revenue**: Added to balance when units are sold downstream
   - **Purchase costs**: Deducted immediately when orders are placed
   - **Holding/backlog costs**: Deducted each round based on inventory/backlog
   - **Bankruptcy check**: Simulation stops if any agent balance ≤ 0
   - Added monetary parameters: `sale_price_per_unit`, `purchase_cost_per_unit`, `production_cost_per_unit`

3. **CLI Interface (`executeMITBeerGame.py`)**:
   - Added `--initial_balance` argument (default: 1000.0)
   - Added `--sale_price`, `--purchase_cost`, `--production_cost` arguments
   - All monetary parameters configurable via command line

4. **Analysis (`analysis_mitb_game.py`)**:
   - Updated plots to use `ending_balance` instead of deprecated `profit` column
   - Fixed visualization compatibility with new data structure

#### Command Examples:
```bash
# Start with $500 balance, 0 backlog, 100 inventory
python executeMITBeerGame.py --initial_balance 500.0 --initial_backlog 0 --initial_inventory 100

# Custom pricing scenario
python executeMITBeerGame.py --sale_price 6.0 --purchase_cost 3.0 --production_cost 2.0 --initial_balance 2000.0
```

#### Cash Flow Logic:
1. **Each Round**:
   - Revenue: `units_sold × sale_price_per_unit` → added to balance
   - Holding cost: `inventory × holding_cost_per_unit` → deducted from balance  
   - Backlog cost: `backlog × backlog_cost_per_unit` → deducted from balance
   - Purchase cost: `order_quantity × purchase_cost_per_unit` → deducted from balance

2. **Bankruptcy Protection**:
   - If any agent's balance ≤ 0, simulation terminates with "Bankruptcy!" message
   - Prevents unrealistic negative cash flow scenarios

#### Prompt Updates Needed:
- [ ] Update LLM prompts to inform agents about cash constraints
- [ ] Add bankruptcy risk awareness to decision-making prompts
- [ ] Include balance information in strategy generation prompts

---

## Previous Changes...

### Applied Second Fix to Beer Game Logic (2025-06-28)
- Fixed double-counting bug in order flow
- Restored proper order processing: orders placed → upstream backlog → fulfillment
- LLM decisions now for strategic planning only, not immediate order placement
- Prevents both inventory explosion and starvation scenarios

### LangSmith Integration (Previous)
- Comprehensive workflow visualization and tracing
- Real-time performance monitoring  
- Agent-level decision tracking
- Dashboard visualization at smith.langchain.com

### Communication System
- Multi-round agent communication before decisions
- Strategy sharing and collaboration proposals
- Message history tracking and context

### Memory System  
- Agent memory for past decisions and strategies
- Configurable retention periods
- Shared memory pool option for cross-agent learning

### Applied Second Fix to Beer Game Logic (2025-06-28)
- Fixed double-counting bug in order flow
- Restored proper order processing: orders placed → upstream backlog → fulfillment
- LLM decisions now for strategic planning only, not immediate order placement
- Prevents both inventory explosion and starvation scenarios

### LangSmith Integration (Previous)
- Comprehensive workflow visualization and tracing
- Real-time performance monitoring  
- Agent-level decision tracking
- Dashboard visualization at smith.langchain.com

### Communication System
- Multi-round agent communication before decisions
- Strategy sharing and collaboration proposals
- Message history tracking and context

### Memory System  
- Agent memory for past decisions and strategies
- Configurable retention periods
- Shared memory pool option for cross-agent learning 

### 2025-06-30 – Optional Anthropic Claude Provider

* Added `AnthropicLLMClient` in `llm_calls_mitb_game.py` (calls Claude via `/v1/messages` endpoint).
* New CLI flags in `executeMITBeerGame.py`:
  * `--provider litellm|anthropic` (default `litellm`)
  * `--anthropic_model <model-name>` (default `claude-3-haiku-20240307`)
* Requires environment variable `ANTHROPIC_API_KEY`.
* Usage example:

```bash
export ANTHROPIC_API_KEY="sk-..."
python executeMITBeerGame.py --provider anthropic --anthropic_model claude-3-haiku-20240307 --num_rounds 5
``` 

### 2025-06-30 - Chinese Whisper 2.0 SQL Evaluation Run

**Files Used:** `Games/8_Chinese_Whisper2.0/generate_sql.py`, `Games/8_Chinese_Whisper2.0/evaluate_sql.py`

**Change Description:**
- Successfully ran SQL evaluation on latest Chinese Whispers simulation (run_2025-06-30_18-32-24)
- Evaluated 26 SQL queries generated from story transformations across 25 agent steps
- Measured semantic drift by comparing result set sizes against baseline (step 5: 8 rows)
- Generated drift visualization plot showing semantic deviation over agent chain

**Key Results:**
- Several queries failed due to missing 'Olivia Hudson' records (expected for synthetic data)
- Significant drift observed: ranging from 0 (perfect match) to 56 rows difference
- Step 13 showed minimal drift (5 rows difference) while steps 11, 12, 17, 24 showed maximum drift (48-56 rows)
- Generated comprehensive evaluation log, results.json, and diff_vs_agent.png visualization

**Files Generated:**
- `evaluation.log` - Detailed execution log with query results
- `results.json` - Structured evaluation results with row counts and drift metrics  
- `diff_vs_agent.png` - Visualization of semantic drift across agent chain

**Impact:**
- Demonstrates measurable semantic drift in LLM chains through SQL result set analysis
- Provides quantitative metrics for evaluating information preservation in multi-agent transformations 

## Latest Changes (January 2025)

### Round Structure Analysis
**Current Round Types Available:**
- **Main Game Rounds** (`--num_rounds`): Core simulation rounds where agents make supply chain decisions
- **Communication Rounds** (`--communication_rounds`): Sub-rounds within each main round for agent message exchange
- **Memory Rounds** (`--memory_retention_rounds`): Number of past rounds agents retain in memory

**Orchestration Rounds: NOT IMPLEMENTED**
- No orchestration round option currently exists in the system
- Orchestration rounds would be special rounds for higher-level strategy coordination
- Could include supply chain alignment, demand forecasting coordination, risk management planning
- Feature request noted for potential future implementation

### Orchestrator Feature (NEW)
- Added LLM-based Orchestrator that provides recommended order quantities to each agent every round.
- CLI flags:
  * `--enable_orchestrator` – turn on orchestrator phase.
  * `--orchestrator_history` – how many past rounds are included in orchestrator prompt.
  * `--orchestrator_override` – if set, agents are forced to follow orchestrator order.
- Logging:
  * `orchestrator_order` column in CSV/JSON.
  * Recommendations stored in aggregated JSON under `orchestrator_recommendations`.
- Visuals: Combined plot now includes "Orchestrator Recommended Orders" subplot.

### 2025-07-18 - Progress Summary PDF Generator

**Files Created:** `generate_progress_summary.py`, `docs/Progress_Summary_July2025.pdf`, updated `requirements.txt`

**Change Description:**
- Added an automated ReportLab-based script that compiles a three-page PDF summarising the entire repository status.
- Embedded four representative plots (Beer Game, Fishery, Prisoner’s Dilemma) for quick visual inspection.
- Included new dependency `reportlab>=3.6.0` in `requirements.txt` and generated the first PDF at the indicated location.

### 2025-01-XX - Agent Information Access Investigation

**Investigation Focus:** Determined what information agents have access to during communication rounds

**Key Findings:**
- ✅ **CONFIRMED**: Agents have **complete access** to their current balance/status information during communication rounds
- Information includes: current inventory, backlog, total profit/balance, recent orders, last order placed, current strategy, and communication history
- **Implementation**: Information is passed via `generate_communication_message()` method in `models_mitb_game.py`
- **Prompt Structure**: Communication prompts include identical state information as decision-making prompts
- **Code Location**: `prompts_mitb_game.py` lines 200-205 show explicit inclusion of balance (`${profit_accumulated:.2f}`) in communication prompts

**Documentation Added:**
- Added "Agent Information Access During Communication" section to explain what data is available
- Updated "Key Simulation Features" to include "Full State Access" feature
- Confirmed agents can make informed communications based on their actual financial and inventory status

### 2025-01-XX - Enhanced Agent Historical Information Access

**Major Feature Enhancement:** Implemented comprehensive profit and balance history tracking for agents

**Key Changes:**
- ✅ **NEW: Profit History Tracking** - Agents now maintain `profit_history` list of last 10 rounds
- ✅ **NEW: Balance History Tracking** - Agents now maintain `balance_history` list of last 10 rounds  
- ✅ **NEW: Trend Analysis Methods** - Added `get_profit_trend()` and `get_balance_trend()` for strategic analysis
- ✅ **Enhanced Decision Prompts** - All decision-making phases now include comprehensive historical information
- ✅ **Enhanced Communication Prompts** - Communication rounds now include profit/balance history for better coordination
- ✅ **Consistent Information Access** - Historical data passed to ALL phases: decision, communication, strategy updates

**Technical Implementation:**
- **Files Modified:** `models_mitb_game.py`, `prompts_mitb_game.py`, `langraph_workflow.py`, `MIT_Beer_Game.py`
- **New Agent Fields:** `profit_history: List[float]`, `balance_history: List[float]`
- **New Methods:** `update_profit_history()`, `get_profit_trend()`, `get_balance_trend()`
- **Updated Prompt Functions:** Enhanced all prompt generators with historical parameters

**Agent Information Now Includes:**
- Current inventory, backlog, balance
- Last round profit + profit history (last 10 rounds)
- Balance history (last 10 rounds) with trend analysis
- Recent demand/orders history
- Profit trend analysis ("Profitable trend", "Loss trend", "Improving", "Declining")
- Balance trend analysis ("Growing balance", "Declining balance", "Stable balance")

**Impact:** Agents can now make more informed decisions based on their financial trajectory and performance trends, enabling better strategic planning and risk management
