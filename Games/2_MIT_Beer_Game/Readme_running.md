# MIT Beer Game - Comprehensive Development Log & Documentation

## Latest Changes (Most Recent First)

### 2025-01-29: Prompt API Safety, Consistency, and Research Compliance
- **Removed all numeric defaults for holding_cost_per_unit and backlog_cost_per_unit** in AgentContext and prompt methods; these must now be explicitly provided (None triggers error).
- **Preflight validation**: All prompt builders now raise a clear error if any of p/c/h/b are missing.
- **Safe currency formatting**: All `:.2f` interpolations are now guarded; None values display as 'N/A'.
- **Communication checklist updated**: No item prescribes what other agents should order; instead, agents are told to suggest chain-level targets or coordination strategies (ranges OK, no role directives).
- **Renamed profit_accumulated → current_balance** everywhere in prompt code and call sites. Added optional cumulative_profit as a separate field; if present, it is labeled and shown distinctly.
- **Added optional hyper-parameters**: safety_stock_Ss, gamma_hint, delta_hint to AgentContext and all prompt builders. Decision prompt now says: "If not provided, choose γ ∈ [0.25,1.0] and δ ∈ [0, 0.5·μ]; include chosen values in calc."
- **Shipment and logging event order**: Ensured exact phrase match for shipment rule and logging event order across all prompts (system, decision, communication).
- **Removed all references to profit_per_unit_sold** and all legacy numeric economics from dataclasses and prompt text.

**Acceptance checklist (all verified):**
- [x] No 0.5/1.5 defaults exist; p/c/h/b validated before use.
- [x] No formatting with None can occur.
- [x] Communication checklist has no role-specific directives.
- [x] current_balance label and parameter names are consistent; cumulative_profit (if shown) is distinct.
- [x] Decision prompt mentions γ/δ ranges; calc includes the chosen values.
- [x] Shipment + logging lines are text-identical across prompts.
- [x] No user-visible mention of profit_per_unit_sold.

**Files Modified:**
- `Games/2_MIT_Beer_Game/scripts/prompts_mitb_game.py`: All changes above implemented and verified.

---

### 2025-01-27: Prompt Structure Restructure for Research Quality
- **Major restructuring of all prompts in BeerGamePrompts class**: Implemented systematic target format for improved reasoning quality and cross-role consistency
- **New unified structure**: All prompts now follow: ROLE & TASK (1 line) → CORE RULES → STATE SNAPSHOT → CONTEXT BLOCKS → RESEARCH GUIDANCE → OUTPUT CONTRACT
- **Consolidated CORE_RULES constant**: Created shared rules section covering lead time (1 round), shipment constraints, profit objectives, bankruptcy guard, and inventory targets
- **Enhanced context organization**: Moved checklists and additional context into proper CONTEXT BLOCKS before research guidance
- **Unified tone**: Removed emojis and shouty language, adopted concise professional style throughout
- **Fixed inventory contradictions**: Changed from "never zero inventory" to "~1 round of cover; zero inventory allowed if cost-optimal"
- **Enhanced system prompt**: Added CORE RULES section and streamlined JSON-only instruction
- **Updated PromptEngine**: Restructured to insert context blocks (market overview, communications, memory, orchestrator advice) before research guidance
- **Research guidance integration**: Added behavioral cue encoding instructions (e.g., [aggressive-ordering], [coordination-seeking])
- **Contract placement**: Made output contract the final section with no text following it
- **Word count reduction**: Achieved ~20% reduction in prompt length while maintaining all information
- **Cross-prompt consistency**: Ensured all cost parameters, rules, and structures match across decision, communication, and strategy prompts

**Files Modified:**
- `Games/2_MIT_Beer_Game/scripts/prompts_mitb_game.py`: Complete restructure of all prompt methods and PromptEngine
- Added _CORE_RULES constant with unified rules
- Updated get_strategy_generation_prompt, get_strategy_update_prompt, get_decision_prompt, get_communication_prompt
- Restructured get_system_prompt with CORE RULES section
- Enhanced PromptEngine.build_prompt to properly handle context blocks

**Research Impact:**
- Improved LLM reasoning quality through clearer structure
- Enhanced cross-role consistency via shared CORE RULES
- Better research suitability with behavioral cue encoding
- Reduced cognitive load with shorter, more focused prompts
- Maintained all existing APIs and JSON schemas

### 2025-01-22: Real-Time LLM Logging System
- **Real-time LLM call logging**: Every LLM call (decision-making, communication, initialization) is now logged immediately after it happens, providing live visibility into agent reasoning.
- **Immediate log writing**: Each LLM interaction is written to `human_readable_log.txt` as soon as it completes, with automatic file flushing for real-time monitoring.
- **Structured logging format**: Each LLM call shows:
  - 🤖 Call type and agent name with optional round number
  - 🔧 Complete system prompt
  - 👤 Complete user prompt  
  - 🎯 Complete model output (JSON formatted)
  - ─ Visual separators for easy reading
- **Phase organization**: Clear headers mark different simulation phases:
  - 🚀 Agent initialization phase
  - 🎲 Round start headers with external demand
  - 💬 Communication phase headers
  - 🎯 Decision phase headers
  - 📊 Round summary with agent states
- **Live debugging capability**: Developers can monitor LLM reasoning in real-time during simulation execution.
- **Simplified round summaries**: Round-end logging now shows concise agent state summaries instead of duplicating detailed LLM calls.
- **Complete traceability**: Every prompt and response is immediately available for analysis without waiting for simulation completion.
- **Fixed Pydantic validation error**: Corrected type annotation for `human_log_file` field to use `Optional[Any]` instead of `TextIO` for proper Pydantic compatibility.

---

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
   - Modified `create_agent()`

---

# MIT Beer Game - Running Log

## 2025-01-28: Naming & Schema Cleanup (Point 7)

**Feature**: Fixed JSON schema naming inconsistencies by removing leading spaces from keys and ensuring proper naming conventions.

**Problem Identified**:
- **Malformed JSON Key**: Strategy generation schema contained `" d_demand_next_round"` with leading space
- **Schema Inconsistency**: Different schemas used different naming patterns for demand forecasting
- **JSON Validation Issues**: Leading spaces in keys could cause parsing problems

**Key Implementation**:

1. **✅ Fixed Leading Space in JSON Key**:
   - **Before**: `" d_demand_next_round": <integer>,` (with leading space)
   - **After**: `"expected_demand_next_round": <integer>,` (clean, consistent naming)
   - **Location**: Strategy generation prompt JSON schema
   - **Impact**: Prevents JSON parsing issues and maintains naming consistency

2. **✅ Verified Schema Consistency**:
   - **Expected Demand Key**: All schemas now use `"expected_demand_next_round"` consistently
   - **No Leading Spaces**: Comprehensive check confirmed no other keys have leading spaces
   - **Clean JSON**: All JSON schemas follow proper formatting conventions
   - **Validation Ready**: Schemas are now properly formatted for JSON validators

3. **✅ Maintained JSON-Only Response Rule**:
   - **No Code Fences**: Confirmed no markdown or triple backticks in JSON responses
   - **Short Responses**: Maintained concise JSON-only output requirement
   - **Clean Format**: All JSON schemas follow the established clean format rules

**Files Modified**:
- `Games/2_MIT_Beer_Game/scripts/prompts_mitb_game.py`: Fixed leading space in strategy generation JSON schema

**Acceptance Checks**:
- ✅ **No key with leading space exists**: Comprehensive regex search found 0 keys with leading spaces
- ✅ **"expected_demand_next_round" present**: Key appears 3 times across different schemas consistently
- ✅ **JSON-only response maintained**: No markdown formatting or code fences in schemas

**Result**: All JSON schemas now have clean, consistent key naming without formatting issues. The `expected_demand_next_round` key is properly formatted across all prompt schemas.

---

## 2025-01-28: Logging / Analysis Alignment (Point 9)

**Feature**: Implemented logging and analysis alignment by specifying the correct event order in prompt text and ensuring no placeholder logging lines remain.

**Problem Identified**:
- **Missing Event Order**: Prompts did not specify the sequence of events that occur in each round
- **Unclear Metrics Source**: No explicit statement that metrics are computed from logged events
- **Potential Placeholder Text**: Risk of placeholder logging lines appearing in prompts or sample text

**Key Implementation**:

1. **✅ Added Event Order Specification**:
   - **Exact Sequence**: "orders placed → production → shipments sent/received → sales → costs → ending inventory/backlog"
   - **Location**: Added to system prompt in the LOGGING & ANALYSIS section
   - **Clarity**: Provides agents with clear understanding of round event sequence
   - **Consistency**: Ensures all agents understand the same event ordering

2. **✅ Specified Metrics Computation**:
   - **Source Clarity**: "All performance metrics are computed from these logged events (no placeholders)"
   - **No Placeholders**: Explicitly states that metrics come from real logged data
   - **Analysis Foundation**: Establishes that all analysis is based on actual event logs
   - **Data Integrity**: Ensures agents understand metrics are not synthetic or placeholder values

3. **✅ Verified No Placeholder Lines**:
   - **Comprehensive Check**: Searched for "Total LLM calls … tokens 0 cost $0" patterns
   - **Clean Prompts**: Confirmed no placeholder logging text in any prompt methods
   - **Real Data Only**: All logging references point to actual computed metrics
   - **Professional Output**: Maintains clean, production-ready prompt text

**Files Modified**:
- `Games/2_MIT_Beer_Game/scripts/prompts_mitb_game.py`: Added logging & analysis guidance to system prompt

**Specific Addition**:
```
**LOGGING & ANALYSIS**: Each round follows this event order: orders placed → production → 
shipments sent/received → sales → costs → ending inventory/backlog. All performance metrics 
are computed from these logged events (no placeholders).
```

**Event Order Details**:
1. **Orders Placed**: Agents make ordering decisions
2. **Production**: Factory schedules and executes production
3. **Shipments Sent/Received**: Goods move through supply chain with lead time
4. **Sales**: Downstream demand is fulfilled from available inventory
5. **Costs**: Holding costs and backlog costs are calculated
6. **Ending Inventory/Backlog**: Final state is recorded for next round

**Acceptance Checks**:
- ✅ **Exact event order appears**: "orders placed → production → shipments sent/received → sales → costs → ending inventory/backlog" present in system prompt
- ✅ **No placeholder logging lines**: Comprehensive search found no "Total LLM calls … tokens 0 cost $0" patterns in prompts
- ✅ **Metrics computed from logs**: Explicit statement that metrics come from logged events, not placeholders
- ✅ **All prompt methods working**: System, decision, and communication prompts all function correctly with new guidance

**Testing Results**:
```python
# ✅ Event order verification
system_prompt = BeerGamePrompts.get_system_prompt('Retailer')
assert 'orders placed → production → shipments sent/received → sales → costs → ending inventory/backlog' in system_prompt

# ✅ Metrics computation verification  
assert 'metrics are computed from these logged events' in system_prompt
assert 'no placeholders' in system_prompt

# ✅ No placeholder patterns found
# Verified no "tokens 0 cost $0" or similar placeholder text exists
```

**Result**: Agents now have clear understanding of the round event sequence and know that all performance metrics are computed from real logged events, not placeholder values. This provides proper foundation for analysis and decision-making based on actual simulation data.

---

## 2025-01-28: Long-Term Mode Toggle Consistency (Point 8)

**Feature**: Enhanced long-term planning mode with clear role viability priority guidance, ensuring agents prioritize survival when backlog is high while maintaining collaborative objectives.

**Problem Identified**:
- **Unclear Priority Hierarchy**: No explicit guidance on when role-specific objectives should override consensus targets
- **Survival vs Collaboration**: Agents needed clearer direction on balancing individual viability with chain-wide cooperation
- **Critical Situation Handling**: Missing guidance for high-backlog or financial crisis situations

**Key Implementation**:

1. **✅ Added Role Viability Priority Clause**:
   - **New Guidance**: "When your backlog is high or financial situation is critical, role-specific objectives dominate consensus targets - your survival enables future collaboration."
   - **Clear Hierarchy**: Establishes that individual survival takes priority when necessary
   - **Justification**: Explains that survival enables future collaboration (not selfish abandonment)
   - **Context-Sensitive**: Applies specifically when backlog or financial situation is critical

2. **✅ Maintained Collaborative Framework**:
   - **Existing Structure**: Preserved all existing long-term planning collaborative guidance
   - **No Deleted Text**: Did not reintroduce any previously removed coaching language
   - **Balanced Approach**: Maintains focus on chain-wide optimization while adding survival priority
   - **Strategic Context**: Keeps emphasis on mutual interdependence and information sharing

3. **✅ Enhanced Decision Logic**:
   - **Situational Awareness**: Agents now understand when to shift from consensus to survival mode
   - **Backlog Trigger**: High backlog explicitly mentioned as a trigger for priority shift
   - **Financial Trigger**: Critical financial situation also triggers role-specific focus
   - **Return Path**: Implies return to collaboration once crisis is resolved

**Files Modified**:
- `Games/2_MIT_Beer_Game/scripts/prompts_mitb_game.py`: Added role viability priority clause to collaborative long-term planning guidance

**Specific Addition**:
```
**Role Viability Priority**: When your backlog is high or financial situation is critical, 
role-specific objectives dominate consensus targets - your survival enables future collaboration.
```

**Acceptance Checks**:
- ✅ **Decision prompt includes role-specific dominance one-liner**: Clear statement added about priority hierarchy
- ✅ **No removed text reintroduced**: Verified no forbidden phrases or deleted coaching language returned
- ✅ **Backlog consideration present**: High backlog explicitly mentioned as trigger condition
- ✅ **Collaborative framework maintained**: All existing long-term planning guidance preserved

**Result**: Long-term planning mode now provides clear guidance on when agents should prioritize individual survival over consensus targets, ensuring both collaborative optimization and individual viability in critical situations.

---

## 2025-01-28: Round Index Alignment (Point 6)

**Feature**: Implemented round index alignment to display the correct round number in decision prompts, ensuring agents know which round they are making decisions for.

**Problem Identified**:
- **Missing Round Context**: Decision prompts did not show which round agents were making decisions for
- **Generic Opening**: All prompts showed "MIT Beer Game" instead of specific round information
- **Potential Confusion**: Agents could not distinguish between different rounds in their decision-making context

**Key Implementation**:

1. **✅ Added round_index Parameter to Decision Prompts**:
   - **`get_decision_prompt()`**: Added `round_index: int = None` parameter
   - **`get_decision_prompt_with_communication()`**: Added `round_index` parameter and pass-through
   - **`get_decision_prompt_with_memory()`**: Added `round_index` parameter and pass-through
   - **Dynamic Display**: Shows "Round {round_index}" when provided, falls back to "MIT Beer Game" when None

2. **✅ Updated Opening Line Logic**:
   - **Conditional Text**: `round_text = f"Round {round_index}" if round_index is not None else "MIT Beer Game"`
   - **Clear Context**: Agents now see "You are the Retailer in Round 5" instead of generic text
   - **Fallback Handling**: Gracefully handles None values for backward compatibility

3. **✅ Enhanced PromptEngine Integration**:
   - **PromptEngine.build_prompt()**: Already had `round_index` parameter, now passes it to `get_decision_prompt()`
   - **Real Round Index**: Ensures the actual current round number is passed through the prompt chain
   - **Phase-Specific**: Only affects decision prompts, communication prompts retain their existing round display

4. **✅ Updated Method Signatures Throughout**:
   - **`decide_order_quantity()`**: Added `round_index=None` parameter
   - **`decide_order_quantity_with_communication()`**: Added `round_index=None` parameter
   - **MIT_Beer_Game.py**: Updated calls to pass current `round_index` variable
   - **Consistent Flow**: Round index flows from game loop through all decision prompt layers

**Files Modified**:
- `Games/2_MIT_Beer_Game/scripts/prompts_mitb_game.py`:
  - Added `round_index` parameter to all decision prompt methods
  - Updated opening line logic with conditional round text
  - Updated internal calls to pass `round_index` through
- `Games/2_MIT_Beer_Game/scripts/models_mitb_game.py`:
  - Added `round_index` parameter to decision methods
  - Updated calls to pass `round_index` to prompt methods
- `Games/2_MIT_Beer_Game/scripts/MIT_Beer_Game.py`:
  - Updated `decide_order_quantity()` calls to pass current `round_index`

**Specific Changes**:

1. **Opening Line Enhancement**:
   ```python
   # Before (generic)
   return f"""
   You are the {role_name} in the MIT Beer Game. {role_context}
   
   # After (round-specific)
   round_text = f"Round {round_index}" if round_index is not None else "MIT Beer Game"
   return f"""
   You are the {role_name} in the {round_text}. {role_context}
   ```

2. **Parameter Flow**:
   ```python
   # MIT_Beer_Game.py → models_mitb_game.py → prompts_mitb_game.py
   agent.decide_order_quantity(..., round_index=round_index, ...)
   → self.prompts.get_decision_prompt(..., round_index=round_index, ...)
   → "You are the Retailer in Round 5"
   ```

3. **PromptEngine Integration**:
   ```python
   # PromptEngine.build_prompt() now passes round_index
   base_prompt = BeerGamePrompts.get_decision_prompt(
       ...,
       round_index=round_index,  # ← Added this line
       longtermplanning_boolean=longtermplanning_boolean,
   )
   ```

**Acceptance Checks**:
- ✅ **Decision prompts show correct round number**: "Round 5", "Round 10" appear correctly
- ✅ **No "Round 0" for later rounds**: Round 10 prompt shows "Round 10", not "Round 0"
- ✅ **PromptEngine passes real round index**: build_prompt(phase="decision") uses actual round number
- ✅ **Backward compatibility**: None values gracefully fall back to "MIT Beer Game"

**Testing Results**:
```python
# Round 5 test
prompt = get_decision_prompt(..., round_index=5)
# ✅ Contains: "You are the Retailer in Round 5"
# ✅ Does NOT contain: "MIT Beer Game"

# Round 10 test  
prompt = get_decision_prompt(..., round_index=10)
# ✅ Contains: "Round 10"
# ✅ Does NOT contain: "Round 0"

# None test
prompt = get_decision_prompt(..., round_index=None)
# ✅ Contains: "MIT Beer Game" (fallback)
```

**Result**: Decision prompts now clearly display the current round number, providing agents with proper temporal context for their decision-making. Agents know exactly which round they are operating in, improving decision quality and debugging capabilities.

---

## 2025-01-28: Parameter Order & Keyword-Args Safety (Point 5)

**Feature**: Implemented parameter order and keyword-args safety to prevent field misalignment and ensure unit economics are explicitly passed in all prompt method calls.

**Problem Identified**:
- **Positional Argument Risk**: Helper functions calling `get_decision_prompt()` with positional args could cause field shifting (e.g., `last_order_placed` being treated as `holding_cost_per_unit`)
- **Missing Unit Economics**: Some calls to prompt methods were missing explicit unit economics parameters (`p`, `c`, `h`, `b`)
- **Parameter Misalignment**: Long positional argument lists prone to errors when method signatures change

**Key Implementation**:

1. **✅ Converted All Internal Calls to Keyword Arguments**:
   - **`get_decision_prompt()`**: All internal calls now use explicit keyword arguments
   - **`get_decision_prompt_with_communication()`**: Converted positional calls to keyword args
   - **`get_decision_prompt_with_memory()`**: Converted positional calls to keyword args
   - **`get_communication_prompt()`**: All internal calls use keyword arguments
   - **`get_communication_prompt_with_memory()`**: Converted positional calls to keyword args

2. **✅ Explicit Unit Economics in All Calls**:
   - **Required Parameters**: `selling_price_per_unit`, `unit_cost_per_unit`, `holding_cost_per_unit`, `backlog_cost_per_unit`
   - **All Prompt Methods**: Every call now explicitly passes unit economics parameters
   - **No Missing Parameters**: Eliminated cases where economics parameters were undefined or missing
   - **Consistent Parameter Flow**: Unit economics flow from execute script through all prompt layers

3. **✅ Enhanced Method Signatures**:
   - **`decide_order_quantity_with_communication()`**: Added missing unit economics parameters
   - **Parameter Defaults**: Proper None defaults for optional economics parameters
   - **Backward Compatibility**: Maintained `profit_per_unit_sold` parameter for compatibility
   - **Type Safety**: All parameters properly typed and documented

4. **✅ Safety Improvements**:
   - **No Positional Lists**: Eliminated long positional argument lists prone to misalignment
   - **Explicit Naming**: Every parameter explicitly named in method calls
   - **Field Alignment**: Prevents parameter shifting when method signatures change
   - **Clear Intent**: Code is more readable and maintainable

**Files Modified**:
- `Games/2_MIT_Beer_Game/scripts/prompts_mitb_game.py`: 
  - Converted all internal `get_decision_prompt()` calls to keyword arguments
  - Converted all internal `get_communication_prompt()` calls to keyword arguments
- `Games/2_MIT_Beer_Game/scripts/models_mitb_game.py`:
  - Updated `decide_order_quantity_with_communication()` method signature with unit economics parameters
  - Updated method call to pass unit economics explicitly

**Specific Changes**:

1. **`get_decision_prompt_with_communication()`**:
   ```python
   # Before (positional - risky)
   base_prompt = BeerGamePrompts.get_decision_prompt(
       role_name, inventory, backlog, recent_demand_or_orders, incoming_shipments,
       current_strategy, selling_price_per_unit, unit_cost_per_unit, ...
   )
   
   # After (keyword - safe)
   base_prompt = BeerGamePrompts.get_decision_prompt(
       role_name=role_name,
       inventory=inventory,
       backlog=backlog,
       recent_demand_or_orders=recent_demand_or_orders,
       ...
   )
   ```

2. **`decide_order_quantity_with_communication()`**:
   ```python
   # Before (missing economics)
   async def decide_order_quantity_with_communication(self, temperature: float = 0.7, 
                                                    profit_per_unit_sold: float = 2.5,
                                                    recent_communications: List[Dict] = None)
   
   # After (complete economics)
   async def decide_order_quantity_with_communication(self, temperature: float = 0.7, 
                                                    selling_price_per_unit: float = None,
                                                    unit_cost_per_unit: float = None,
                                                    holding_cost_per_unit: float = None,
                                                    backlog_cost_per_unit: float = None,
                                                    ...)
   ```

**Acceptance Checks**:
- ✅ **No long positional argument lists**: All prompt method calls use keyword arguments
- ✅ **Unit economics present in every call**: All calls explicitly pass `p`, `c`, `h`, `b` parameters
- ✅ **Field alignment safety**: No risk of parameter misalignment due to positional arguments
- ✅ **Method signature consistency**: All prompt methods have consistent parameter handling

**Methods Updated**:
```
✅ get_decision_prompt() - all calls use keywords
✅ get_decision_prompt_with_communication() - all calls use keywords  
✅ get_decision_prompt_with_memory() - all calls use keywords
✅ get_communication_prompt() - all calls use keywords
✅ get_communication_prompt_with_memory() - all calls use keywords
```

**Result**: All prompt method calls are now safe from parameter misalignment, with explicit unit economics parameters passed consistently throughout the system. Code is more maintainable and less prone to subtle bugs from parameter shifting.

---

## 2025-01-28: Communication Prompts: Factual, Role-Bounded (Point 4)

**Feature**: Implemented factual, role-bounded communication guidelines to eliminate emotive language, prevent prescriptive directives to other roles, and ensure proper metric labeling.

**Key Implementation**:

1. **✅ Factual Communication Guidelines**:
   - **Share Only Specific Data**: inventory, backlog, recent mean demand μ, next order plan, γ (backlog-clearance rate), and material risks
   - **No Prescriptive Quantities**: Agents cannot tell other roles exact quantities to order
   - **Ranges/Targets Acceptable**: Coordination through ranges and targets is permitted
   - **Operational Focus**: Share factual information about situation only
   - **No Emotive Language**: Removed urgent directives, emotional appeals, and charged language

2. **✅ Fixed Mislabeled Metrics**:
   - **Replaced "Total profit so far"**: Now correctly labeled as "Current balance" when referring to bank balance
   - **Clear Distinction**: Bank balance vs. cumulative profit are properly differentiated
   - **Accurate Labeling**: All financial metrics now have precise, non-misleading labels
   - **Consistent Terminology**: Balance and profit terms used correctly throughout

3. **✅ Shipment Rule Consistency**:
   - **Standardized Wording**: "You can ship at most min(inventory_available, downstream_order + your_backlog). Shipment is executed by the simulator; you do not decide shipments in your reply."
   - **Consistent Across Prompts**: Same exact wording in both decision and communication prompts
   - **Clear Simulator Role**: Explicitly states that simulator handles shipment execution

4. **✅ Communication Restrictions**:
   - **No Exact Quantity Prescriptions**: Agents cannot tell others "order 15 units" or similar
   - **No Emotive Language**: Removed words like "critical," "urgent," "collapse," "must"
   - **No Directives**: Agents cannot command other roles' specific actions
   - **Factual Information Only**: Focus on operational facts, not emotional appeals
   - **Role-Bounded**: Each agent shares only their own situation and constraints

**Specific Changes**:

- **Communication Guidelines**: Replaced emotional objectives with factual sharing requirements
- **Metric Correction**: "Total profit so far" → "Current balance" when referring to bank balance
- **Language Cleanup**: Removed emotive language and urgent directives
- **Shipment Rule**: Updated to match decision prompt wording exactly
- **Restriction Clarity**: Explicit rules about what agents can and cannot communicate

**Files Modified**:
- `Games/2_MIT_Beer_Game/scripts/prompts_mitb_game.py`: Communication prompt updated with factual guidelines and proper metric labeling

**Acceptance Checks**:
- ✅ **No emotive language or directives**: Removed urgent language and commands to other roles
- ✅ **No "profit so far" for balance**: Correctly labeled as "Current balance" 
- ✅ **Shipment rule consistency**: Exact same wording as decision prompts
- ✅ **Share only specified data**: inventory, backlog, μ, order plan, γ, material risks
- ✅ **Ranges/targets acceptable**: Coordination allowed through ranges, not exact quantities

**Communication Allowed**:
```
📊 INVENTORY STATUS: Current inventory and backlog
📈 DEMAND PATTERN: Recent mean demand μ observations  
📋 ORDER PLAN: Next order plan (ranges/targets, not exact quantities for others)
⚙️ BACKLOG STRATEGY: γ (backlog-clearance rate) approach
⚠️ MATERIAL RISKS: Specific operational risks observed
```

**Communication Restrictions**:
```
❌ Do NOT prescribe exact quantities to other roles
❌ Do NOT use emotive language or urgent directives  
❌ Do NOT command other roles' specific actions
✅ Share factual information about your situation only
✅ Ranges and targets acceptable for coordination
✅ Focus on operational facts, not emotional appeals
```

**Result**: Communication is now strictly factual, role-bounded, and free from emotive language or prescriptive directives. Agents share operational data without commanding others' actions.

---

## 2025-01-28: Mandatory Controller & Self-Audit System (Point 3)

**Feature**: Implemented mandatory controller and self-audit system in decision prompts to enforce structured decision-making with required calculations and validation checks.

**Key Implementation**:

1. **✅ Mandatory Controller Framework**:
   - **μ = EWMA of recent demand**: Exponentially weighted moving average for demand forecasting
   - **IP = on_hand + in_transit − backlog**: Inventory position calculation
   - **S* = μ*(lead_time+1) + S_s**: Target stock level with safety stock
   - **BacklogClear = γ * backlog (γ ∈ [0,1])**: Backlog clearing strategy with tunable parameter
   - **O_raw = max(0, S* − IP + BacklogClear)**: Raw order calculation before adjustments
   - **Final Order**: Smoothed within ±δ around μ, with solvency cap ensuring (balance − c*O_final) > 0

2. **✅ Self-Audit Boolean Checks**:
   - **coverage_ok**: Validates that if on_hand=0 then order ≥ expected_demand
   - **includes_backlog**: Confirms backlog consideration in order calculation  
   - **solvency_ok**: Ensures order doesn't cause bankruptcy

3. **✅ Extended JSON Schema**:
   - **Required calc object** with all controller parameters and audit results
   - **Fields**: mu, S_star, IP, gamma, delta, O_raw, O_final, solvency_ok, coverage_ok, includes_backlog
   - **Validation**: Forces agents to show their calculation work and self-audit

4. **🎯 Decision Prompts Only**:
   - **Scope**: Applied only to decision prompts (not communication or strategy prompts)
   - **Consistency**: All decision prompt variants include the controller requirements
   - **Integration**: Works with existing economics parameters and rules

**Retailer Special Case**:
- **Scenario**: With backlog >> expected_demand and on_hand=0
- **Enforcement**: Instructions clearly force orders to cover expected_demand via coverage_ok
- **Backlog Handling**: Address backlog proportionally via γ parameter in BacklogClear formula
- **Safety**: Prevents both stockouts (coverage_ok) and excessive ordering (solvency_ok)

**Files Modified**:
- `Games/2_MIT_Beer_Game/scripts/prompts_mitb_game.py`: Decision prompt updated with controller and calc object requirement

**Acceptance Checks**:
- ✅ **Decision prompts explicitly require calc object**: `"calc": {` appears in JSON schema
- ✅ **Self-audit booleans in schema**: coverage_ok, includes_backlog, solvency_ok all appear in decision prompt
- ✅ **Retailer edge case handled**: With backlog >> expected_demand and on_hand=0, instructions force coverage via γ parameter

**JSON Schema Extension**:
```json
{
  "order_quantity": <integer>,
  "calc": {
    "mu": <float>,
    "S_star": <float>, 
    "IP": <float>,
    "gamma": <float>,
    "delta": <float>,
    "O_raw": <float>,
    "O_final": <integer>,
    "solvency_ok": <boolean>,
    "coverage_ok": <boolean>,
    "includes_backlog": <boolean>
  }
}
```

**Result**: All decision-making now follows structured controller logic with mandatory self-audit checks, ensuring consistent and validated ordering decisions across all agents.

---

## 2025-01-28: Shipment & Inventory Rules Standardization (Point 2)

**Feature**: Standardized shipment and inventory rules text across all prompts with clear, consistent language.

**Key Changes**:

1. **✅ New Standardized Shipment Rule**:
   - **Old**: Various inconsistent descriptions about shipping constraints
   - **New**: "You can ship at most min(inventory_available, downstream_order + your_backlog). Shipment is executed by the simulator; you do not decide shipments in your reply."
   - **Clarity**: Makes it clear that shipment is handled by the simulator, not the agent

2. **✅ New Standardized Inventory Policy**:
   - **Old**: "Never let the inventory go to zero" and similar restrictive language
   - **New**: "Target a safety stock S_s. It's acceptable to be at zero if demand is met and backlog is low."
   - **Flexibility**: Allows agents to have zero inventory when appropriate, removing overly restrictive constraints

3. **🔄 Universal Application**:
   - **Strategy Generation Prompt**: Updated with new rules
   - **Strategy Update Prompt**: Updated with new rules  
   - **Decision Prompt**: Updated with new rules
   - **Communication Prompt**: Updated with new rules
   - **System Prompt**: Updated with new rules

4. **🚫 Removed Problematic Language**:
   - ❌ "equal to (their_order + your_backlog)" - completely eliminated
   - ❌ "Never let the inventory go to zero" - completely eliminated
   - ❌ Inconsistent shipment constraint descriptions - standardized

**Files Modified**:
- `Games/2_MIT_Beer_Game/scripts/prompts_mitb_game.py`: All prompt methods updated with standardized rules

**Acceptance Checks**:
- ✅ **No forbidden language**: `grep` confirms no prompts contain "equal to (their_order + your_backlog)"
- ✅ **No restrictive inventory rules**: `grep` confirms no prompts contain "Never let the inventory go to zero"
- ✅ **Consistent application**: New shipment and inventory rules appear in all 5 prompt locations
- ✅ **Clear simulator responsibility**: All rules clarify that shipment is executed by simulator

**Impact**:
- **Consistent Messaging**: All agents receive identical rule descriptions regardless of prompt type
- **Realistic Flexibility**: Agents can now make realistic inventory decisions including zero levels when appropriate
- **Clear Responsibilities**: Distinction between agent decisions (ordering) and simulator actions (shipping) is explicit
- **Better Understanding**: Agents have clearer understanding of shipment constraints using min() function notation

**Result**: All prompts now use standardized, clear, and realistic shipment and inventory rules that eliminate confusion and overly restrictive constraints.

---

## 2025-01-28: Economics Single Source of Truth (Point 1)

**Feature**: Implemented economics parameters as single source of truth from execute script, removing all hardcoded values from prompt code.

**Key Changes**:

1. **✅ New Economics Parameters System**:
   - **selling_price_per_unit (p)**: Revenue per unit sold
   - **unit_cost_per_unit (c)**: Cost per unit ordered/produced (purchase_cost for most agents, production_cost for Factory)
   - **holding_cost_per_unit (h)**: Cost per unit held in inventory per round
   - **backlog_cost_per_unit (b)**: Cost per unit of unfulfilled demand per round

2. **📊 Explicit Profit Formula Added**:
   - **Formula**: `profit_t = p*sales_t - c*orders_t - h*inventory_end_t - b*backlog_end_t`
   - **Display**: Shows both symbolic and numeric versions in all prompts
   - **Location**: Appears in system prompt and all decision/communication prompts

3. **🔄 Parameter Flow**:
   - **Source**: `executeMITBeerGame.py` command-line arguments (`--sale_price`, `--purchase_cost`, `--production_cost`, `--holding_cost_per_unit`, `--backlog_cost_per_unit`)
   - **Flow**: Execute script → MIT_Beer_Game.py → models_mitb_game.py → prompts_mitb_game.py
   - **Factory Special**: Uses `production_cost_per_unit` instead of `purchase_cost_per_unit` for unit cost

4. **🚫 Deprecated profit_per_unit_sold**:
   - **Status**: Kept for backward compatibility but no longer used in calculations or text
   - **Replacement**: Now calculated dynamically as (selling_price - unit_cost) where needed
   - **Prompts**: All prompts now use p, c, h, b parameters instead of hardcoded values

**Files Modified**:
- `Games/2_MIT_Beer_Game/scripts/prompts_mitb_game.py`: All prompt methods updated to accept and use p,c,h,b parameters
- `Games/2_MIT_Beer_Game/scripts/models_mitb_game.py`: Agent methods updated to pass new parameters
- `Games/2_MIT_Beer_Game/scripts/MIT_Beer_Game.py`: Simulation updated to pass economics parameters from execute script

**Acceptance Checks**:
- ✅ **No hardcoded values**: `grep` confirms no "holding cost of 0.5", "backlog cost of 1.5", or "profit per unit sold 2.5" remain
- ✅ **Runtime parameters**: Decision/strategy/communication prompts now include {p, c, h, b} values from execute script
- ✅ **Profit formula**: Formula string appears in system and decision prompts with both symbolic and numeric values

**Usage Example**:
```bash
# Custom economics parameters
python executeMITBeerGame.py --sale_price 6.0 --purchase_cost 3.0 --holding_cost_per_unit 0.2 --backlog_cost_per_unit 2.0
```

**Result**: All economics are now single source of truth from execute script, eliminating hardcoded values and enabling flexible economic scenarios.

---

## 2025-01-28: Global Constraints Fixes (Point 0)

**Feature**: Implemented critical prompt fixes to comply with global constraints and remove problematic language.

**Key Fixes Applied**:

1. **❌ Removed "Never let the inventory go to zero"**: 
   - Eliminated all instances of this forbidden constraint from strategy generation, strategy update, and decision prompts
   - This rule was overly restrictive and conflicted with realistic supply chain management

2. **🔄 Fixed Shipment Constraint Language**:
   - **Before**: "You can only ship to downstream partners up to (their_order + your_backlog)"
   - **After**: "The simulator will ship at most the amount requested by downstream partners plus your current backlog. The simulator handles all shipment calculations automatically."
   - Clarified that the simulator handles shipment calculations, not the agent

3. **🚫 Removed Emotionally Charged Language**:
   - Replaced "supply chain collapse" with "supply chain instability" or "supply chain failure"
   - Removed "PREVENT COLLAPSE" objective, replaced with "RISK AWARENESS"
   - Changed "Warning signs of potential market collapse" to "Warning signs of potential market instability"
   - Updated "Emphasize that supply chain collapse hurts EVERYONE's profits" to use "instability"

4. **📢 Fixed Prescriptive Communication Language**:
   - **Before**: "Propose a concrete order plan (e.g. 'I will order 8')"
   - **After**: "Share your general ordering approach and reasoning"
   - Removed specific quantity examples that were too prescriptive
   - Made communication guidelines more factual and role-bounded

**Files Modified**:
- `Games/2_MIT_Beer_Game/scripts/prompts_mitb_game.py`: All prompt methods updated to comply with constraints

**Impact**: 
- Prompts are now more realistic and less prescriptive
- Agents have more flexibility in inventory management decisions
- Communication remains factual without emotional manipulation
- Shipment rules are clearer and correctly attribute responsibility to the simulator

**Compliance**: ✅ All global constraints from point 0 now satisfied

---

## 2025-01-28: Implemented Long-Term Planning vs Selfish Mode Parameter

**Feature**: Added `longtermplanning_boolean` parameter to enable sophisticated strategic behavior switching between individual profit maximization and collaborative supply chain optimization.

**Key Implementation**:
- **Command-Line Parameter**: `--longtermplanning_boolean` flag in `executeMITBeerGame.py` (defaults to False - selfish mode)
- **Behavioral Modes**:
  - **Selfish Mode (Default)**: Agents focus on individual profit maximization, competitive advantage, strategic information control, and opportunistic behavior
  - **Collaborative Mode**: Agents optimize for total supply chain profitability with emphasis on mutual interdependence, information sharing, coordinated buffer management, and risk mitigation

**Sophisticated Prompt Engineering**:
- **Individual Profit Maximization Mode**: Encourages self-interest priority, competitive advantage seeking, strategic information control, opportunistic behavior, resource optimization focused on individual costs, and short-term gains prioritization
- **Collaborative Long-Term Optimization Mode**: Promotes collective success strategy, mutual interdependence recognition, transparent information sharing, coordinated buffer management, long-term sustainability focus, and risk mitigation through strategic coordination

**Technical Changes**:
- Added `_get_objective_guidance()` method in `BeerGamePrompts` class that returns mode-specific strategic guidance
- Updated both `get_decision_prompt()` and `get_communication_prompt()` methods to include objective guidance
- Enhanced `PromptEngine.build_prompt()` to pass the parameter through to prompt generation
- Modified agent methods (`llm_decision`, `decide_order_quantity`, `generate_communication_message`) to accept and forward the parameter
- Updated simulation functions (`run_beer_game_generation`, `run_beer_game_simulation`) to pass parameter through the entire pipeline

**Files Modified**:
- `Games/2_MIT_Beer_Game/scripts/executeMITBeerGame.py`: Added command-line argument
- `Games/2_MIT_Beer_Game/scripts/prompts_mitb_game.py`: Added objective guidance system and updated prompt methods
- `Games/2_MIT_Beer_Game/scripts/models_mitb_game.py`: Updated agent methods to handle parameter
- `Games/2_MIT_Beer_Game/scripts/MIT_Beer_Game.py`: Updated simulation functions to pass parameter

**Usage Examples**:
```bash
# Selfish mode (default)
python executeMITBeerGame.py --num_rounds 10 --communication_rounds 2

# Collaborative long-term planning mode
python executeMITBeerGame.py --num_rounds 10 --communication_rounds 2 --longtermplanning_boolean
```

**Result**: Agents now exhibit fundamentally different strategic behaviors based on the mode, enabling research into cooperative vs competitive supply chain dynamics.

---

## 2025-01-28: Enhanced Parameter Display in Combined Plots

**Enhancement**: Updated combined plots to display all simulation parameters as a subtitle at the bottom instead of in a small text box in the corner.

**Key Improvements**:
- **Comprehensive Parameter Display**: Now shows all command-line parameters including `initial_inventory`, `initial_balance`, `temperature`, cost parameters, communication settings, orchestrator settings, etc.
- **Better Visibility**: Parameters displayed as subtitle at bottom of plots instead of small corner text box
- **Automatic Line Wrapping**: Long parameter lists automatically split into multiple lines for readability
- **Professional Format**: Parameters formatted as `key=value` pairs separated by `|` for clean presentation

**Technical Changes**:
- Added missing parameters (`initial_inventory`, `initial_backlog`, `initial_balance`) to `run_beer_game_generation()` function signature
- Updated `run_settings` dictionary to include all simulation parameters
- Modified `analysis_mitb_game.py` to display parameters as figure subtitle using `fig.suptitle()`
- Added automatic text wrapping for long parameter lists

**Files Modified**:
- `Games/2_MIT_Beer_Game/scripts/MIT_Beer_Game.py`: Enhanced `run_settings` dictionary and function signatures
- `Games/2_MIT_Beer_Game/scripts/analysis_mitb_game.py`: Updated combined plot layout with bottom subtitle

**Result**: Combined plots now provide complete simulation configuration visibility for easy reproducibility and analysis.

---

## 2025-01-28: Fixed CMU Serif Font Warnings in Plotting

**Issue**: Plotting system was generating numerous font warnings:
```
UserWarning: Glyph 8722 (\N{MINUS SIGN}) missing from font(s) CMU Serif.
```

**Root Cause**: The `analysis_mitb_game.py` file was configured to use CMU Serif font, which doesn't have proper Unicode minus sign glyphs on all systems.

**Solution**: Updated font configuration in `analysis_mitb_game.py`:
- Changed from serif fonts (CMU Serif) to reliable sans-serif fonts (Arial, DejaVu Sans)
- Added `"axes.unicode_minus": False` to use ASCII minus instead of Unicode minus
- Changed mathtext fontset from "cm" to "dejavusans"

**Result**: All plotting now works without font warnings. Combined plots and individual plots generate cleanly after every round.

**Files Modified**:
- `Games/2_MIT_Beer_Game/scripts/analysis_mitb_game.py`: Updated matplotlib font configuration

---

## 2025-01-22: Real-Time LLM Logging System

**Implemented**: Real-time logging of all LLM calls (decision, communication, initialization) to human-readable log file with immediate writing after each call.

**Key Features**:
- Structured logging with emojis and separators for readability
- System prompts, user prompts, and model outputs all logged
- Round numbers included in headers
- Immediate file flushing for real-time monitoring
- Simplified round summaries replacing detailed end-of-round logging

**Files Modified**:
- `Games/2_MIT_Beer_Game/scripts/models_mitb_game.py`: Added `log_llm_call_immediately()` method and calls after each LLM interaction
- `Games/2_MIT_Beer_Game/scripts/MIT_Beer_Game.py`: Enhanced phase headers and simplified round summaries

**Bug Fixed**: Resolved Pydantic validation error for `human_log_file` field by changing type annotation from `TextIO` to `Optional[Any]`.

---

## 2025-01-22: Real-Time File Saving and Plotting Implementation

**Implemented**: System now saves CSV/JSON files and generates plots after every round (not just at the end).

**Key Changes**:
- CSV logs are appended after each round for real-time data access
- JSON logs are completely rewritten after each round (overwrite mode for clean structure)
- All plots (inventory, backlog, balance, orders, combined) generated after every round
- Combined plots include run settings display in top-right corner

**Files Modified**:
- `Games/2_MIT_Beer_Game/scripts/MIT_Beer_Game.py`: Added per-round file writing and plotting calls
- `Games/2_MIT_Beer_Game/scripts/analysis_mitb_game.py`: Enhanced `plot_beer_game_results()` to display run settings and handle empty dataframes

**Benefits**: 
- Real-time monitoring of simulation progress
- Immediate access to data for analysis during long runs
- Visual feedback on agent performance as simulation progresses

---

## 2025-01-22: Fixed LLM Client Initialization Issue

**Issue**: Connection errors when using `--provider anthropic` flag due to premature `lite_client` initialization.

**Root Cause**: `lite_client` was being initialized at module import time before `executeMITBeerGame.py` could override it with the correct provider.

**Solution**: 
- Changed `lite_client` initialization to lazy loading with `get_default_client()` function
- Updated all LLM call sites to use `client = lite_client or get_default_client()`
- Fixed model name resolution to use dynamic imports instead of global variables

**Files Modified**:
- `Games/2_MIT_Beer_Game/scripts/llm_calls_mitb_game.py`: Added lazy client initialization
- `Games/2_MIT_Beer_Game/scripts/models_mitb_game.py`: Updated to use dynamic model names
- `Games/2_MIT_Beer_Game/scripts/orchestrator_mitb_game.py`: Updated client usage

**Result**: Anthropic API integration now works correctly with proper model name resolution.

---

## 2025-01-22: Dynamic Cost Parameters in Prompts

**Enhanced**: All prompt functions now accept dynamic cost parameters instead of using hardcoded values.

**Changes Made**:
- `get_decision_prompt()`: Now accepts `holding_cost_per_unit` and `backlog_cost_per_unit` parameters
- `get_communication_prompt()`: Now accepts `profit_per_unit_sold`, `holding_cost_per_unit`, `backlog_cost_per_unit`
- `AgentContext` dataclass: Added cost parameter fields
- `PromptEngine.build_prompt()`: Updated to pass cost parameters to underlying methods

**Files Modified**:
- `Games/2_MIT_Beer_Game/scripts/prompts_mitb_game.py`: Updated all prompt functions and dataclass
- `Games/2_MIT_Beer_Game/scripts/MIT_Beer_Game.py`: Updated function calls to pass cost parameters

**Benefit**: Agents now receive accurate, dynamic cost information in their prompts based on simulation configuration.

---

## 2025-01-22: Added Zero-Order Option for High Holding Costs

**Enhancement**: Added guidance in decision prompts allowing agents to order 0 units when holding costs are excessive.

**Rationale**: Agents can now strategically reduce inventory to save money when holding costs outweigh potential sales benefits.

**Implementation**: Added specific language in `get_decision_prompt()` emphasizing that ordering 0 units is a valid strategic option for cost management.

**Files Modified**:
- `Games/2_MIT_Beer_Game/scripts/prompts_mitb_game.py`: Enhanced decision prompt with zero-order guidance

---

## 2025-01-22: Enhanced Agent Financial Awareness

**Added**: Strong warnings about balance management and long-term strategic thinking in agent prompts.

**Key Additions**:
1. **Balance Survival Warning**: "BALANCE IS YOUR LIFELINE" messaging emphasizing bankruptcy risk
2. **Long-term Strategy Guidance**: Encouragement to plan inventory for multiple future rounds
3. **Strategic Planning**: Advice to consider demand trends and maintain adequate buffer inventory

**Files Modified**:
- `Games/2_MIT_Beer_Game/scripts/prompts_mitb_game.py`: Enhanced decision and communication prompts

**Expected Impact**: Agents should make more financially prudent decisions and avoid bankruptcy through better balance management.

---

## 2025-01-22: Factory vs Other Agents Analysis

**Analysis**: Confirmed that Factory agent has distinct cost structure and operational role:

**Factory Differences**:
- **Lower Production Cost**: $1.50/unit vs $2.50/unit purchase cost for other agents
- **No Upstream Supplier**: Factory produces goods rather than purchasing from upstream
- **Production Scheduling**: Factory schedules production with 1-round lead time
- **Revenue Source**: Generates money through production rather than just markup

**Other Agents (Retailer, Wholesaler, Distributor)**:
- **Higher Purchase Cost**: $2.50/unit from upstream suppliers
- **Supply Chain Position**: Intermediaries between suppliers and customers
- **Markup Model**: Profit from difference between purchase ($2.50) and sale price ($5.00)

**Conclusion**: The cost difference reflects realistic supply chain economics where manufacturers (Factory) have lower unit costs than distributors.

---

## 2025-07-28: Strengthened Inventory Buffer Guidance

**Objective**: Prevent agents from running their inventory to zero and ensure they maintain at least a safety-stock buffer.

**Key Changes**:
- Replaced permissive language that allowed zero inventory with guidance to keep on-hand inventory **at or above** the safety stock `S_s`.
- Tightened **ORDERING OPTIONS** language: ordering 0 units is now *only* acceptable when on-hand inventory already exceeds `S_s` and there is a short-term need to shed excess stock.

**Files Modified**:
- `Games/2_MIT_Beer_Game/scripts/prompts_mitb_game.py`: Updated text in **strategy generation**, **strategy update**, **decision**, and **system** prompts.

**Test Plan**:
1. Run `pytest -q` to ensure no regressions.
2. Execute a short Beer-Game simulation and verify that agents no longer converge to zero inventory levels.

**Rollback**: Revert the specific prompt text changes in `prompts_mitb_game.py` if unintended agent behaviour emerges.

---

## 2025-07-28: Implemented Hyperparameter System with Proper Economic Names

**Objective**: Define and implement the missing hyperparameters (`safety_stock_Ss`, `gamma_hint`, `delta_hint`) with proper economic parameter names that flow from execution script to prompts.

**Key Changes**:
1. **Added Command-Line Arguments** in `executeMITBeerGame.py`:
   - `--safety_stock_target` (default: 10.0 units) - Target safety stock level S_s
   - `--backlog_clearance_rate` (default: 0.5) - Backlog clearance rate γ ∈ [0,1] 
   - `--demand_smoothing_factor` (default: 0.3) - Demand smoothing parameter δ

2. **Parameter Flow Implementation**:
   - Added parameters to `run_beer_game_simulation()` and `run_beer_game_generation()` function signatures
   - Updated agent method calls (`initialize_strategy`, `llm_decision`, `decide_order_quantity`) to pass hyperparameters
   - Enhanced `AgentContext` dataclass with properly named hyperparameter fields
   - Updated `llm_decision` method to map hyperparameters to `AgentContext`

3. **Prompt Function Updates**:
   - Renamed confusing parameter names across all prompt functions:
     - `safety_stock_Ss` → `safety_stock_target` 
     - `gamma_hint` → `backlog_clearance_rate`
     - `delta_hint` → `demand_smoothing_factor`
   - Updated all prompt functions: `get_strategy_generation_prompt`, `get_strategy_update_prompt`, `get_decision_prompt`, `get_communication_prompt`, etc.

**Files Modified**:
- `Games/2_MIT_Beer_Game/scripts/executeMITBeerGame.py`: Added CLI arguments and parameter passing
- `Games/2_MIT_Beer_Game/scripts/MIT_Beer_Game.py`: Updated function signatures and agent method calls  
- `Games/2_MIT_Beer_Game/scripts/models_mitb_game.py`: Enhanced `AgentContext` and agent methods
- `Games/2_MIT_Beer_Game/scripts/prompts_mitb_game.py`: Renamed parameters and updated all prompt functions

**Result**: Agents now receive specific hyperparameter guidance in prompts:
- Safety stock targets to maintain inventory buffers
- Backlog clearance rates for inventory management
- Demand smoothing factors for order quantity adjustments

**Test Plan**: Run simulation with custom hyperparameters to verify agents receive and use the guidance correctly.

---

## 2025-07-28: Unified Decision-Making Logic for Consistent Logging

**Issue**: When `communication_rounds=0`, the system used legacy `decide_order_quantity()` method instead of modern `llm_decision("decision", ...)` method, causing different terminal logging formats (missing `🤖[Claude:...]` tracers).

**Root Cause**: The decision logic branched based on `use_comm and recent_messages`. When communication rounds = 0, `recent_messages` was empty, triggering the legacy path with different logging.

**Solution**: 
- **Unified Logic**: Always use `llm_decision("decision", ...)` method regardless of communication settings
- **Parameter Handling**: Pass `comm_history=None` when no communication messages exist
- **Consistent Logging**: All agent decisions now use the same logging format with Anthropic client tracers

**Files Modified**:
- `Games/2_MIT_Beer_Game/scripts/MIT_Beer_Game.py`: Removed branching logic, always use `llm_decision()` method

**Result**: Terminal logging is now consistent whether communication is enabled or disabled. All agent decisions show the `🤖[Claude:model] Agent → decision_prompt` format and proper response logging.

**Test Plan**: Run simulations with `--communication_rounds 0` and `--communication_rounds 2` to verify identical logging formats.

---

## 2025-07-28: Added 3-Round Inventory Buffer Rule

**Objective**: Ensure agents maintain sufficient inventory resilience by requiring a minimum buffer of 3 rounds of expected demand.

**Key Change**: 
- Added **MINIMUM BUFFER RULE** to all prompt variants: "Always try to maintain enough on-hand inventory to serve at least 3 rounds of expected demand (3 × μ). This provides resilience against demand spikes and supply disruptions."

**Files Modified**:
- `Games/2_MIT_Beer_Game/scripts/prompts_mitb_game.py`: Updated all 4 prompt functions (strategy generation, strategy update, decision, system)

**Rationale**: 
- Prevents agents from running dangerously low inventory levels
- Provides buffer against demand volatility and supply chain disruptions  
- Encourages more stable, forward-looking inventory management
- Complements existing safety stock guidance with concrete quantitative target

**Test Plan**: Run simulation and verify agents maintain higher inventory levels, reducing stockout frequency while balancing holding costs.

**Rollback**: Remove the MINIMUM BUFFER RULE lines if agents become too conservative and inventory costs become excessive.

---

## 2025-07-28: Improved Plot Spacing in Combined Visualization  

**Objective**: Enhance readability of the combined plots by increasing vertical spacing between subplots.

**Key Change**: 
- Increased `hspace` parameter from 0.45 to 0.65 in `analysis_mitb_game.py` for better visual separation between inventory, backlog, balance, orders, and external demand plots.

**Files Modified**:
- `Games/2_MIT_Beer_Game/scripts/analysis_mitb_game.py`: Updated subplot spacing parameter

**Impact**: 
- Better visual clarity when analyzing simulation results
- Reduced overlap between subplot titles and axis labels
- Improved readability of multi-panel visualization

**Test Plan**: Run simulation and verify the combined plots have better spacing and are easier to read.

---

## 2025-07-28: Added Learning from Mistakes Guidance

**Objective**: Enable agents to learn from their past performance and adapt their strategies based on system feedback about backlog, inventory, and profit levels.

**Key Change**: 
- Added **📚 LEARN FROM MISTAKES** guidance to all prompt variants: "Analyze your recent performance history. If you experienced high backlog in previous rounds, increase your order quantities and safety stock to prevent stockouts. If you had excessive inventory leading to high holding costs and low profits, reduce order quantities but maintain minimum buffer levels. Adapt your ordering strategy based on what went wrong in past rounds - the system's responses (profit, backlog, inventory levels) are teaching you optimal behavior."

**Files Modified**:
- `Games/2_MIT_Beer_Game/scripts/prompts_mitb_game.py`: Updated all 5 prompt functions (strategy generation, strategy update, decision, communication, system)

**Rationale**: 
- **Adaptive Learning**: Agents can now recognize patterns in their performance history
- **Mistake Correction**: Explicit guidance to adjust based on past backlog/inventory issues  
- **System Feedback Loop**: Encourages agents to treat system responses as learning signals
- **Strategic Evolution**: Promotes continuous improvement rather than static decision-making
- **Communication Enhancement**: Agents can share lessons learned with other supply chain partners

**Expected Impact**:
- Reduced repeated mistakes (e.g., chronic stockouts or excessive inventory)
- Better convergence to optimal inventory policies over time
- More sophisticated inter-agent coordination based on shared learning
- Improved supply chain stability through collective experience

**Test Plan**: Run multi-round simulations and verify agents adjust their strategies when experiencing performance issues (high backlog → higher orders, excessive inventory → lower orders while maintaining buffers).

**Rollback**: Remove the LEARN FROM MISTAKES guidance if agents become overly reactive or unstable in their decision-making.

---

## 2025-07-28: Fixed Directory Creation Issue for CSV/JSON Logging

**Issue**: Simulation was crashing with `FileNotFoundError` when trying to write CSV and JSON log files because the results directory didn't exist yet.

**Root Cause**: 
- Results directory was created in `run_beer_game_simulation()` 
- But CSV/JSON file writing happened in `run_beer_game_generation()` 
- Race condition where file operations occurred before directory was fully established

**Solution**: 
- Added `os.makedirs(os.path.dirname(csv_log_path), exist_ok=True)` before CSV file writing
- Added `os.makedirs(os.path.dirname(json_log_path), exist_ok=True)` before JSON file writing
- These safety checks ensure the directory exists before any file operations

**Files Modified**:
- `Games/2_MIT_Beer_Game/scripts/MIT_Beer_Game.py`: Added directory creation safety checks for both CSV and JSON logging

**Impact**: 
- ✅ **Prevents crashes** during simulation execution
- ✅ **Robust file handling** with automatic directory creation
- ✅ **No data loss** from failed logging operations
- ✅ **Better error resilience** for file system operations

**Test Plan**: Run simulation and verify it completes successfully without FileNotFoundError crashes.

**Rollback**: Remove the `os.makedirs()` calls if they cause permission issues (unlikely).

---

## 2025-07-28: Expanded History Tracking from 10 to 20 Rounds

**Issue**: Agents were only tracking the last 3-10 rounds of history instead of the expected 20 rounds, limiting their ability to learn from past mistakes and adapt strategies.

**Root Cause**: 
- **Profit/Balance History**: Limited to 10 rounds in `update_profit_history()`
- **Recent Demand/Orders**: Mixed limits - `llm_decision()` used 10 rounds, `decide_order_quantity()` hardcoded 3 rounds
- **Inconsistent Implementation**: Different methods used different history lengths

**Solution**: 
- **Increased profit/balance history** from 10 to 20 rounds
- **Fixed hardcoded `[-3:]` limits** in `decide_order_quantity()` to use `history_limit` parameter  
- **Increased default `history_limit`** from 10 to 20 rounds across all methods
- **Unified history tracking** - all methods now consistently use 20-round history

**Files Modified**:
- `Games/2_MIT_Beer_Game/scripts/models_mitb_game.py`: Updated history limits and fixed hardcoded values
- `Games/2_MIT_Beer_Game/scripts/MIT_Beer_Game.py`: Updated agent decision calls to use 20-round history

**Impact**: 
- ✅ **True 20-round history tracking** for all agent decisions
- ✅ **Better learning from mistakes** with longer performance history
- ✅ **Consistent behavior** across all decision methods
- ✅ **Improved strategy adaptation** based on extended past performance
- ✅ **Enhanced EWMA calculations** with more historical demand data

**Expected Behavior**: 
- Agents will now see `recent_demand_or_orders` with up to 20 historical values
- `profit_history` and `balance_history` will show up to 20 rounds
- Better pattern recognition and trend analysis over longer periods

**Test Plan**: Run simulation and verify agent logs show extended history arrays instead of just 2-3 recent values.

**Rollback**: Change history limits back to 10 if memory usage becomes excessive or performance degrades.

---

## 2025-07-28: Fixed Missing cumulative_profit Attribute Error

**Issue**: Simulation was crashing with `AttributeError: 'AgentContext' object has no attribute 'cumulative_profit'` during communication phase.

**Root Cause**: 
- The `PromptEngine.build_prompt()` method was trying to access `ctx.cumulative_profit`
- But the `AgentContext` class was missing the `cumulative_profit` attribute
- The `ctx_kwargs` dictionary wasn't passing this value when creating the context

**Solution**: 
- **Added `cumulative_profit` attribute** to `AgentContext` class as `Optional[float] = None`
- **Added cumulative profit calculation** in `llm_decision()` method: `sum(self.profit_history) if self.profit_history else 0.0`
- **Updated context creation** to include the calculated cumulative profit

**Files Modified**:
- `Games/2_MIT_Beer_Game/scripts/prompts_mitb_game.py`: Added `cumulative_profit` attribute to `AgentContext`
- `Games/2_MIT_Beer_Game/scripts/models_mitb_game.py`: Added cumulative profit calculation to context creation

**Impact**: 
- ✅ **Prevents crashes** during communication phase
- ✅ **Enables cumulative profit tracking** for agent decision-making
- ✅ **Provides agents with total profit context** for better strategic decisions
- ✅ **Supports communication prompts** that reference cumulative performance

**Expected Behavior**: 
- Agents can now access their total cumulative profit across all rounds
- Communication phase will complete without AttributeError crashes
- Better strategic decision-making with full profit context

**Test Plan**: Run simulation with communication enabled and verify it completes without crashing.

**Rollback**: Remove the cumulative_profit attribute if it causes any unexpected behavior (unlikely).

---

## 2025-07-28: Fixed Orchestrator JSON Parsing Error

**Issue**: Orchestrator was failing with `'str' object has no attribute 'get'` error despite generating valid JSON responses.

**Root Cause**: 
- The orchestrator LLM returns a JSON array: `[{...}, {...}, {...}]`
- But the code was using `safe_parse_json()` which is designed for single JSON objects (dict)
- When `safe_parse_json()` failed to parse the array, the fallback logic expected a list but got a string

**Solution**: 
- **Replaced `safe_parse_json()`** with direct `json.loads()` for better array handling
- **Added format detection**: Handles both JSON arrays `[...]` and single objects `{...}`
- **Added validation**: Ensures each parsed item is a dictionary before calling `.get()`
- **Improved error logging**: Better debugging information when JSON parsing fails

**Files Modified**:
- `Games/2_MIT_Beer_Game/scripts/orchestrator_mitb_game.py`: Fixed JSON parsing and added validation

**Impact**: 
- ✅ **Prevents orchestrator crashes** during recommendation generation
- ✅ **Handles both array and object formats** from LLM responses
- ✅ **Better error recovery** with detailed logging
- ✅ **Robust validation** prevents type errors in downstream processing

**Expected Behavior**: 
- Orchestrator recommendations will be generated successfully
- Proper logging of orchestrator advice in both text and JSON files
- Agents will receive orchestrator guidance without system crashes

**Test Plan**: Run simulation with `--enable_orchestrator` and verify orchestrator recommendations appear in logs without errors.

**Rollback**: Revert to `safe_parse_json()` if the new parsing logic causes issues (unlikely).