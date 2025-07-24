import json
from typing import List, Dict, Optional, Literal
from dataclasses import dataclass

class BeerGamePrompts:
    """
    Provides system and user prompts for the MIT Beer Game.
    Each agent will use these to:
      1. Generate an initial ordering strategy,
      2. Update the strategy after each generation,
      3. Decide on a new order quantity each round.
    """

    # ------------------------------------------------------------------
    # 🆕 Role-specific context & system prompt templates
    # ------------------------------------------------------------------

    _ROLE_SPECIALTIES = {
        "Retailer": (
            "You directly face external customer demand and are the only stage that observes *true* market demand in real time. "
            "You cannot produce goods and must order from the Wholesaler. Your key objective is to avoid stock-outs to keep customers satisfied while minimising holding & backlog costs."),
        "Wholesaler": (
            "You are the intermediary between Retailer and Distributor. You see aggregated orders from the Retailer (not end-customer demand) and fulfil them from your inventory or by ordering from the Distributor. Your objective is to dampen demand variability while minimising costs."),
        "Distributor": (
            "You bridge the Wholesaler and Factory. You consolidate orders from the Wholesaler, buffer lead-time variability, and place orders to the Factory. Your objective is to provide stable, timely supply downstream while avoiding excessive inventory."),
        "Factory": (
            "You are the production stage. Instead of ordering upstream you *schedule production*. You can produce any quantity, but production appears downstream with a 1-round delay. Your objective is to balance production levels with downstream orders and to keep backlog low without creating costly excess stock.")
    }

    @classmethod
    def _role_context(cls, role_name: str) -> str:
        """Return one-sentence role context."""
        return cls._ROLE_SPECIALTIES.get(role_name, "")

    # ------------------------------------------------------------------
    # Strategy generation prompt
    # ------------------------------------------------------------------

    @staticmethod
    def get_strategy_generation_prompt(role_name: str, inventory: int = 100, backlog: int = 0, profit_per_unit_sold: float = 2.5, holding_cost_per_unit: float = 0.5, backlog_cost_per_unit: float = 1.5) -> str:
        role_context = BeerGamePrompts._role_context(role_name)
        return f"""
        You are the {role_name} in the MIT Beer Game. {role_context}
        Your task is to develop an ordering strategy that will minimize total costs 
        (holding costs + backlog costs - profits) over multiple rounds.

        Current State:
          • Initial Inventory: {inventory} units
          • Initial Backlog: {backlog} units
          • Profit per unit sold: ${profit_per_unit_sold}

        Consider:
          • Your current role's position in the supply chain
          • You have a 1-round lead time for the orders you place
          • You observe demand (if Retailer) or incoming orders (for other roles)
          • You want to avoid large swings (the Bullwhip effect)
          • You have a holding cost of {holding_cost_per_unit} per unit per round
          • You have a backlog cost of {backlog_cost_per_unit} per unit per round of unmet demand
          • You earn ${profit_per_unit_sold} profit for each unit sold
          • 🚨 BANKRUPTCY RULE: If your bank-account balance ever reaches $0 or below, you are bankrupt and the simulation ends. Plan orders so your balance always stays positive.
          • IMPORTANT: When determining order quantities, you must account for BOTH your current backlog AND expected new demand - backlog represents unfilled orders that must be fulfilled in addition to meeting new demand
          • SHIPMENT CONSTRAINT: You can only ship to downstream partners up to (their_order + your_backlog). Even with excess inventory, you cannot oversupply beyond this limit.
          • Never let the inventory go to zero.
          • **If your profits are negative or consistently low, consider that high inventory may be causing excessive storage (holding) costs. In such cases, you should consider reducing your inventory levels to help improve profitability.**
          • **ORDERING OPTIONS: You can choose to order 0 units if holding costs are too high and you want to decrease your inventory to save money. This is a valid strategy when you have excess inventory and want to reduce storage costs.**

        Please return only valid JSON with the following fields in order:

        {{
          "role_name": "{role_name}",
          "inventory": {inventory},
          "backlog": {backlog},
          "confidence": <float between 0 and 1>,
          "rationale": "<brief explanation of your reasoning>",
          "risk_assessment": "<describe any risks you anticipate>",
          " d_demand_next_round": <integer>,
          "order_quantity": <integer>
        }}

        IMPORTANT: Output ONLY the JSON object, with no markdown, no triple backticks, no code fences, and do NOT write the word 'json' anywhere. Your reply must be a single valid JSON object, nothing else. If you include anything else, your answer will be rejected. KEEP IT RATHER SHORT
        """

    @staticmethod
    def get_strategy_update_prompt(role_name: str, performance_log: str, current_strategy: dict,
                                 inventory: int = 100, backlog: int = 0, profit_per_unit_sold: float = 2.5, holding_cost_per_unit: float = 0.5, backlog_cost_per_unit: float = 1.5) -> str:
        role_context = BeerGamePrompts._role_context(role_name)
        return f"""
        You are the {role_name} in the MIT Beer Game. {role_context}
        
        Current State:
          • Current Inventory: {inventory} units
          • Current Backlog: {backlog} units
          • Profit per unit sold: ${profit_per_unit_sold}
        
        Here is your recent performance log:
        {performance_log}

        Your current strategy is:
        {json.dumps(current_strategy, indent=2)}

        Based on your performance and the desire to minimize holding & backlog costs while maximizing profits, 
        please propose any improvements to your ordering policy. 
        
        Remember:
          • You have a holding cost of 0.5 per unit per round
          • You have a backlog cost of 1.5 per unit per round of unmet demand (3x higher than holding cost)
          • You earn ${profit_per_unit_sold} profit for each unit sold
          • 🚨 BANKRUPTCY RULE: If your bank-account balance ever reaches $0 or below, you are bankrupt and the simulation ends. Adjust your strategy to keep a positive balance.
          • IMPORTANT: When determining order quantities, you must account for BOTH your current backlog AND expected new demand - backlog represents unfilled orders that must be fulfilled in addition to meeting new demand
          • SHIPMENT CONSTRAINT: You can only ship to downstream partners up to (their_order + your_backlog). Even with excess inventory, you cannot oversupply beyond this limit.
          • Never let the inventory go to zero.
          • **If your profits are negative or consistently low, you should consider that high inventory may be causing excessive storage (holding) costs. In such cases, consider reducing your inventory levels to help improve profitability.**
          • **ORDERING OPTIONS: You can choose to order 0 units if holding costs are too high and you want to decrease your inventory to save money. This is a valid strategy when you have excess inventory and want to reduce storage costs.**
          
        Return only valid JSON with the following fields in order:

        {{
          "role_name": "{role_name}",
          "inventory": {inventory},
          "backlog": {backlog},
          "confidence": <float between 0 and 1>,
          "rationale": "<brief explanation>",
          "risk_assessment": "<describe any risks>",
          "expected_demand_next_round": <integer>,
          "order_quantity": <integer>
        }}

        IMPORTANT: Output ONLY the JSON object, with no markdown, no triple backticks, no code fences, and do NOT write the word 'json' anywhere. Your reply must be a single valid JSON object, nothing else. If you include anything else, your answer will be rejected. KEEP IT RATHER SHORT
        """

    @staticmethod
    def get_decision_prompt(role_name: str,
                            inventory: int,
                            backlog: int,
                            recent_demand_or_orders: List[int],
                            incoming_shipments: List[int],
                            current_strategy: dict,
                            profit_per_unit_sold: float = 2.5,
                            holding_cost_per_unit: float = 0.5,
                            backlog_cost_per_unit: float = 1.5,
                            last_order_placed: int = None,
                            last_profit: float = None,
                            profit_history: List[float] = None,
                            balance_history: List[float] = None,
                            current_balance: float = None) -> str:
        role_context = BeerGamePrompts._role_context(role_name)
        # Handle None values gracefully for formatting
        display_balance = f"${current_balance:.2f}" if current_balance is not None else "N/A"
        profits_list = profit_history or []
        balance_list = balance_history or []
        return f"""
        You are the {role_name} in the MIT Beer Game. {role_context}
        Current State:
          - Inventory: {inventory} units
          - Backlog: {backlog} units
          - Recent downstream demand or orders: {recent_demand_or_orders}
          - Incoming shipments this round: {incoming_shipments}
          - Profit per unit sold: ${profit_per_unit_sold}
          - Last order placed: {last_order_placed}
          - Last round profit: {last_profit}
          - Current bank balance: {display_balance}
          - Profit history (last {len(profits_list)} rounds): {profits_list}
          - Balance history (last {len(balance_list)} rounds): {[f"${b:.2f}" for b in balance_list]}

        Your known lead time is 1 round for any order you place.

        Economics:
          - Holding cost: ${holding_cost_per_unit} per unit per round
          - Backlog cost: ${backlog_cost_per_unit} per unfilled unit per round
          - Profit: ${profit_per_unit_sold} per unit sold
          - 💀 **BALANCE IS YOUR LIFELINE**: Your bank balance is the most critical survival metric. Every order you place costs money immediately (purchase/production cost), and every unit you hold costs money per round (holding cost). If your balance reaches $0 or below, you go bankrupt and the entire simulation ends. Monitor your spending carefully and always ensure you have enough funds to cover your costs.
          - 🚨 BANKRUPTCY RULE: Keep your bank-account balance > $0 at all times. Planning that risks a zero or negative balance is unacceptable.
          - Never let the inventory go to zero.
          - **If your profits are negative or consistently low (for example, if last round profit is negative), consider that high inventory may be causing excessive storage (holding) costs. In such cases, you should consider reducing your inventory levels to help improve profitability.**
          - **ORDERING OPTIONS: You can choose to order 0 units if holding costs are too high and you want to decrease your inventory to save money. This is a valid strategy when you have excess inventory and want to reduce storage costs.**

        **Important Supply Chain Rules:**
        - You should avoid letting your inventory reach zero, as this causes stockouts and lost sales.
        - When deciding how much to order, consider your expected demand and spending over the next round (the lead time before your order arrives).
        - CRITICAL: You must account for BOTH your current backlog ({backlog} units) AND expected new demand. The backlog represents unfilled orders that must be fulfilled - your order quantity should cover both clearing backlog and meeting new demand.
        - SHIPMENT CONSTRAINT: You can only ship to downstream partners up to (their_order + your_backlog). Even with excess inventory, you cannot oversupply beyond this limit.
        - Review how much you have ordered and earned in the last round(s) to inform your decision.
        - Try to maintain a buffer of inventory to cover expected demand during the lead time.
        - 🎯 **LONG-TERM STRATEGY**: Think beyond just the next round. Consider trends in demand patterns, seasonal variations, and build inventory strategically to serve multiple future rounds. Don't just react to immediate needs - plan ahead to ensure you can consistently meet demand while managing costs effectively.

        Current Strategy:
        {json.dumps(current_strategy, indent=2)}

        Given this state, return valid JSON with the following fields in order:

        {{
          "role_name": "{role_name}",
          "inventory": {inventory},
          "backlog": {backlog},
          "recent_demand_or_orders": {recent_demand_or_orders},
          "incoming_shipments": {incoming_shipments},
          "last_order_placed": {last_order_placed},
          "expected_demand_next_round": <integer>,
          "confidence": <float between 0 and 1>,
          "rationale": "<brief explanation>",
          "risk_assessment": "<describe any risks>",
          "order_quantity": <integer>
        }}

        IMPORTANT: Output ONLY the JSON object, with no markdown, no triple backticks, no code fences, and do NOT write the word 'json' anywhere. Your reply must be a single valid JSON object, nothing else. If you include anything else, your answer will be rejected. KEEP IT RATHER SHORT
        """

    @staticmethod
    def get_communication_prompt(role_name: str, inventory: int, backlog: int, 
                                recent_demand_or_orders: List[int], current_strategy: dict,
                                message_history: List[Dict], other_agent_roles: List[str],
                                round_index: int, last_order_placed: int = None,
                                profit_accumulated: float = 0.0,
                                profit_per_unit_sold: float = 2.5,
                                holding_cost_per_unit: float = 0.5,
                                backlog_cost_per_unit: float = 1.5,
                                profit_history: List[float] = None,
                                balance_history: List[float] = None,
                                last_profit: float = None) -> str:
        role_context = BeerGamePrompts._role_context(role_name)
        return f"""
        You are the {role_name} in the MIT Beer Game, Round {round_index}. {role_context}
        
        Your Current State:
          - Inventory: {inventory} units
          - Backlog: {backlog} units  
          - Recent demand/orders: {recent_demand_or_orders}
          - Last order placed: {last_order_placed}
          - Last round profit: {last_profit}
          - Total profit so far: ${profit_accumulated:.2f}
          - Profit history (last {len(profit_history or [])} rounds): {profit_history or []}
          - Balance history (last {len(balance_history or [])} rounds): {[f"${b:.2f}" for b in (balance_history or [])]}
          - Current strategy: {json.dumps(current_strategy, indent=2)}
        
        CRITICAL SUPPLY CHAIN CONTEXT:
        ⏱️ LEAD TIME: All orders take EXACTLY 1 round to arrive (no exceptions)
        💰 ECONOMICS: Holding cost ${holding_cost_per_unit}/unit, Backlog cost ${backlog_cost_per_unit}/unit, Profit ${profit_per_unit_sold}/unit sold
        💀 **BALANCE SURVIVAL**: Your bank balance is your lifeline - if it reaches $0, you bankrupt and the simulation ends. Every order costs money immediately, every unit held costs money per round. Monitor spending carefully!
        📦 SHIPMENT RULE: Can only ship (downstream_order + your_backlog) - no oversupply allowed
        🏭 FACTORY SPECIAL: Factory schedules production (not orders) with same 1-round delay
        🎯 **STRATEGIC PLANNING**: Consider long-term demand trends and build inventory to serve multiple future rounds, not just immediate needs.
        
        Other agents in the supply chain: {', '.join(other_agent_roles)}
        
        Previous communication history:
        {json.dumps(message_history[-6:], indent=2) if message_history else "No previous messages"}
        
        You can now send a message to all other agents before this round's order decisions.
        Your goal is to share information that helps everyone optimize the supply chain while
        still maximizing your own profit.
        
        IMPORTANT: If other agents have spoken before you in this round, you MUST:
        1. Reflect on what they specifically said and how it affects you
        2. React by mentioning what you heard: "I hear [Agent X] said [specific point]..."
        3. State your conclusion and response: "Based on this, my conclusion is..."
        4. Explain how you will react: "Therefore, I will..."
        
        CRITICAL COMMUNICATION OBJECTIVES:
        📊 PROFIT FOCUS: Emphasize YOUR current profit (${profit_accumulated:.2f}) and how decisions affect it
        📦 INVENTORY STATUS: Share your inventory ({inventory} units) and backlog ({backlog} units) situation
        🤝 COLLABORATIVE PROFIT: Propose strategies to maximize TOTAL supply chain profits
        ⚠️ PREVENT COLLAPSE: Warn about risks of stockouts, oversupply, or market instability
        
        You MUST elaborate on:
        - Your profit trajectory and what threatens it
        - How your inventory/backlog affects the entire chain
        - Specific coordination to prevent bullwhip effect
        - Joint strategies to keep everyone profitable
        - Warning signs of potential market collapse
        - Your shipment constraints (can only ship orders + backlog)
        - Timing considerations: orders arrive in 1 round, plan accordingly
        - Cost implications: holding vs backlog costs affect profit margins
        - Supply chain stability: how to maintain steady flow without oscillations
         
        Be transparent about your situation while proposing win-win strategies.
        Emphasize that supply chain collapse hurts EVERYONE's profits.
        
        Return only valid JSON with these fields:
        
        {{
          "message": "<your message to other agents>",
          "strategy_hint": "<brief hint about your approach>", 
          "collaboration_proposal": "<specific proposal for working together>",
          "information_shared": "<key information you're willing to share>",
          "confidence": <float between 0 and 1>
        }}
        
        IMPORTANT: Output ONLY the JSON object, no markdown, no code fences. Keep it concise.
        """




    @staticmethod
    def get_decision_prompt_with_communication(role_name: str, inventory: int, backlog: int,
                                            recent_demand_or_orders: List[int], incoming_shipments: List[int],
                                            current_strategy: dict, profit_per_unit_sold: float = 2.5,
                                            holding_cost_per_unit: float = 0.5, backlog_cost_per_unit: float = 1.5,
                                            last_order_placed: int = None, last_profit: float = None,
                                            recent_communications: List[Dict] = None,
                                            profit_history: List[float] = None,
                                            balance_history: List[float] = None,
                                            current_balance: float = None) -> str:
        base_prompt = BeerGamePrompts.get_decision_prompt(
            role_name, inventory, backlog, recent_demand_or_orders, incoming_shipments,
            current_strategy, profit_per_unit_sold, last_order_placed, last_profit,
            profit_history, balance_history, current_balance
        )
        
        if recent_communications:
            comm_context = "\n\nRecent Communications from Other Agents:\n"
            for msg in recent_communications[-6:]:  # Last 6 messages
                comm_context += f"- {msg['sender']}: {msg['message']}\n"
                if msg.get('collaboration_proposal'):
                    comm_context += f"  Proposal: {msg['collaboration_proposal']}\n"
            
            comm_context += "\nConsider this information when making your order decision. Look for opportunities to coordinate while optimizing your own performance.\n"
            
            prompt_parts = base_prompt.split("Given this state, return valid JSON")
            enhanced_prompt = prompt_parts[0] + comm_context + "\nGiven this state and the communication context, return valid JSON" + prompt_parts[1]
            return enhanced_prompt
        
        return base_prompt

    @staticmethod
    def get_memory_context_prompt(agent_memory, num_rounds: int = 5) -> str:
        """Format agent memory into a prompt context section for LLM input."""
        if not agent_memory or not hasattr(agent_memory, 'get_memory_context_for_decision'):
            return ""
        
        decision_context = agent_memory.get_memory_context_for_decision()
        communication_context = agent_memory.get_memory_context_for_communication()
        
        if not decision_context and not communication_context:
            return ""
        
        memory_context = "\n\n## Your Past Experiences and Learning\n\n"
        
        if decision_context:
            memory_context += "**Past Decision Patterns:**\n"
            memory_context += decision_context + "\n\n"
        
        if communication_context:
            memory_context += "**Past Communication Patterns:**\n"
            memory_context += communication_context + "\n\n"
        
        memory_context += "Use these past experiences to inform your current decision, but adapt to changing conditions.\n"
        
        return memory_context




    @staticmethod
    def get_decision_prompt_with_memory(role_name: str, inventory: int, backlog: int,
                                      recent_demand_or_orders: List[int], incoming_shipments: List[int],
                                      current_strategy: dict, profit_per_unit_sold: float = 2.5,
                                      holding_cost_per_unit: float = 0.5, backlog_cost_per_unit: float = 1.5,
                                      last_order_placed: int = None, last_profit: float = None,
                                      agent_memory = None, memory_retention_rounds: int = 5,
                                      profit_history: List[float] = None,
                                      balance_history: List[float] = None,
                                      current_balance: float = None) -> str:
        """Enhanced decision prompt that incorporates agent memory if available."""
        base_prompt = BeerGamePrompts.get_decision_prompt(
            role_name, inventory, backlog, recent_demand_or_orders, incoming_shipments,
            current_strategy, profit_per_unit_sold, last_order_placed, last_profit,
            profit_history, balance_history, current_balance
        )
        
        if agent_memory:
            memory_context = BeerGamePrompts.get_memory_context_prompt(
                agent_memory, memory_retention_rounds
            )
            
            if memory_context:
                prompt_parts = base_prompt.split("Given this state, return valid JSON")
                enhanced_prompt = (
                    prompt_parts[0]
                    + memory_context
                    + "\nCarefully analyze your past behaviour and learning to craft a more sustainable, stable ordering decision. After this reflection, return valid JSON"
                    + prompt_parts[1]
                )
                return enhanced_prompt
        
        return base_prompt

    @staticmethod
    def get_communication_prompt_with_memory(role_name: str, inventory: int, backlog: int, 
                                           recent_demand_or_orders: List[int], current_strategy: dict,
                                           message_history: List[Dict], other_agent_roles: List[str],
                                           round_index: int, last_order_placed: int = None,
                                           profit_accumulated: float = 0.0, agent_memory = None,
                                           memory_retention_rounds: int = 5,
                                           profit_per_unit_sold: float = 2.5,
                                           holding_cost_per_unit: float = 0.5,
                                           backlog_cost_per_unit: float = 1.5,
                                           profit_history: List[float] = None,
                                           balance_history: List[float] = None,
                                           last_profit: float = None) -> str:
        """Enhanced communication prompt that incorporates agent memory if available."""
        base_prompt = BeerGamePrompts.get_communication_prompt(
            role_name, inventory, backlog, recent_demand_or_orders, current_strategy,
            message_history, other_agent_roles, round_index, last_order_placed, profit_accumulated,
            profit_history, balance_history, last_profit
        )
        
        if agent_memory:
            memory_context = BeerGamePrompts.get_memory_context_prompt(
                agent_memory, memory_retention_rounds
            )
            
            if memory_context:
                prompt_parts = base_prompt.split("Return only valid JSON with these fields:")
                enhanced_prompt = (
                    prompt_parts[0]
                    + memory_context
                    + "\nReflect on your past communication behaviour and lessons learned to propose more sustainable, coordinated actions. Then return only valid JSON with these fields:"
                    + prompt_parts[1]
                )
                return enhanced_prompt
        
        return base_prompt

    @staticmethod
    def get_system_prompt(role_name: str, enable_communication: bool = True) -> str:
        """Return a role-specific system prompt for the MIT Beer Game agent.

        The prompt describes the simulation, the agent's objective, the information
        it will receive in the user message, and the expected JSON-only output.
        """
        comm_clause = "You may also broadcast one concise message to the other agents before ordering each round." if enable_communication else "Communication is disabled in the current simulation run."
        role_context = BeerGamePrompts._role_context(role_name)

        return f"""
You are the {role_name} agent in the MIT Beer Game — a four-stage supply-chain simulation consisting of Retailer → Wholesaler → Distributor → Factory.

ROLE SPECIALTY: {role_context}

Your sole objective is to maximise YOUR cumulative profit across all rounds.  In every round you can:
1. Observe your private state (inventory, backlog, recent orders / demand, incoming shipments, last order placed, last round profit).
2. Decide an *order_quantity* for your upstream partner (Factory schedules production instead of ordering).
3. {comm_clause}

CRITICAL SHIPMENT CONSTRAINT: You can only ship to your downstream partner an amount equal to (their_order + your_current_backlog). You cannot ship more than this even if you have excess inventory. This prevents oversupplying and maintains realistic supply chain constraints.

After a full generation you may be asked to update your ordering strategy based on performance logs.

The upcoming USER message will always provide:
• Current round state and cost parameters (holding cost, backlog cost, profit per unit sold).
• Your current strategy JSON and any relevant hyper-parameters.
• When enabled, a short history of other agents' communications.
• When enabled, a summary of your past memories/experiences.

**IMPORTANT PROFITABILITY NOTE:** If your profits are negative or consistently low, you should consider that high inventory may be causing excessive storage (holding) costs. In such cases, consider reducing your inventory levels to help improve profitability.

**ORDERING OPTIONS:** You can choose to order 0 units if holding costs are too high and you want to decrease your inventory to save money. This is a valid strategy when you have excess inventory and want to reduce storage costs.

Respond ONLY with valid JSON that strictly follows the schema specified in the USER message for the current task (strategy_initialization, strategy_update, order_decision, or communication).  Do NOT include markdown, code fences, comments, or any text outside the JSON object.
"""       

@dataclass
class AgentContext:
    """Lightweight container with the info needed to build prompts."""
    inventory: int
    backlog: int
    recent_demand_or_orders: list
    incoming_shipments: list
    current_strategy: dict
    profit_per_unit_sold: float
    holding_cost_per_unit: float = 0.5
    backlog_cost_per_unit: float = 1.5
    last_order_placed: Optional[int] = None
    last_profit: Optional[float] = None
    profit_history: Optional[list] = None
    balance_history: Optional[list] = None
    current_balance: Optional[float] = None
    # Market-level metrics (optional)
    total_chain_inventory: Optional[int] = None
    total_chain_backlog: Optional[int] = None

class PromptEngine:
    """Unified interface that builds either a decision or communication prompt.
    For now it delegates to the existing prompt helpers so behaviour is unchanged.
    """

    @staticmethod
    def build_prompt(
        role_name: str,
        phase: Literal["decision", "communication"],
        ctx: AgentContext,
        comm_history: Optional[list] = None,
        memory_text: str = None,
        orchestrator_advice: str = None,
        history_limit: int = 10,
        other_agent_roles: Optional[list] = None,
        round_index: int = 0,
    ) -> str:
        """Return a full prompt string for the requested phase.
        memory_text and orchestrator_advice are appended if supplied (simple concat for now).
        """
        # Trim histories if needed
        demand_hist = (ctx.recent_demand_or_orders[-history_limit:] if history_limit else ctx.recent_demand_or_orders)
        shipments_hist = ctx.incoming_shipments[-history_limit:] if history_limit else ctx.incoming_shipments
        profit_hist = (ctx.profit_history[-history_limit:] if ctx.profit_history else None)
        balance_hist = (ctx.balance_history[-history_limit:] if ctx.balance_history else None)

        if phase == "decision":
            base_prompt = BeerGamePrompts.get_decision_prompt(
                role_name=role_name,
                inventory=ctx.inventory,
                backlog=ctx.backlog,
                recent_demand_or_orders=demand_hist,
                incoming_shipments=shipments_hist,
                current_strategy=ctx.current_strategy,
                profit_per_unit_sold=ctx.profit_per_unit_sold,
                holding_cost_per_unit=ctx.holding_cost_per_unit,
                backlog_cost_per_unit=ctx.backlog_cost_per_unit,
                last_order_placed=ctx.last_order_placed,
                last_profit=ctx.last_profit,
                profit_history=profit_hist,
                balance_history=balance_hist,
                current_balance=ctx.current_balance,
            )

            # Insert market overview and improvement sentence
            market_lines = ""
            if ctx.total_chain_inventory is not None and ctx.total_chain_backlog is not None:
                market_lines = (
                    f"\nMARKET OVERVIEW (entire chain):\n"
                    f"  - Total inventory across chain: {ctx.total_chain_inventory} units\n"
                    f"  - Total backlog  across chain: {ctx.total_chain_backlog} units\n"
                )

            improve_sentence = ("Based on your performance and the desire to minimise holding & backlog costs "
                                 "while maximising profits longterm, please propose any improvements to your ordering plan.")

            base_prompt = market_lines + "\n" + improve_sentence + "\n" + base_prompt
            # Inject optional memory / orchestrator sections
            inserts = []
            if comm_history:
                inserts.append("\n\nRecent Communications:\n" + json.dumps(comm_history[-history_limit:], indent=2))
            if memory_text:
                inserts.append("\n\nMEMORY CONTEXT:\n" + memory_text)
            if orchestrator_advice:
                inserts.append("\n\nORCHESTRATOR ADVICE:\n" + orchestrator_advice)
            return "\n".join(inserts) + "\n" + base_prompt if inserts else base_prompt
        else:  # communication
            base_prompt = BeerGamePrompts.get_communication_prompt(
                role_name=role_name,
                inventory=ctx.inventory,
                backlog=ctx.backlog,
                recent_demand_or_orders=demand_hist,
                current_strategy=ctx.current_strategy,
                message_history=comm_history or [],
                other_agent_roles=other_agent_roles or [],
                round_index=round_index,
                last_order_placed=ctx.last_order_placed,
                profit_accumulated=ctx.current_balance or 0.0,
                profit_per_unit_sold=ctx.profit_per_unit_sold,
                holding_cost_per_unit=ctx.holding_cost_per_unit,
                backlog_cost_per_unit=ctx.backlog_cost_per_unit,
                profit_history=profit_hist,
                balance_history=balance_hist,
                last_profit=ctx.last_profit,
            )
            if orchestrator_advice:
                base_prompt += "\n\nORCHESTRATOR ADVICE for All Agents:\n" + orchestrator_advice
            if memory_text:
                base_prompt += "\n\nMEMORY CONTEXT:\n" + memory_text

        # after building communication prompt, add checklist for speaking
        checklist = (
            "\nWhen you speak:\n"
            "  1. State your inventory / backlog truthfully.\n"
            "  2. Propose a concrete order plan (e.g. “I will order 8”).\n"
            "  3. Summarize what other agents have proposed or are planning, and state what the group should conclude from this.\n"
            "  4. Suggest a shared target (e.g. “Let’s all aim for 1-round cover”).\n"
            "  5. Propose a strategy to improve the supply chain (e.g. “Let’s all aim for 1-round cover”).\n"
        )
        return base_prompt + checklist
        