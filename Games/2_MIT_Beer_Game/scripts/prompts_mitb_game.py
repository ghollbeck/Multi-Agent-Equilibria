import json
from typing import List, Dict

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
    def get_strategy_generation_prompt(role_name: str, inventory: int = 100, backlog: int = 0, profit_per_unit_sold: float = 2.5) -> str:
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
          • You have a holding cost of 0.5 per unit per round
          • You have a backlog cost of 1.5 per unit per round of unmet demand (3x higher than holding cost)
          • You earn ${profit_per_unit_sold} profit for each unit sold
          • IMPORTANT: When determining order quantities, you must account for BOTH your current backlog AND expected new demand - backlog represents unfilled orders that must be fulfilled in addition to meeting new demand
          • SHIPMENT CONSTRAINT: You can only ship to downstream partners up to (their_order + your_backlog). Even with excess inventory, you cannot oversupply beyond this limit.
          • Never let the inventory go to zero.

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
                                 inventory: int = 100, backlog: int = 0, profit_per_unit_sold: float = 2.5) -> str:
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
          • IMPORTANT: When determining order quantities, you must account for BOTH your current backlog AND expected new demand - backlog represents unfilled orders that must be fulfilled in addition to meeting new demand
          • SHIPMENT CONSTRAINT: You can only ship to downstream partners up to (their_order + your_backlog). Even with excess inventory, you cannot oversupply beyond this limit.
          • Never let the inventory go to zero.
          
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
                            last_order_placed: int = None,
                            last_profit: float = None) -> str:
        role_context = BeerGamePrompts._role_context(role_name)
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

        Your known lead time is 1 round for any order you place.

        Economics:
          - Holding cost: $0.5 per unit per round
          - Backlog cost: $1.5 per unfilled unit per round (3x higher than holding cost)
          - Profit: ${profit_per_unit_sold} per unit sold
          - Never let the inventory go to zero.

        **Important Supply Chain Rules:**
        - You should avoid letting your inventory reach zero, as this causes stockouts and lost sales.
        - When deciding how much to order, consider your expected demand and spending over the next round (the lead time before your order arrives).
        - CRITICAL: You must account for BOTH your current backlog ({backlog} units) AND expected new demand. The backlog represents unfilled orders that must be fulfilled - your order quantity should cover both clearing backlog and meeting new demand.
        - SHIPMENT CONSTRAINT: You can only ship to downstream partners up to (their_order + your_backlog). Even with excess inventory, you cannot oversupply beyond this limit.
        - Review how much you have ordered and earned in the last round(s) to inform your decision.
        - Try to maintain a buffer of inventory to cover expected demand during the lead time.

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
                                profit_accumulated: float = 0.0) -> str:
        role_context = BeerGamePrompts._role_context(role_name)
        return f"""
        You are the {role_name} in the MIT Beer Game, Round {round_index}. {role_context}
        
        Your Current State:
          - Inventory: {inventory} units
          - Backlog: {backlog} units  
          - Recent demand/orders: {recent_demand_or_orders}
          - Last order placed: {last_order_placed}
          - Total profit so far: ${profit_accumulated:.2f}
          - Current strategy: {json.dumps(current_strategy, indent=2)}
        
        CRITICAL SUPPLY CHAIN CONTEXT:
        ⏱️ LEAD TIME: All orders take EXACTLY 1 round to arrive (no exceptions)
        💰 ECONOMICS: Holding cost $0.5/unit, Backlog cost $1.5/unit, Profit $2.5/unit sold
        📦 SHIPMENT RULE: Can only ship (downstream_order + your_backlog) - no oversupply allowed
        🏭 FACTORY SPECIAL: Factory schedules production (not orders) with same 1-round delay
        
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
                                            last_order_placed: int = None, last_profit: float = None,
                                            recent_communications: List[Dict] = None) -> str:
        base_prompt = BeerGamePrompts.get_decision_prompt(
            role_name, inventory, backlog, recent_demand_or_orders, incoming_shipments,
            current_strategy, profit_per_unit_sold, last_order_placed, last_profit
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
                                      last_order_placed: int = None, last_profit: float = None,
                                      agent_memory = None, memory_retention_rounds: int = 5) -> str:
        """Enhanced decision prompt that incorporates agent memory if available."""
        base_prompt = BeerGamePrompts.get_decision_prompt(
            role_name, inventory, backlog, recent_demand_or_orders, incoming_shipments,
            current_strategy, profit_per_unit_sold, last_order_placed, last_profit
        )
        
        if agent_memory:
            memory_context = BeerGamePrompts.get_memory_context_prompt(
                agent_memory, memory_retention_rounds
            )
            
            if memory_context:
                prompt_parts = base_prompt.split("Given this state, return valid JSON")
                enhanced_prompt = prompt_parts[0] + memory_context + "\nGiven this state and your past experiences, return valid JSON" + prompt_parts[1]
                return enhanced_prompt
        
        return base_prompt

    @staticmethod
    def get_communication_prompt_with_memory(role_name: str, inventory: int, backlog: int, 
                                           recent_demand_or_orders: List[int], current_strategy: dict,
                                           message_history: List[Dict], other_agent_roles: List[str],
                                           round_index: int, last_order_placed: int = None,
                                           profit_accumulated: float = 0.0, agent_memory = None,
                                           memory_retention_rounds: int = 5) -> str:
        """Enhanced communication prompt that incorporates agent memory if available."""
        base_prompt = BeerGamePrompts.get_communication_prompt(
            role_name, inventory, backlog, recent_demand_or_orders, current_strategy,
            message_history, other_agent_roles, round_index, last_order_placed, profit_accumulated
        )
        
        if agent_memory:
            memory_context = BeerGamePrompts.get_memory_context_prompt(
                agent_memory, memory_retention_rounds
            )
            
            if memory_context:
                prompt_parts = base_prompt.split("Return only valid JSON with these fields:")
                enhanced_prompt = prompt_parts[0] + memory_context + "\nReturn only valid JSON with these fields:" + prompt_parts[1]
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

Respond ONLY with valid JSON that strictly follows the schema specified in the USER message for the current task (strategy_initialization, strategy_update, order_decision, or communication).  Do NOT include markdown, code fences, comments, or any text outside the JSON object.
"""       