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

    @staticmethod
    def _validate_economics_params(p, c, h, b):
        """Validate that all economics parameters are provided and not None."""
        missing = []
        if p is None:
            missing.append("selling_price_per_unit")
        if c is None:
            missing.append("unit_cost_per_unit")
        if h is None:
            missing.append("holding_cost_per_unit")
        if b is None:
            missing.append("backlog_cost_per_unit")
        
        if missing:
            raise ValueError(f"Missing required economics parameters: {', '.join(missing)}. All p/c/h/b parameters must be provided.")

    @staticmethod
    def _safe_format_currency(value):
        """Safely format currency values, handling None cases."""
        if value is None:
            return "N/A"
        return f"${value:.2f}"

    @staticmethod
    def _get_objective_guidance(longtermplanning_boolean: bool = False) -> str:
        """Return strategic objective guidance based on planning mode."""
        if longtermplanning_boolean:
            return """
🤝 **COLLABORATIVE LONG-TERM OPTIMIZATION MODE**:
Your primary objective is to maximize the TOTAL supply chain profitability while ensuring your own survival. This requires sophisticated coordination and strategic thinking:

• **Collective Success Strategy**: Recognize that supply chain instability hurts everyone. A thriving chain benefits all participants through stable demand, reduced volatility, and sustainable profits.
• **Mutual Interdependence**: Your upstream and downstream partners' health directly impacts your success. Their bankruptcy terminates the entire simulation, ending your profit potential.
• **Information Sharing**: Transparent communication about inventory levels, demand patterns, and financial health enables better chain-wide planning and reduces the bullwhip effect.
• **Coordinated Buffer Management**: Work together to maintain optimal inventory levels across the chain, avoiding both costly excess inventory and devastating stockouts.
• **Long-term Sustainability**: Make decisions that ensure the supply chain remains profitable and stable over many rounds, not just immediate gains.
• **Risk Mitigation**: Help weaker chain members avoid bankruptcy through strategic coordination, as their failure ends the game for everyone.

**Balance Individual & Collective Goals**: While optimizing for chain-wide success, ensure your own financial sustainability. You cannot help others if you go bankrupt yourself.

**Role Viability Priority**: When your backlog is high or financial situation is critical, role-specific objectives dominate consensus targets - your survival enables future collaboration.
"""
        else:
            return """
💰 **INDIVIDUAL PROFIT MAXIMIZATION MODE**:
Your primary and sole objective is to maximize YOUR individual cumulative profit across all rounds. While you operate within a supply chain, your strategic focus should be:

• **Self-Interest Priority**: Every decision should primarily benefit your bottom line. Other agents' success is secondary to your own profitability.
• **Competitive Advantage**: Use your knowledge of supply chain dynamics to gain advantages over other participants.
• **Strategic Information Control**: Share information only when it directly benefits your position. Withhold insights that might help competitors.
• **Opportunistic Behavior**: Capitalize on other agents' mistakes or suboptimal decisions to increase your market position.
• **Resource Optimization**: Focus on minimizing your own costs (holding, backlog) and maximizing your revenue, regardless of chain-wide effects.
• **Short-term Gains**: Prioritize immediate profit opportunities, even if they might create long-term supply chain instability.

**Individual Survival**: While you should avoid actions that would cause total supply chain failure (which ends the game), your primary concern is outperforming other agents in profitability.
"""

    @classmethod
    def _role_context(cls, role_name: str) -> str:
        """Return one-sentence role context."""
        return cls._ROLE_SPECIALTIES.get(role_name, "")

    # ------------------------------------------------------------------
    # Strategy generation prompt
    # ------------------------------------------------------------------

    @staticmethod
    def get_strategy_generation_prompt(role_name: str, inventory: int = 100, backlog: int = 0, 
                                     selling_price_per_unit: float = None, unit_cost_per_unit: float = None,
                                     holding_cost_per_unit: float = None, backlog_cost_per_unit: float = None,
                                     safety_stock_target: Optional[float] = None,
                                     backlog_clearance_rate: Optional[float] = None,
                                     demand_smoothing_factor: Optional[float] = None) -> str:
        role_context = BeerGamePrompts._role_context(role_name)
        
        # Validate required economics parameters
        BeerGamePrompts._validate_economics_params(selling_price_per_unit, unit_cost_per_unit, 
                                                 holding_cost_per_unit, backlog_cost_per_unit)
        
        # Use validated parameters
        p = selling_price_per_unit
        c = unit_cost_per_unit  
        h = holding_cost_per_unit
        b = backlog_cost_per_unit
        
        # Optional hyper-params section
        hyperparams_text = ""
        if safety_stock_target is not None or backlog_clearance_rate is not None or demand_smoothing_factor is not None:
            hyperparams_text = "\n\nHYPER-PARAMETERS PROVIDED:\n"
            if safety_stock_target is not None:
                hyperparams_text += f"• Safety stock S_s: {safety_stock_target} units\n"
            if backlog_clearance_rate is not None:
                hyperparams_text += f"• Backlog clearance rate γ hint: {backlog_clearance_rate}\n"
            if demand_smoothing_factor is not None:
                hyperparams_text += f"• Smoothing parameter δ hint: {demand_smoothing_factor}\n"
        
        return f"""
        You are the {role_name} in the MIT Beer Game. {role_context}
        Your task is to develop an ordering strategy that will minimize total costs 
        (holding costs + backlog costs - profits) over multiple rounds.

        Current State:
          • Initial Inventory: {inventory} units
          • Initial Backlog: {backlog} units

        Economics (per round):
          • Selling price per unit (p): {BeerGamePrompts._safe_format_currency(p)[1:]}
          • Unit cost per unit (c): {BeerGamePrompts._safe_format_currency(c)[1:]}  
          • Holding cost per unit (h): {BeerGamePrompts._safe_format_currency(h)[1:]}
          • Backlog cost per unit (b): {BeerGamePrompts._safe_format_currency(b)[1:]}
          
        Profit Formula:
          profit_t = p*sales_t - c*orders_t - h*inventory_end_t - b*backlog_end_t
          profit_t = {p}*sales_t - {c}*orders_t - {h}*inventory_end_t - {b}*backlog_end_t
{hyperparams_text}
        Consider:
          • Your current role's position in the supply chain
          • You have a 1-round lead time for the orders you place
          • You observe demand (if Retailer) or incoming orders (for other roles)
          • You want to avoid large swings (the Bullwhip effect)
          • 🚨 BANKRUPTCY RULE: If your bank-account balance ever reaches $0 or below, you are bankrupt and the simulation ends. Plan orders so your balance always stays positive.
          • IMPORTANT: When determining order quantities, you must account for BOTH your current backlog AND expected new demand - backlog represents unfilled orders that must be fulfilled in addition to meeting new demand
          • Shipment rule: You can ship at most min(inventory_available, downstream_order + your_backlog). Shipment is executed by the simulator; you do not decide shipments in your reply.
          • Inventory policy: Maintain on-hand inventory at or above the safety stock S_s to buffer demand variability. Avoid allowing inventory to drop to zero as this increases stock-out risk.
          • **MINIMUM BUFFER RULE**: Always try to maintain enough on-hand inventory to serve at least 3 rounds of expected demand (3 × μ). This provides resilience against demand spikes and supply disruptions.
          • 📚 **LEARN FROM MISTAKES**: Analyze your recent performance history. If you experienced high backlog in previous rounds, increase your order quantities and safety stock to prevent stockouts. If you had excessive inventory leading to high holding costs and low profits, reduce order quantities but maintain minimum buffer levels. Adapt your ordering strategy based on what went wrong in past rounds - the system's responses (profit, backlog, inventory levels) are teaching you optimal behavior.
          • **If your profits are negative or consistently low, consider that high inventory may be causing excessive storage (holding) costs. In such cases, you should consider reducing your inventory levels to help improve profitability.**
          • **ORDERING OPTIONS: Ordering 0 units is only acceptable when your current on-hand inventory already exceeds the safety stock buffer and there is a justified short-term need to reduce excess stock. Repeated zero orders that push inventory below S_s are discouraged.**

        Please return only valid JSON with the following fields in order:

        {{
          "role_name": "{role_name}",
          "inventory": {inventory},
          "backlog": {backlog},
          "confidence": <float between 0 and 1>,
          "rationale": "<brief explanation of your reasoning>",
          "risk_assessment": "<describe any risks you anticipate>",
          "expected_demand_next_round": <integer>,
          "order_quantity": <integer>
        }}

        IMPORTANT: Output ONLY the JSON object, with no markdown, no triple backticks, no code fences, and do NOT write the word 'json' anywhere. Your reply must be a single valid JSON object, nothing else. If you include anything else, your answer will be rejected. KEEP IT RATHER SHORT
        """

    @staticmethod
    def get_strategy_update_prompt(role_name: str, performance_log: str, current_strategy: dict,
                                 inventory: int = 100, backlog: int = 0, 
                                 selling_price_per_unit: float = None, unit_cost_per_unit: float = None,
                                 holding_cost_per_unit: float = None, backlog_cost_per_unit: float = None,
                                 safety_stock_target: Optional[float] = None,
                                 backlog_clearance_rate: Optional[float] = None,
                                 demand_smoothing_factor: Optional[float] = None) -> str:
        role_context = BeerGamePrompts._role_context(role_name)
        
        # Validate required economics parameters
        BeerGamePrompts._validate_economics_params(selling_price_per_unit, unit_cost_per_unit, 
                                                 holding_cost_per_unit, backlog_cost_per_unit)
        
        # Use validated parameters
        p = selling_price_per_unit
        c = unit_cost_per_unit  
        h = holding_cost_per_unit
        b = backlog_cost_per_unit
        
        # Optional hyper-params section
        hyperparams_text = ""
        if safety_stock_target is not None or backlog_clearance_rate is not None or demand_smoothing_factor is not None:
            hyperparams_text = "\n\nHYPER-PARAMETERS PROVIDED:\n"
            if safety_stock_target is not None:
                hyperparams_text += f"• Safety stock S_s: {safety_stock_target} units\n"
            if backlog_clearance_rate is not None:
                hyperparams_text += f"• Backlog clearance rate γ hint: {backlog_clearance_rate}\n"
            if demand_smoothing_factor is not None:
                hyperparams_text += f"• Smoothing parameter δ hint: {demand_smoothing_factor}\n"
        
        return f"""
        You are the {role_name} in the MIT Beer Game. {role_context}
        
        Current State:
          • Current Inventory: {inventory} units
          • Current Backlog: {backlog} units
          
        Economics (per round):
          • Selling price per unit (p): {BeerGamePrompts._safe_format_currency(p)[1:]}
          • Unit cost per unit (c): {BeerGamePrompts._safe_format_currency(c)[1:]}  
          • Holding cost per unit (h): {BeerGamePrompts._safe_format_currency(h)[1:]}
          • Backlog cost per unit (b): {BeerGamePrompts._safe_format_currency(b)[1:]}
          
        Profit Formula:
          profit_t = p*sales_t - c*orders_t - h*inventory_end_t - b*backlog_end_t
          profit_t = {p}*sales_t - {c}*orders_t - {h}*inventory_end_t - {b}*backlog_end_t
{hyperparams_text}        
        Here is your recent performance log:
        {performance_log}

        Your current strategy is:
        {json.dumps(current_strategy, indent=2)}

        Based on your performance and the desire to minimize holding & backlog costs while maximizing profits, 
        please propose any improvements to your ordering policy. 
        
        Remember:
          • 🚨 BANKRUPTCY RULE: If your bank-account balance ever reaches $0 or below, you are bankrupt and the simulation ends. Adjust your strategy to keep a positive balance.
          • IMPORTANT: When determining order quantities, you must account for BOTH your current backlog AND expected new demand - backlog represents unfilled orders that must be fulfilled in addition to meeting new demand
          • Shipment rule: You can ship at most min(inventory_available, downstream_order + your_backlog). Shipment is executed by the simulator; you do not decide shipments in your reply.
          • Inventory policy: Maintain on-hand inventory at or above the safety stock S_s to buffer demand variability. Avoid allowing inventory to drop to zero as this increases stock-out risk.
          • **MINIMUM BUFFER RULE**: Always try to maintain enough on-hand inventory to serve at least 3 rounds of expected demand (3 × μ). This provides resilience against demand spikes and supply disruptions.
          • 📚 **LEARN FROM MISTAKES**: Analyze your recent performance history. If you experienced high backlog in previous rounds, increase your order quantities and safety stock to prevent stockouts. If you had excessive inventory leading to high holding costs and low profits, reduce order quantities but maintain minimum buffer levels. Adapt your ordering strategy based on what went wrong in past rounds - the system's responses (profit, backlog, inventory levels) are teaching you optimal behavior.
          • **If your profits are negative or consistently low, you should consider that high inventory may be causing excessive storage (holding) costs. In such cases, consider reducing your inventory levels to help improve profitability.**
          • **ORDERING OPTIONS: Ordering 0 units is only acceptable when your current on-hand inventory already exceeds the safety stock buffer and there is a justified short-term need to reduce excess stock. Repeated zero orders that push inventory below S_s are discouraged.**
          
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
                            selling_price_per_unit: float = None,
                            unit_cost_per_unit: float = None,
                            holding_cost_per_unit: float = None,
                            backlog_cost_per_unit: float = None,
                            last_order_placed: int = None,
                            last_profit: float = None,
                            profit_history: List[float] = None,
                            balance_history: List[float] = None,
                            current_balance: float = None,
                            round_index: int = None,
                            longtermplanning_boolean: bool = False,
                            safety_stock_target: Optional[float] = None,
                            backlog_clearance_rate: Optional[float] = None,
                            demand_smoothing_factor: Optional[float] = None) -> str:
        role_context = BeerGamePrompts._role_context(role_name)
        
        # Validate required economics parameters
        BeerGamePrompts._validate_economics_params(selling_price_per_unit, unit_cost_per_unit, 
                                                 holding_cost_per_unit, backlog_cost_per_unit)
        
        # Handle None values gracefully for formatting
        display_balance = BeerGamePrompts._safe_format_currency(current_balance)
        profits_list = profit_history or []
        balance_list = balance_history or []
        objective_guidance = BeerGamePrompts._get_objective_guidance(longtermplanning_boolean)
        
        # Use validated parameters
        p = selling_price_per_unit
        c = unit_cost_per_unit  
        h = holding_cost_per_unit
        b = backlog_cost_per_unit
        
        round_text = f"Round {round_index}" if round_index is not None else "MIT Beer Game"
        
        # Optional hyper-params section with default guidance
        hyperparams_text = ""
        if safety_stock_target is not None or backlog_clearance_rate is not None or demand_smoothing_factor is not None:
            hyperparams_text = "\n\nHYPER-PARAMETERS PROVIDED:\n"
            if safety_stock_target is not None:
                hyperparams_text += f"• Safety stock S_s: {safety_stock_target} units\n"
            if backlog_clearance_rate is not None:
                hyperparams_text += f"• Backlog clearance rate γ hint: {backlog_clearance_rate}\n"
            if demand_smoothing_factor is not None:
                hyperparams_text += f"• Smoothing parameter δ hint: {demand_smoothing_factor}\n"
        else:
            hyperparams_text = "\n\nHYPER-PARAMETERS GUIDANCE:\nIf not provided, choose γ ∈ [0.25,1.0] and δ ∈ [0, 0.5·μ]; include chosen values in calc.\n"
        
        return f"""
        You are the {role_name} in the {round_text}. {role_context}
        
{objective_guidance}

        Current State:
          - Inventory: {inventory} units
          - Backlog: {backlog} units
          - Recent downstream demand or orders: {recent_demand_or_orders}
          - Incoming shipments this round: {incoming_shipments}
          - Last order placed: {last_order_placed}
          - Last round profit: {last_profit}
          - Current bank balance: {display_balance}
          - Profit history (last {len(profits_list)} rounds): {profits_list}
          - Balance history (last {len(balance_list)} rounds): {[BeerGamePrompts._safe_format_currency(b) for b in balance_list]}

        Your known lead time is 1 round for any order you place.

        Economics (per round):
          - Selling price per unit (p): {BeerGamePrompts._safe_format_currency(p)[1:]}
          - Unit cost per unit (c): {BeerGamePrompts._safe_format_currency(c)[1:]}  
          - Holding cost per unit (h): {BeerGamePrompts._safe_format_currency(h)[1:]}
          - Backlog cost per unit (b): {BeerGamePrompts._safe_format_currency(b)[1:]}
          
        Profit Formula:
          profit_t = p*sales_t - c*orders_t - h*inventory_end_t - b*backlog_end_t
          profit_t = {p}*sales_t - {c}*orders_t - {h}*inventory_end_t - {b}*backlog_end_t
          
          - 💀 **BALANCE IS YOUR LIFELINE**: Your bank balance is the most critical survival metric. Every order you place costs money immediately (purchase/production cost), and every unit you hold costs money per round (holding cost). If your balance reaches $0 or below, you go bankrupt and the entire simulation ends. Monitor your spending carefully and always ensure you have enough funds to cover your costs.
          - 🚨 BANKRUPTCY RULE: Keep your bank-account balance > $0 at all times. Planning that risks a zero or negative balance is unacceptable.
          - **If your profits are negative or consistently low (for example, if last round profit is negative), consider that high inventory may be causing excessive storage (holding) costs. In such cases, you should consider reducing your inventory levels to help improve profitability.**
          - **ORDERING OPTIONS: Ordering 0 units is only acceptable when your current on-hand inventory already exceeds the safety stock buffer and you have a justified short-term need to reduce excess stock. Repeated zero orders that push inventory below S_s are discouraged.**

        **Important Supply Chain Rules:**
        - You should avoid stockouts as they cause lost sales and customer dissatisfaction.
        - When deciding how much to order, consider your expected demand and spending over the next round (the lead time before your order arrives).
        - CRITICAL: You must account for BOTH your current backlog ({backlog} units) AND expected new demand. The backlog represents unfilled orders that must be fulfilled - your order quantity should cover both clearing backlog and meeting new demand.
        - Shipment rule: You can ship at most min(inventory_available, downstream_order + your_backlog). Shipment is executed by the simulator; you do not decide shipments in your reply.
        - Inventory policy: Maintain on-hand inventory at or above the safety stock S_s to buffer demand variability. Avoid allowing inventory to drop to zero as this increases stock-out risk.
        - **MINIMUM BUFFER RULE**: Always try to maintain enough on-hand inventory to serve at least 3 rounds of expected demand (3 × μ). This provides resilience against demand spikes and supply disruptions.
        - Review how much you have ordered and earned in the last round(s) to inform your decision.
        - Try to maintain a buffer of inventory to cover expected demand during the lead time.
        - 🎯 **LONG-TERM STRATEGY**: Think beyond just the next round. Consider trends in demand patterns, seasonal variations, and build inventory strategically to serve multiple future rounds, not just immediate needs. Don't just react to immediate needs - plan ahead to ensure you can consistently meet demand while managing costs effectively.
        - 📚 **LEARN FROM MISTAKES**: Analyze your recent performance history. If you experienced high backlog in previous rounds, increase your order quantities and safety stock to prevent stockouts. If you had excessive inventory leading to high holding costs and low profits, reduce order quantities but maintain minimum buffer levels. Adapt your ordering strategy based on what went wrong in past rounds - the system's responses (profit, backlog, inventory levels) are teaching you optimal behavior.
{hyperparams_text}
        **MANDATORY CONTROLLER & SELF-AUDIT:**
        
        You must use this controller to determine your order quantity:
        
        Controller you must use:
        μ = EWMA of recent demand
        IP = on_hand + in_transit − backlog  
        S* = μ*(lead_time+1) + S_s
        BacklogClear = γ * backlog (γ ∈ [0,1])
        O_raw = max(0, S* − IP + BacklogClear) → smooth within ±δ around μ → apply solvency cap so (balance − c*O_final) > 0
        
        Self-audit booleans:
        coverage_ok (if on_hand=0 then order ≥ expected_demand)
        includes_backlog  
        solvency_ok

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
          "order_quantity": <integer>,
          "calc": {{
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
          }}
        }}

        IMPORTANT: Output ONLY the JSON object, with no markdown, no triple backticks, no code fences, and do NOT write the word 'json' anywhere. Your reply must be a single valid JSON object, nothing else. If you include anything else, your answer will be rejected. KEEP IT RATHER SHORT
        """

    @staticmethod
    def get_communication_prompt(role_name: str, inventory: int, backlog: int, 
                                recent_demand_or_orders: List[int], current_strategy: dict,
                                message_history: List[Dict], other_agent_roles: List[str],
                                round_index: int, last_order_placed: int = None,
                                current_balance: float = 0.0,
                                selling_price_per_unit: float = None,
                                unit_cost_per_unit: float = None,
                                holding_cost_per_unit: float = None,
                                backlog_cost_per_unit: float = None,
                                profit_history: List[float] = None,
                                balance_history: List[float] = None,
                                last_profit: float = None,
                                longtermplanning_boolean: bool = False,
                                cumulative_profit: Optional[float] = None,
                                safety_stock_target: Optional[float] = None,
                                backlog_clearance_rate: Optional[float] = None,
                                demand_smoothing_factor: Optional[float] = None) -> str:
        role_context = BeerGamePrompts._role_context(role_name)
        
        # Validate required economics parameters
        BeerGamePrompts._validate_economics_params(selling_price_per_unit, unit_cost_per_unit, 
                                                 holding_cost_per_unit, backlog_cost_per_unit)
        
        objective_guidance = BeerGamePrompts._get_objective_guidance(longtermplanning_boolean)
        
        # Use validated parameters
        p = selling_price_per_unit
        c = unit_cost_per_unit  
        h = holding_cost_per_unit
        b = backlog_cost_per_unit
        
        # Format cumulative profit if provided
        cumulative_profit_text = ""
        if cumulative_profit is not None:
            cumulative_profit_text = f"           - Cumulative profit: {BeerGamePrompts._safe_format_currency(cumulative_profit)}\n"
        
        return f"""
        You are the {role_name} in the MIT Beer Game, Round {round_index}. {role_context}
        
{objective_guidance}
        
        Your Current State:
           - Inventory: {inventory} units
           - Backlog: {backlog} units  
           - Recent demand/orders: {recent_demand_or_orders}
           - Last order placed: {last_order_placed}
           - Last round profit: {last_profit}
           - Current balance: {BeerGamePrompts._safe_format_currency(current_balance)}
{cumulative_profit_text}           - Profit history (last {len(profit_history or [])} rounds): {profit_history or []}
           - Balance history (last {len(balance_history or [])} rounds): {[BeerGamePrompts._safe_format_currency(b) for b in (balance_history or [])]}
           - Current strategy: {json.dumps(current_strategy, indent=2)}
        
        CRITICAL SUPPLY CHAIN CONTEXT:
        ⏱️ LEAD TIME: All orders take EXACTLY 1 round to arrive (no exceptions)
        
        Economics (per round):
        💰 Selling price per unit (p): {BeerGamePrompts._safe_format_currency(p)[1:]}
        💸 Unit cost per unit (c): {BeerGamePrompts._safe_format_currency(c)[1:]}  
        🏪 Holding cost per unit (h): {BeerGamePrompts._safe_format_currency(h)[1:]}
        ⏰ Backlog cost per unit (b): {BeerGamePrompts._safe_format_currency(b)[1:]}
        
        Profit Formula:
        profit_t = p*sales_t - c*orders_t - h*inventory_end_t - b*backlog_end_t
        profit_t = {p}*sales_t - {c}*orders_t - {h}*inventory_end_t - {b}*backlog_end_t
        
        💀 **BALANCE SURVIVAL**: Your bank balance is your lifeline - if it reaches $0, you bankrupt and the simulation ends. Every order costs money immediately, every unit held costs money per round. Monitor spending carefully!
        📦 SHIPMENT RULE: You can ship at most min(inventory_available, downstream_order + your_backlog). Shipment is executed by the simulator; you do not decide shipments in your reply.
        🏭 FACTORY SPECIAL: Factory schedules production (not orders) with same 1-round delay
        🎯 **STRATEGIC PLANNING**: Consider long-term demand trends and build inventory to serve multiple future rounds, not just immediate needs.
        📚 **LEARN FROM MISTAKES**: Analyze your recent performance history. If you experienced high backlog in previous rounds, increase your order quantities and safety stock to prevent stockouts. If you had excessive inventory leading to high holding costs and low profits, reduce order quantities but maintain minimum buffer levels. Adapt your ordering strategy based on what went wrong in past rounds - the system's responses (profit, backlog, inventory levels) are teaching you optimal behavior.
        
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
        
        COMMUNICATION GUIDELINES - Share Only:
        📊 INVENTORY STATUS: Your current inventory ({inventory} units) and backlog ({backlog} units)
        📈 DEMAND PATTERN: Recent mean demand μ from your observations
        📋 ORDER PLAN: Your next order plan (ranges/targets acceptable, not exact quantities for others)
        ⚙️ BACKLOG STRATEGY: Your γ (backlog-clearance rate) approach
        ⚠️ MATERIAL RISKS: Specific operational risks you observe
        
        COMMUNICATION RESTRICTIONS:
        - Do NOT prescribe exact quantities to other roles
        - Do NOT use emotive language or urgent directives
        - Share factual information about your situation only
        - Ranges and targets are acceptable when discussing coordination
        - Focus on operational facts, not emotional appeals
        
        SHIPMENT INFORMATION:
        - Shipment rule: You can ship at most min(inventory_available, downstream_order + your_backlog). Shipment is executed by the simulator; you do not decide shipments in your reply.
        - Lead time: 1 round for all orders/shipments
        - Your role constraints and capabilities within the supply chain
        
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
                                             current_strategy: dict, 
                                             selling_price_per_unit: float = None,
                                             unit_cost_per_unit: float = None,
                                             holding_cost_per_unit: float = None, 
                                             backlog_cost_per_unit: float = None,
                                             last_order_placed: int = None, last_profit: float = None,
                                             recent_communications: List[Dict] = None,
                                             profit_history: List[float] = None,
                                             balance_history: List[float] = None,
                                             current_balance: float = None,
                                             round_index: int = None,
                                             safety_stock_target: Optional[float] = None,
                                             backlog_clearance_rate: Optional[float] = None,
                                             demand_smoothing_factor: Optional[float] = None) -> str:
        base_prompt = BeerGamePrompts.get_decision_prompt(
            role_name=role_name,
            inventory=inventory,
            backlog=backlog,
            recent_demand_or_orders=recent_demand_or_orders,
            incoming_shipments=incoming_shipments,
            current_strategy=current_strategy,
            selling_price_per_unit=selling_price_per_unit,
            unit_cost_per_unit=unit_cost_per_unit,
            holding_cost_per_unit=holding_cost_per_unit,
            backlog_cost_per_unit=backlog_cost_per_unit,
            last_order_placed=last_order_placed,
            last_profit=last_profit,
            profit_history=profit_history,
            balance_history=balance_history,
            current_balance=current_balance,
            round_index=round_index,
            safety_stock_target=safety_stock_target,
            backlog_clearance_rate=backlog_clearance_rate,
            demand_smoothing_factor=demand_smoothing_factor
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
                                      current_strategy: dict, 
                                      selling_price_per_unit: float = None,
                                      unit_cost_per_unit: float = None,
                                      holding_cost_per_unit: float = None, 
                                      backlog_cost_per_unit: float = None,
                                      last_order_placed: int = None, last_profit: float = None,
                                      agent_memory = None, memory_retention_rounds: int = 5,
                                      profit_history: List[float] = None,
                                      balance_history: List[float] = None,
                                      current_balance: float = None,
                                      round_index: int = None,
                                      safety_stock_target: Optional[float] = None,
                                      backlog_clearance_rate: Optional[float] = None,
                                      demand_smoothing_factor: Optional[float] = None) -> str:
        """Enhanced decision prompt that incorporates agent memory if available."""
        base_prompt = BeerGamePrompts.get_decision_prompt(
            role_name=role_name,
            inventory=inventory,
            backlog=backlog,
            recent_demand_or_orders=recent_demand_or_orders,
            incoming_shipments=incoming_shipments,
            current_strategy=current_strategy,
            selling_price_per_unit=selling_price_per_unit,
            unit_cost_per_unit=unit_cost_per_unit,
            holding_cost_per_unit=holding_cost_per_unit,
            backlog_cost_per_unit=backlog_cost_per_unit,
            last_order_placed=last_order_placed,
            last_profit=last_profit,
            profit_history=profit_history,
            balance_history=balance_history,
            current_balance=current_balance,
            round_index=round_index,
            safety_stock_target=safety_stock_target,
            backlog_clearance_rate=backlog_clearance_rate,
            demand_smoothing_factor=demand_smoothing_factor
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
                                           current_balance: float = 0.0, agent_memory = None,
                                           memory_retention_rounds: int = 5,
                                           selling_price_per_unit: float = None,
                                           unit_cost_per_unit: float = None,
                                           holding_cost_per_unit: float = None,
                                           backlog_cost_per_unit: float = None,
                                           profit_history: List[float] = None,
                                           balance_history: List[float] = None,
                                           last_profit: float = None,
                                           cumulative_profit: Optional[float] = None,
                                           safety_stock_target: Optional[float] = None,
                                           backlog_clearance_rate: Optional[float] = None,
                                           demand_smoothing_factor: Optional[float] = None) -> str:
        """Enhanced communication prompt that incorporates agent memory if available."""
        base_prompt = BeerGamePrompts.get_communication_prompt(
            role_name=role_name,
            inventory=inventory,
            backlog=backlog,
            recent_demand_or_orders=recent_demand_or_orders,
            current_strategy=current_strategy,
            message_history=message_history,
            other_agent_roles=other_agent_roles,
            round_index=round_index,
            last_order_placed=last_order_placed,
            current_balance=current_balance,
            selling_price_per_unit=selling_price_per_unit,
            unit_cost_per_unit=unit_cost_per_unit,
            holding_cost_per_unit=holding_cost_per_unit,
            backlog_cost_per_unit=backlog_cost_per_unit,
            profit_history=profit_history,
            balance_history=balance_history,
            last_profit=last_profit,
            longtermplanning_boolean=longtermplanning_boolean,
            cumulative_profit=cumulative_profit,
            safety_stock_target=safety_stock_target,
            backlog_clearance_rate=backlog_clearance_rate,
            demand_smoothing_factor=demand_smoothing_factor
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

Your sole objective is to maximise YOUR cumulative profit across all rounds. 

Profit Formula (per round):
profit_t = p*sales_t - c*orders_t - h*inventory_end_t - b*backlog_end_t

Where:
- p = selling price per unit
- c = unit cost per unit  
- h = holding cost per unit
- b = backlog cost per unit

**LOGGING & ANALYSIS**: Each round follows this event order: orders placed → production → shipments sent/received → sales → costs → ending inventory/backlog. All performance metrics are computed from these logged events (no placeholders).

In every round you can:
1. Observe your private state (inventory, backlog, recent orders / demand, incoming shipments, last order placed, last round profit).
2. Decide an *order_quantity* for your upstream partner (Factory schedules production instead of ordering).
3. {comm_clause}

Shipment rule: You can ship at most min(inventory_available, downstream_order + your_backlog). Shipment is executed by the simulator; you do not decide shipments in your reply.

Inventory policy: Maintain on-hand inventory at or above the safety stock S_s to buffer demand variability. Avoid allowing inventory to drop to zero as this increases stock-out risk.

**MINIMUM BUFFER RULE**: Always try to maintain enough on-hand inventory to serve at least 3 rounds of expected demand (3 × μ). This provides resilience against demand spikes and supply disruptions.

📚 **LEARN FROM MISTAKES**: Analyze your recent performance history. If you experienced high backlog in previous rounds, increase your order quantities and safety stock to prevent stockouts. If you had excessive inventory leading to high holding costs and low profits, reduce order quantities but maintain minimum buffer levels. Adapt your ordering strategy based on what went wrong in past rounds - the system's responses (profit, backlog, inventory levels) are teaching you optimal behavior.

After a full generation you may be asked to update your ordering strategy based on performance logs.

The upcoming USER message will always provide:
• Current round state and cost parameters (selling price, unit cost, holding cost, backlog cost).
• Your current strategy JSON and any relevant hyper-parameters.
• When enabled, a short history of other agents' communications.
• When enabled, a summary of your past memories/experiences.

**IMPORTANT PROFITABILITY NOTE:** If your profits are negative or consistently low, you should consider that high inventory may be causing excessive storage (holding) costs. In such cases, consider reducing your inventory levels to help improve profitability.

**ORDERING OPTIONS:** Ordering 0 units is only acceptable when your current on-hand inventory already exceeds the safety stock buffer and you have a justified short-term need to reduce excess stock. Repeated zero orders that push inventory below S_s are discouraged.

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
    selling_price_per_unit: float
    unit_cost_per_unit: float
    holding_cost_per_unit: Optional[float] = None  # Must be provided
    backlog_cost_per_unit: Optional[float] = None  # Must be provided
    last_order_placed: Optional[int] = None
    last_profit: Optional[float] = None
    profit_history: Optional[list] = None
    balance_history: Optional[list] = None
    current_balance: Optional[float] = None
    # Market-level metrics (optional)
    total_chain_inventory: Optional[int] = None
    total_chain_backlog: Optional[int] = None
    # Additional financial metrics
    cumulative_profit: Optional[float] = None
    # Optional hyper-parameters
    safety_stock_target: Optional[float] = None
    backlog_clearance_rate: Optional[float] = None
    demand_smoothing_factor: Optional[float] = None
    # Deprecated - keep for backward compatibility
    profit_per_unit_sold: Optional[float] = None

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
        longtermplanning_boolean: bool = False,
    ) -> str:
        """Return a full prompt string for the requested phase.
        memory_text and orchestrator_advice are appended if supplied (simple concat for now).
        """
        # Validate economics parameters
        BeerGamePrompts._validate_economics_params(ctx.selling_price_per_unit, ctx.unit_cost_per_unit,
                                                 ctx.holding_cost_per_unit, ctx.backlog_cost_per_unit)
        
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
                selling_price_per_unit=ctx.selling_price_per_unit,
                unit_cost_per_unit=ctx.unit_cost_per_unit,
                holding_cost_per_unit=ctx.holding_cost_per_unit,
                backlog_cost_per_unit=ctx.backlog_cost_per_unit,
                last_order_placed=ctx.last_order_placed,
                last_profit=ctx.last_profit,
                profit_history=profit_hist,
                balance_history=balance_hist,
                current_balance=ctx.current_balance,
                round_index=round_index,
                longtermplanning_boolean=longtermplanning_boolean,
                safety_stock_target=ctx.safety_stock_target,
                backlog_clearance_rate=ctx.backlog_clearance_rate,
                demand_smoothing_factor=ctx.demand_smoothing_factor,
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
                current_balance=ctx.current_balance or 0.0,
                selling_price_per_unit=ctx.selling_price_per_unit,
                unit_cost_per_unit=ctx.unit_cost_per_unit,
                holding_cost_per_unit=ctx.holding_cost_per_unit,
                backlog_cost_per_unit=ctx.backlog_cost_per_unit,
                profit_history=profit_hist,
                balance_history=balance_hist,
                last_profit=ctx.last_profit,
                longtermplanning_boolean=longtermplanning_boolean,
                cumulative_profit=ctx.cumulative_profit,
                safety_stock_target=ctx.safety_stock_target,
                backlog_clearance_rate=ctx.backlog_clearance_rate,
                demand_smoothing_factor=ctx.demand_smoothing_factor
            )
            if orchestrator_advice:
                base_prompt += "\n\nORCHESTRATOR ADVICE for All Agents:\n" + orchestrator_advice
            if memory_text:
                base_prompt += "\n\nMEMORY CONTEXT:\n" + memory_text

        # after building communication prompt, add checklist for speaking
        checklist = (
            "\nWhen you speak:\n"
            "  1. State your inventory and backlog truthfully.\n"
            "  2. State how much profit you made last round.\n"
            "  3. Share your general ordering approach and reasoning.\n"
            "  4. Clearly state how much you plan to order this round.\n"
            "  5. Suggest chain-level targets or coordination strategies (ranges acceptable, no role-specific directives).\n"
            "  6. Acknowledge and consider what other agents have proposed or are planning, and state what the group should conclude from this.\n"
            "  7. Suggest a shared target or coordination strategy.\n"
            "  8. Propose a strategy to improve the supply chain.\n"
        )
        return base_prompt + checklist
        
    # ------------------------------------------------------------------
    # Orchestrator prompt
    # ------------------------------------------------------------------
    @staticmethod
    def get_orchestrator_prompt(state_block: str, external_demand: int, round_index: int, history_block: str, history_window: int) -> str:
        """Return the orchestrator user prompt string."""
        return (
            "You are the ORCHESTRATOR overseeing the entire MIT Beer Game supply chain.\n"
            "Your goal each round is to recommend order quantities for every role so that:\n"
            "• Total backlog and holding costs across the chain stay minimal.\n"
            "• The chain remains profitable as a whole, even if one role must temporarily reduce its own profit.\n"
            "• Inventories stay within reasonable bounds to avoid the bullwhip effect.\n\n"
            f"ROUND: {round_index}  |  External customer demand this round: {external_demand}\n"
            "Current state (inventory, backlog, balance, last_order):\n" + state_block + "\n\n"
            f"Recent history (last {history_window} rounds):\n{history_block}\n\n"
            "Return valid JSON ONLY in the following list format (no markdown):\n"
            "[\n"
            "  {\"role_name\": \"Retailer\", \"order_quantity\": <int>, \"rationale\": \"<short reason>\"},\n"
            "  {\"role_name\": \"Wholesaler\", \"order_quantity\": <int>, \"rationale\": \"<short reason>\"},\n"
            "  {\"role_name\": \"Distributor\", \"order_quantity\": <int>, \"rationale\": \"<short reason>\"},\n"
            "  {\"role_name\": \"Factory\", \"order_quantity\": <int>, \"rationale\": \"<short reason>\"}\n"
            "]\n"
            "IMPORTANT: output ONLY valid JSON – a list of four objects, one per role, nothing else."
        )
        