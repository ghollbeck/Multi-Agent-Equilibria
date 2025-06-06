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

    @staticmethod
    def get_strategy_generation_prompt(role_name: str, inventory: int = 100, backlog: int = 0, profit_per_unit_sold: float = 5) -> str:
        return f"""
        You are the {role_name} in the MIT Beer Game. 
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
         • Never let the inventory go to zero.

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
                                 inventory: int = 100, backlog: int = 0, profit_per_unit_sold: float = 5) -> str:
        return f"""
        You are the {role_name} in the MIT Beer Game. 
        
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
                            profit_per_unit_sold: float = 5,
                            last_order_placed: int = None,
                            last_profit: float = None) -> str:
        return f"""
        You are the {role_name} in the MIT Beer Game. 
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

        **Important:**
        - You should avoid letting your inventory reach zero, as this causes stockouts and lost sales.
        - When deciding how much to order, consider your expected demand and spending over the next round (the lead time before your order arrives).
        - CRITICAL: You must account for BOTH your current backlog ({backlog} units) AND expected new demand. The backlog represents unfilled orders that must be fulfilled - your order quantity should cover both clearing backlog and meeting new demand.
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