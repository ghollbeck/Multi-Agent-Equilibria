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

    @staticmethod
    def get_communication_prompt(role_name: str, inventory: int, backlog: int, 
                               recent_demand_or_orders: List[int], current_strategy: dict,
                               message_history: List[Dict], other_agent_roles: List[str],
                               round_index: int, last_order_placed: int = None,
                               profit_accumulated: float = 0.0) -> str:
        return f"""
        You are the {role_name} in the MIT Beer Game, Round {round_index}.
        
        Your Current State:
          - Inventory: {inventory} units
          - Backlog: {backlog} units  
          - Recent demand/orders: {recent_demand_or_orders}
          - Last order placed: {last_order_placed}
          - Total profit so far: ${profit_accumulated:.2f}
          - Current strategy: {json.dumps(current_strategy, indent=2)}
        
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
        
        Consider sharing:
        - What demand patterns or trends you've observed
        - Suggestions for coordination to reduce bullwhip effect
        - Your capacity constraints or inventory situation
        - Proposals for information sharing or collaboration
        - Strategic hints that benefit everyone (including yourself)
        - Your specific reactions to what other agents have shared
        
        Be strategic - acknowledge what others have said, share helpful information that 
        encourages good collaboration, but maintain your competitive advantage. Keep messages 
        concise but informative.
        
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
                                            current_strategy: dict, profit_per_unit_sold: float = 5,
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