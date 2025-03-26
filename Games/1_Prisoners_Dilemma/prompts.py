import json

class Prompts1:
    @staticmethod
    def get_strategy_generation_prompt(available_strategies):
        strategies_text = "\n".join(f"- {strategy}" for strategy in available_strategies)
        return f"""
        You are tasked with developing a novel strategy for a strategic decision-making game. 
        In this game, you interact with another player over multiple rounds, and your objective is to maximize your total score over the long run. 
        You have two possible actions in each round: "Cooperate" or "Defect".

        Considerations for Strategy Development:
        - Focus on building long-term beneficial relationships with the other player.
        - Implement mechanisms to correct errors in decision-making.
        - Develop adaptive response patterns based on the other player's behavior.
        - Balance the potential for both cooperation and defection to optimize your score.

        Available Strategies:
        {strategies_text}

        Guidelines:
        - Do not use any pre-existing theories or strategies from your training data.
        - Base your strategy purely on your own behavior and observations to maximize your score over time.

        Output Format:
        Your strategy should be structured in JSON format as follows:

        {{
            "strategy_rules": [list of conditional statements],
            "forgiveness_factor": 0-1,
            "retaliation_threshold": 0-1,
            "adaptability": 0-1,
            "rationale": "str"
        }}
        """.strip()

    @staticmethod
    def get_strategy_update_prompt(history_str, strategy_matrix, allowed_strategy_keys):
        allowed_strategy_text = ", ".join(allowed_strategy_keys)
        return f"""
        Based on your current strategy {json.dumps(strategy_matrix)} and the following interaction history:
        {history_str}

        Please update your strategy to maximize your long-term score.
        You must choose exactly one of the following predefined strategies:
        {allowed_strategy_text}

        Return only valid JSON with the keys:
        {{
            "strategy_rules": [list of conditional statements],
            "forgiveness_factor": 0-1,
            "retaliation_threshold": 0-1,
            "adaptability": 0-1,
            "rationale": "str"
        }}

        Ensure that the "rationale" value is exactly one of the following: {allowed_strategy_text}.
        """.strip()

    @staticmethod
    def get_decision_analysis_prompt(agent_name, own_history, strategy_matrix, opponent_name, opponent_history, opponent_model, opponent_coop_rate):
        return f"""Analyze this interaction history with {opponent_name}:
        Your Entire History:
        {own_history}

        Opponent's Entire History:
        {opponent_history}

        Your Strategy: {json.dumps(strategy_matrix)}
        Opponent's Model: {opponent_model}
        Opponent's Cooperation Rate: {opponent_coop_rate:.2f}

        Output MUST be valid JSON with:
        {{
            "action": "C/D",
            "confidence": 0-1,
            "rationale": "str",
            "expected_opponent_action": "C/D",
            "risk_assessment": "str"
        }}
        """.strip()






class Prompts2:
    @staticmethod
    def get_strategy_generation_prompt(available_strategies):
        return """
        This is a custom strategy generation prompt for Prompts2.
        Please develop a unique strategy based on your own insights.
        """.strip()

    @staticmethod
    def get_strategy_update_prompt(history_str, strategy_matrix, allowed_strategy_keys):
        return """
        This is a custom strategy update prompt for Prompts2.
        Please update your strategy based on your own analysis.
        """.strip()

    @staticmethod
    def get_decision_analysis_prompt(agent_name, own_history, strategy_matrix, opponent_name, opponent_history, opponent_model, opponent_coop_rate):
        return """
        This is a custom decision analysis prompt for Prompts2.
        Please analyze the situation and decide the best action.
        """.strip()



