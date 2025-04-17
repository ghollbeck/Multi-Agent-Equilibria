class Prompts1:
    """
    Long-term focused prompts that emphasize sustainable cooperation and building trust.
    These prompts encourage agents to think about long-term payoffs and reputation.
    """
    
    @staticmethod    
    def get_first_action_prompt():
        return """
You are playing the Prisoner's Dilemma game for the first time against an unknown opponent.
This is a long-term repeated interaction where building trust and maintaining cooperation can lead to better outcomes for everyone.

Game Rules:
- If both players cooperate (C), both get 3 points
- If both players defect (D), both get 1 point
- If one cooperates and one defects, the defector gets 5 points and the cooperator gets 0 points

Consider:
1. Building long-term trust can lead to sustained mutual cooperation
2. Your reputation matters for future interactions
3. The first move often sets the tone for the relationship
4. Successful long-term strategies often involve clear signaling and reciprocity

This will be an iterated game, meaning you'll play multiple rounds against the same opponent.
Choose your first move (C for cooperate or D for defect) with a long-term perspective in mind.

Return your decision as a JSON object with the format:
{
  "action": "C or D",
  "rationale": "Brief explanation of your decision"
}
"""
    
    @staticmethod
    def get_next_round_decision_prompt(agent_name, own_last_action, opponent_name, opponent_last_action):
        return f"""
You are {agent_name} playing an iterated Prisoner's Dilemma game against {opponent_name}.
This is a long-term repeated interaction where building trust and maintaining cooperation can lead to better outcomes for everyone.

Game Rules:
- If both players cooperate (C), both get 3 points
- If both players defect (D), both get 1 point
- If one cooperates and one defects, the defector gets 5 points and the cooperator gets 0 points

Consider:
1. Building long-term trust can lead to sustained mutual cooperation
2. Your reputation matters for future interactions
3. While defection might give immediate gains, it can damage trust and lead to mutual defection
4. Successful long-term strategies often involve forgiveness and clear signaling

Last round:
- You played: {own_last_action}
- {opponent_name} played: {opponent_last_action}

Based on this context, choose your next move (C or D) and explain your reasoning.
Focus on building a sustainable long-term relationship.

Return your decision as a JSON object with the format:
{{
  "action": "C or D",
  "rationale": "Brief explanation of your decision"
}}
"""

class Prompts2:
    """
    Short-term focused prompts that emphasize immediate gains and tactical advantage.
    These prompts encourage agents to think about maximizing current round payoffs.
    """
    
    @staticmethod
    def get_first_action_prompt():
        return """
You are playing the Prisoner's Dilemma game for the first time against an unknown opponent.
Your goal is to maximize your immediate payoff in this round.

Game Rules:
- If both players cooperate (C), both get 3 points
- If both players defect (D), both get 1 point
- If one cooperates and one defects, the defector gets 5 points and the cooperator gets 0 points

Consider:
1. Immediate payoff maximization is crucial
2. The highest possible single-round payoff (5) comes from defecting while the opponent cooperates
3. Protecting yourself from exploitation is important
4. Evaluate the risks and benefits for this specific round

This will be an iterated game, but focus on your immediate advantage in this first round.
Choose your first move (C for cooperate or D for defect) to maximize your current payoff.

Return your decision as a JSON object with the format:
{
  "action": "C or D",
  "rationale": "Brief explanation of your decision"
}
"""
    
    @staticmethod
    def get_next_round_decision_prompt(agent_name, own_last_action, opponent_name, opponent_last_action):
        return f"""
You are {agent_name} playing a Prisoner's Dilemma game against {opponent_name}.
Your goal is to maximize your immediate payoff in the current round.

Game Rules:
- If both players cooperate (C), both get 3 points
- If both players defect (D), both get 1 point
- If one cooperates and one defects, the defector gets 5 points and the cooperator gets 0 points

Consider:
1. Immediate payoff maximization is crucial
2. The highest possible single-round payoff (5) comes from defecting while the opponent cooperates
3. Protecting yourself from exploitation is important
4. Each round should be treated as a tactical opportunity

Last round:
- You played: {own_last_action}
- {opponent_name} played: {opponent_last_action}

Based on this context, choose your next move (C or D) and explain your reasoning.
Focus on maximizing your immediate payoff.

Return your decision as a JSON object with the format:
{{
  "action": "C or D",
  "rationale": "Brief explanation of your decision"
}}
"""

# Global prompt for first action decision
prompt_first_action = """
You are playing the Prisoner's Dilemma game for the first time against an unknown opponent.

In this game:
- If both players cooperate (C), both get 3 points.
- If both players defect (D), both get 1 point.
- If one cooperates and one defects, the defector gets 5 points and the cooperator gets 0 points.

This will be an iterated game, meaning you'll play multiple rounds against the same opponent.
Choose your first move (C for cooperate or D for defect) and explain your reasoning.

Return your decision as a JSON object with the format:
{
  "action": "C or D",
  "rationale": "Brief explanation of your decision"
}
"""