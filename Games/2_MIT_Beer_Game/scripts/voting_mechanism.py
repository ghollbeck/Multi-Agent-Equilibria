import collections
from typing import Callable, Any, Optional, List, Dict

class VotingMechanism:
    """
    Implements the 'First-to-ahead-by-k' voting mechanism for reliable agent actions.
    """
    def __init__(self, k: int = 3, max_votes: int = 10):
        """
        Args:
            k: The margin of votes the leader must have over the runner-up.
            max_votes: The maximum number of votes to cast before forcing a decision.
        """
        self.k = k
        self.max_votes = max_votes

    def vote(self, 
             prompt_func: Callable[[], Any], 
             validation_func: Callable[[Any], bool],
             fallback_value: Any = None) -> Any:
        """
        Executes the voting process.

        Args:
            prompt_func: A function that calls the LLM and returns a raw response.
            validation_func: A function that takes the raw response and returns True if valid.
            fallback_value: Value to return if no valid votes are obtained.

        Returns:
            The winning value.
        """
        votes = collections.Counter()
        
        for _ in range(self.max_votes):
            # 1. Generate
            try:
                response = prompt_func()
            except Exception as e:
                # Log error if needed, but continue voting
                continue

            # 2. Validate (Red-flagging)
            if not validation_func(response):
                continue

            # 3. Tally
            # We assume response is hashable (int, str, frozen dict). 
            # If it's a dict, we might need to serialize it or pick a specific field.
            votes[response] += 1

            # 4. Check Condition
            if len(votes) == 0:
                continue
                
            most_common = votes.most_common(2)
            leader, leader_count = most_common[0]
            
            if len(most_common) < 2:
                # Only one option so far
                if leader_count >= self.k:
                    return leader
            else:
                runner_up, runner_up_count = most_common[1]
                if leader_count - runner_up_count >= self.k:
                    return leader

        # 5. Fallback (Max votes reached)
        if len(votes) > 0:
            return votes.most_common(1)[0][0]
        
        return fallback_value
