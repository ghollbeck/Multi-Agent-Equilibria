from typing import List, Dict, Any
import json
import asyncio

from models_mitb_game import BeerGameAgent, BeerGameLogger
from llm_calls_mitb_game import lite_client, MODEL_NAME, safe_parse_json, get_default_client
from prompts_mitb_game import BeerGamePrompts

class BeerGameOrchestrator:
    """LLM-based orchestrator that recommends order quantities for each role each round."""

    def __init__(self, history_window: int = 3, model_name: str = MODEL_NAME,
                 logger: BeerGameLogger = None):
        self.history_window = max(0, history_window)
        self.model_name = model_name
        self.logger = logger

    # ------------------------------------------------------------------
    # Prompt construction helpers
    # ------------------------------------------------------------------
    def _build_prompt(self, agents: List[BeerGameAgent], external_demand: int,
                      round_index: int, history: List[Dict[str, Any]]) -> str:
        state_lines = [
            f"- {ag.role_name}: inv={ag.inventory}, backlog={ag.backlog}, balance={ag.balance:.2f}, last_order={ag.last_order_placed}"
            for ag in agents
        ]
        state_block = "\n".join(state_lines)
        # History summary (optional)
        hist_str = (
            json.dumps(history[-self.history_window:], indent=2)[:1500]
            if history else "No prior history provided."
        )
        return BeerGamePrompts.get_orchestrator_prompt(state_block, external_demand, round_index, hist_str, self.history_window)

    async def get_recommendations(self, agents: List[BeerGameAgent], external_demand: int,
                                  round_index: int, history: List[Dict[str, Any]] = None,
                                  temperature: float = 0.3) -> Dict[str, Dict[str, Any]]:
        """Call the LLM and return a dictionary keyed by role."""
        if history is None:
            history = []
        prompt = self._build_prompt(agents, external_demand, round_index, history)
        sys_prompt = "You are a top-tier operations research expert coordinating a supply chain."

        # Log system and user prompts
        if self.logger:
            self.logger.log("\n🧠 [Orchestrator] SYSTEM PROMPT:\n" + sys_prompt)
            self.logger.log("👤 [Orchestrator] USER PROMPT:\n" + prompt)
            if getattr(self.logger, 'file_handle', None):
                fh = self.logger.file_handle
                fh.write("\n🧠 ORCHESTRATOR SYSTEM PROMPT\n")
                fh.write(sys_prompt + "\n")
                fh.write("👤 ORCHESTRATOR USER PROMPT\n")
                fh.write(prompt + "\n")

        try:
            client = lite_client or get_default_client()
            # Get the current model name from the module
            from llm_calls_mitb_game import MODEL_NAME as current_model
            resp = await client.chat_completion(
                model=current_model,
                system_prompt=sys_prompt,
                user_prompt=prompt,
                temperature=temperature,
                agent_role="Orchestrator",
                decision_type="orchestrator_recommendation"
            )
        except Exception as e:
            if self.logger:
                self.logger.log(f"[Orchestrator] LLM call failed: {e}")
            resp = "[]"

        if self.logger:
            self.logger.log(f"[Orchestrator] Raw response: {resp}")

        try:
            # Try to parse as JSON - could be a list or dict
            parsed_json = json.loads(resp.strip())
            
            # If it's a list, use it directly
            if isinstance(parsed_json, list):
                parsed = parsed_json
            # If it's a dict, wrap it in a list
            elif isinstance(parsed_json, dict):
                parsed = [parsed_json]
            else:
                # Fallback for unexpected format
                raise ValueError("Unexpected JSON format")
                
        except Exception as e:
            if self.logger:
                self.logger.log(f"[Orchestrator] JSON parsing failed: {e}")
            # Fallback – equal split order 10 each
            parsed = [
                {"role_name": ag.role_name, "order_quantity": 10, "rationale": "default"}
                for ag in agents
            ]

        # Convert list to dict
        recs: Dict[str, Dict[str, Any]] = {}
        for item in parsed:
            # Ensure item is a dictionary
            if not isinstance(item, dict):
                if self.logger:
                    self.logger.log(f"[Orchestrator] Skipping non-dict item: {item}")
                continue
                
            role = item.get("role_name")
            if role:
                recs[role] = {
                    "order_quantity": int(item.get("order_quantity", 10)),
                    "rationale": item.get("rationale", "")
                }
        return recs 