from typing import List, Dict, Any
import json
import asyncio

from models_mitb_game import BeerGameAgent, BeerGameLogger
from llm_calls_mitb_game import lite_client, MODEL_NAME, safe_parse_json, get_default_client

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
        """Return a user prompt string describing current + historical state."""
        state_lines = []
        for ag in agents:
            state_lines.append(
                f"- {ag.role_name}: inv={ag.inventory}, backlog={ag.backlog}, "
                f"balance={ag.balance:.2f}, last_order={ag.last_order_placed}"
            )
        state_block = "\n".join(state_lines)

        # History summary (optional)
        if history:
            # Keep it compact
            hist_str = json.dumps(history[-self.history_window:], indent=2)[:1500]
        else:
            hist_str = "No prior history provided."

        return (
            "You are the ORCHESTRATOR overseeing the entire MIT Beer Game supply chain.\n"
            "Your goal each round is to recommend order quantities for every role so that:\n"
            "• Total backlog and holding costs across the chain stay minimal.\n"
            "• The chain remains profitable as a whole, even if one role must temporarily reduce its own profit.\n"
            "• Inventories stay within reasonable bounds to avoid the bull-whip effect.\n\n"
            f"ROUND: {round_index}  |  External customer demand this round: {external_demand}\n"
            "Current state (inventory, backlog, balance, last_order):\n" + state_block + "\n\n"
            f"Recent history (last {self.history_window} rounds):\n{hist_str}\n\n"
            "Return valid JSON ONLY in the following list format (no markdown):\n"
            "[\n"
            "  {\"role_name\": \"Retailer\", \"order_quantity\": <int>, \"rationale\": \"<short reason>\"},\n"
            "  {\"role_name\": \"Wholesaler\", \"order_quantity\": <int>, \"rationale\": \"<short reason>\"},\n"
            "  {\"role_name\": \"Distributor\", \"order_quantity\": <int>, \"rationale\": \"<short reason>\"},\n"
            "  {\"role_name\": \"Factory\", \"order_quantity\": <int>, \"rationale\": \"<short reason>\"}\n"
            "]\n"
            "IMPORTANT: output ONLY valid JSON – a list of four objects, one per role, nothing else."
        )

    async def get_recommendations(self, agents: List[BeerGameAgent], external_demand: int,
                                  round_index: int, history: List[Dict[str, Any]] = None,
                                  temperature: float = 0.3) -> Dict[str, Dict[str, Any]]:
        """Call the LLM and return a dictionary keyed by role."""
        if history is None:
            history = []
        prompt = self._build_prompt(agents, external_demand, round_index, history)
        sys_prompt = "You are a top-tier operations research expert coordinating a supply chain."

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
            parsed = safe_parse_json(resp)
        except Exception:
            # Fallback – equal split order 10 each
            parsed = [
                {"role_name": ag.role_name, "order_quantity": 10, "rationale": "default"}
                for ag in agents
            ]

        # Convert list to dict
        recs: Dict[str, Dict[str, Any]] = {}
        for item in parsed:
            role = item.get("role_name")
            if role:
                recs[role] = {
                    "order_quantity": int(item.get("order_quantity", 10)),
                    "rationale": item.get("rationale", "")
                }
        return recs 