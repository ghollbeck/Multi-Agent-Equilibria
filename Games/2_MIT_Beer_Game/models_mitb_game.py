import asyncio
import json
from pydantic import BaseModel, Field
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, ClassVar
from prompts_mitb_game import BeerGamePrompts
from llm_calls_mitb_game import lite_client, safe_parse_json, MODEL_NAME

@dataclass
class RoundData:
    """
    Records data from each round of the Beer Game for a single agent.
    """
    generation: int
    round_index: int
    role_name: str
    inventory: int
    backlog: int
    order_placed: int
    shipment_received: int
    shipment_sent_downstream: int
    profit: float

@dataclass
class SimulationData:
    hyperparameters: dict
    rounds_log: List[RoundData] = field(default_factory=list)
    communication_log: List[Dict] = field(default_factory=list)

    def add_round_entry(self, entry: RoundData):
        self.rounds_log.append(entry)

    def add_communication_entry(self, entry: Dict):
        self.communication_log.append(entry)

    def to_dict(self):
        return {
            'hyperparameters': self.hyperparameters,
            'rounds_log': [asdict(r) for r in self.rounds_log],
            'communication_log': self.communication_log
        }

class BeerGameLogger:
    def __init__(self, log_to_file=None):
        self.logs = []
        self.log_to_file = log_to_file
        self.file_handle = open(log_to_file, 'a') if log_to_file else None

    def log(self, msg):
        self.logs.append(msg)
        print(msg)
        if self.file_handle:
            self.file_handle.write(msg + '\n')

    def get_logs(self):
        return self.logs

    def close(self):
        if self.file_handle:
            self.file_handle.close()

class BeerGameAgent(BaseModel):
    role_name: str  # "Retailer", "Wholesaler", "Distributor", "Factory"
    inventory: int = 100
    backlog: int = 0
    profit_accumulated: float = 0.0
    last_profit: Optional[float] = None
    last_order_placed: Optional[int] = None
    shipments_in_transit: Dict[int,int] = Field(default_factory=lambda: {0:10, 1:10})
    downstream_orders_history: List[int] = Field(default_factory=list)
    strategy: dict = Field(default_factory=dict)
    prompts: ClassVar[BeerGamePrompts] = BeerGamePrompts
    logger: BeerGameLogger = None
    last_decision_prompt: str = ""
    last_decision_output: dict = Field(default_factory=dict)
    last_update_prompt: str = ""
    last_update_output: dict = Field(default_factory=dict)
    last_init_prompt: str = ""
    last_init_output: dict = Field(default_factory=dict)
    message_history: List[Dict[str, str]] = Field(default_factory=list)
    last_communication_prompt: str = ""
    last_communication_output: dict = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    async def initialize_strategy(self, temperature=0.7, profit_per_unit_sold=5):
        if self.logger:
            self.logger.log(f"[Agent {self.role_name}] Initializing strategy...")
        prompt = self.prompts.get_strategy_generation_prompt(
            self.role_name, self.inventory, self.backlog, profit_per_unit_sold)
        self.last_init_prompt = prompt
        system_prompt = "You are an expert supply chain manager. Return valid JSON only."
        try:
            response_str = await lite_client.chat_completion(
                model=MODEL_NAME,
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=temperature
            )
        except Exception as e:
            print(f"❌ [Agent {self.role_name}] initialize_strategy: LLM call failed. Error: {e}")
            response_str = ''
        if self.logger:
            self.logger.log(f"[Agent {self.role_name}] Strategy response: {response_str}")
        default_strategy = {
            "order_quantity": 10,
            "confidence": 1.0,
            "rationale": "Default initial strategy",
            "risk_assessment": "No risk",
            "expected_demand_next_round": 10
        }
        try:
            response = safe_parse_json(response_str)
        except Exception:
            response = default_strategy
        self.strategy = response
        self.last_init_output = response

    async def update_strategy(self, performance_log: str, temperature=0.7, profit_per_unit_sold=5):
        if self.logger:
            self.logger.log(f"[Agent {self.role_name}] Updating strategy with performance log: {performance_log}")
        prompt = self.prompts.get_strategy_update_prompt(
            self.role_name, performance_log, self.strategy,
            self.inventory, self.backlog, profit_per_unit_sold
        )
        self.last_update_prompt = prompt
        system_prompt = "You are an expert supply chain manager. Return valid JSON only."
        try:
            response_str = await lite_client.chat_completion(
                model=MODEL_NAME,
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=temperature
            )
        except Exception as e:
            print(f"❌ [Agent {self.role_name}] update_strategy: LLM call failed. Error: {e}")
            response_str = ''
        if self.logger:
            self.logger.log(f"[Agent {self.role_name}] Update response: {response_str}")
        default_update = {
            "order_quantity": 10,
            "confidence": 1.0,
            "rationale": "Default update strategy",
            "risk_assessment": "No risk",
            "expected_demand_next_round": 10
        }
        try:
            response = safe_parse_json(response_str)
        except Exception:
            response = default_update
        self.strategy = response
        self.last_update_output = response

    async def decide_order_quantity(self, temperature=0.7, profit_per_unit_sold=5) -> dict:
        if self.logger:
            self.logger.log(f"[Agent {self.role_name}] Deciding order quantity. Inventory: {self.inventory}, Backlog: {self.backlog}, Downstream: {self.downstream_orders_history[-3:]}, Shipments: {[self.shipments_in_transit[1]]}")
        last_order_placed = self.last_order_placed
        last_profit = self.last_profit
        prompt = self.prompts.get_decision_prompt(
            role_name=self.role_name,
            inventory=self.inventory,
            backlog=self.backlog,
            recent_demand_or_orders=self.downstream_orders_history[-3:],
            incoming_shipments=[self.shipments_in_transit[1]],
            current_strategy=self.strategy,
            profit_per_unit_sold=profit_per_unit_sold,
            last_order_placed=last_order_placed,
            last_profit=last_profit
        )
        self.last_decision_prompt = prompt
        system_prompt = "You are an expert supply chain manager. Return valid JSON only."
        try:
            response_str = await lite_client.chat_completion(
                model=MODEL_NAME,
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=temperature
            )
        except Exception as e:
            print(f"❌ [Agent {self.role_name}] decide_order_quantity: LLM call failed. Error: {e}")
            response_str = ''
        if self.logger:
            self.logger.log(f"[Agent {self.role_name}] Decision response: {response_str}")
        default_decision = {
            "order_quantity": 10,
            "confidence": 1.0,
            "rationale": "Default decision",
            "risk_assessment": "No risk",
            "expected_demand_next_round": 10
        }
        try:
            response = safe_parse_json(response_str)
        except Exception:
            response = default_decision
        self.last_decision_output = response
        self.last_profit = response.get('profit', None)
        return response

    async def generate_communication_message(self, round_index: int, other_agents: List['BeerGameAgent'], 
                                          message_history: List[Dict], temperature: float = 0.7) -> dict:
        """Generate a message to communicate with other agents about strategy and collaboration."""
        if self.logger:
            self.logger.log(f"[Agent {self.role_name}] Generating communication message for round {round_index}")
        
        prompt = self.prompts.get_communication_prompt(
            role_name=self.role_name,
            inventory=self.inventory,
            backlog=self.backlog,
            recent_demand_or_orders=self.downstream_orders_history[-3:],
            current_strategy=self.strategy,
            message_history=message_history,
            other_agent_roles=[agent.role_name for agent in other_agents],
            round_index=round_index,
            last_order_placed=self.last_order_placed,
            profit_accumulated=self.profit_accumulated
        )
        
        self.last_communication_prompt = prompt
        system_prompt = "You are an expert supply chain manager. Return valid JSON only."
        
        try:
            response_str = await lite_client.chat_completion(
                model=MODEL_NAME,
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=temperature
            )
        except Exception as e:
            print(f"❌ [Agent {self.role_name}] generate_communication_message: LLM call failed. Error: {e}")
            response_str = ''
        
        default_message = {
            "message": f"Hello from {self.role_name}. Let's work together efficiently.",
            "strategy_hint": "Maintaining steady orders",
            "collaboration_proposal": "Share demand information",
            "information_shared": "Current inventory and backlog status",
            "confidence": 0.5
        }
        
        try:
            response = safe_parse_json(response_str)
        except Exception:
            response = default_message
        
        self.last_communication_output = response
        return response

    async def decide_order_quantity_with_communication(self, temperature: float = 0.7, 
                                                     profit_per_unit_sold: float = 5,
                                                     recent_communications: List[Dict] = None) -> dict:
        """Enhanced decision making that incorporates communication messages."""
        if recent_communications:
            prompt = self.prompts.get_decision_prompt_with_communication(
                role_name=self.role_name,
                inventory=self.inventory,
                backlog=self.backlog,
                recent_demand_or_orders=self.downstream_orders_history[-3:],
                incoming_shipments=[self.shipments_in_transit[1]],
                current_strategy=self.strategy,
                profit_per_unit_sold=profit_per_unit_sold,
                last_order_placed=self.last_order_placed,
                last_profit=self.last_profit,
                recent_communications=recent_communications
            )
            
            self.last_decision_prompt = prompt
            system_prompt = "You are an expert supply chain manager. Return valid JSON only."
            
            try:
                response_str = await lite_client.chat_completion(
                    model=MODEL_NAME,
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    temperature=temperature
                )
            except Exception as e:
                print(f"❌ [Agent {self.role_name}] decide_order_quantity_with_communication: LLM call failed. Error: {e}")
                response_str = ''
            
            default_decision = {
                "order_quantity": 10,
                "confidence": 1.0,
                "rationale": "Default decision with communication context",
                "risk_assessment": "No risk",
                "expected_demand_next_round": 10
            }
            
            try:
                response = safe_parse_json(response_str)
            except Exception:
                response = default_decision
            
            self.last_decision_output = response
            self.last_profit = response.get('profit', None)
            return response
        else:
            return await self.decide_order_quantity(temperature, profit_per_unit_sold)      