import asyncio
import json
from pydantic import BaseModel, Field
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, ClassVar, Literal, Any
from prompts_mitb_game import BeerGamePrompts
from llm_calls_mitb_game import lite_client, safe_parse_json, MODEL_NAME, get_default_client
from memory_storage import AgentMemory

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
    order_received: int  # New orders received from downstream customers this round
    shipment_received: int
    shipment_sent_downstream: int
    starting_balance: float
    revenue: float
    purchase_cost: float
    holding_cost: float
    backlog_cost: float
    ending_balance: float
    # Orchestrator recommendation
    orchestrator_order: int = 0
    orchestrator_rationale: str = ""
    # NEW: Track order and production delays
    orders_in_transit_0: int = 0  # Orders arriving this round
    orders_in_transit_1: int = 0  # Orders arriving next round
    production_queue_0: int = 0   # Production completing this round (Factory only)
    production_queue_1: int = 0   # Production completing next round (Factory only)

@dataclass
class SimulationData:
    hyperparameters: dict
    rounds_log: List[RoundData] = field(default_factory=list)
    communication_log: List[Dict] = field(default_factory=list)
    aggregated_rounds: List[Dict] = field(default_factory=list)

    def add_round_entry(self, entry: RoundData):
        self.rounds_log.append(entry)

    def add_communication_entry(self, entry: Dict):
        self.communication_log.append(entry)

    def add_aggregated_round(self, round_dict: Dict):
        """Store combined information for one round (agents + comms)."""
        self.aggregated_rounds.append(round_dict)

    def to_dict(self):
        return {
            'hyperparameters': self.hyperparameters,
            'rounds_log': [asdict(r) for r in self.rounds_log],
            'communication_log': self.communication_log,
            'aggregated_rounds': self.aggregated_rounds
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
    balance: float = 1000.0  # Bank account balance starts at $1000
    last_profit: Optional[float] = None
    last_order_placed: Optional[int] = None
    shipments_in_transit: Dict[int,int] = Field(default_factory=lambda: {0:0, 1:0})
    orders_in_transit: Dict[int,int] = Field(default_factory=lambda: {0:0, 1:0})  # NEW: Order delay pipeline
    production_queue: Dict[int,int] = Field(default_factory=lambda: {0:0, 1:0})  # NEW: Factory production delay
    downstream_orders_history: List[int] = Field(default_factory=list)
    # NEW: Track profit and balance history over rounds
    profit_history: List[float] = Field(default_factory=list)
    balance_history: List[float] = Field(default_factory=list)
    strategy: dict = Field(default_factory=dict)
    prompts: ClassVar[BeerGamePrompts] = BeerGamePrompts
    logger: BeerGameLogger = Field(default=None, exclude=True)
    human_log_file: Optional[Any] = Field(default=None, exclude=True)  # NEW: For real-time logging
    last_decision_prompt: str = ""
    last_decision_output: dict = Field(default_factory=dict)
    last_update_prompt: str = ""
    last_update_output: dict = Field(default_factory=dict)
    last_init_prompt: str = ""
    last_init_output: dict = Field(default_factory=dict)
    message_history: List[Dict[str, str]] = Field(default_factory=list)
    last_communication_prompt: str = ""
    last_communication_output: dict = Field(default_factory=dict)
    # Store system prompts for logging
    last_decision_system_prompt: str = ""
    last_update_system_prompt: str = ""
    last_init_system_prompt: str = ""
    last_communication_system_prompt: str = ""
    memory: Optional[AgentMemory] = None

    @classmethod
    def create_agent(cls, role_name: str, initial_inventory: int = 100, initial_backlog: int = 0, 
                    initial_balance: float = 1000.0, logger: BeerGameLogger = None, human_log_file: Optional[Any] = None):
        """Create a BeerGameAgent with configurable initial values."""
        return cls(
            role_name=role_name,
            inventory=initial_inventory,
            backlog=initial_backlog,
            balance=initial_balance,
            shipments_in_transit={0: 0, 1: 0},  # Start with no shipments in transit
            orders_in_transit={0: 0, 1: 0},    # NEW: Start with no orders in transit
            production_queue={0: 0, 1: 0},     # NEW: Start with no production in queue
            logger=logger,
            human_log_file=human_log_file
        )

    class Config:
        arbitrary_types_allowed = True

    def log_llm_call_immediately(self, call_type: str, system_prompt: str, user_prompt: str, model_output: dict, round_index: int = None):
        """Log LLM call immediately to human-readable log file."""
        if not self.human_log_file:
            return
        
        try:
            self.human_log_file.write(f"\n🤖 LLM {call_type.upper()} CALL - {self.role_name}")
            if round_index is not None:
                self.human_log_file.write(f" (Round {round_index})")
            self.human_log_file.write("\n")
            self.human_log_file.write("─" * 60 + "\n")
            
            if system_prompt:
                self.human_log_file.write("🔧 System Prompt:\n")
                self.human_log_file.write(f"{system_prompt}\n\n")
            
            if user_prompt:
                self.human_log_file.write("👤 User Prompt:\n")
                self.human_log_file.write(f"{user_prompt}\n\n")
            
            if model_output:
                self.human_log_file.write("🎯 Model Output:\n")
                self.human_log_file.write(f"{json.dumps(model_output, indent=2)}\n\n")
            
            self.human_log_file.write("─" * 60 + "\n\n")
            self.human_log_file.flush()  # Ensure immediate write
        except Exception as e:
            if self.logger:
                self.logger.log(f"Error logging LLM call for {self.role_name}: {e}")

    async def llm_decision(self, phase: Literal["decision", "communication"], *,
                          comm_history: Optional[list] = None,
                          orchestrator_advice: str = None,
                          history_limit: int = 20,
                          temperature: float = 0.7,
                          selling_price_per_unit: float = None,
                          unit_cost_per_unit: float = None,
                          other_agent_roles: Optional[list] = None,
                          round_index: int = 0,
                          longtermplanning_boolean: bool = False,
                          profit_per_unit_sold: float = None,  # Deprecated - for backward compatibility
                          **kwargs) -> dict:
        """Unified gateway to the LLM – returns JSON dict."""
        from prompts_mitb_game import AgentContext, PromptEngine  # local import to avoid circular

        ctx_kwargs = dict(
            inventory=self.inventory,
            backlog=self.backlog,
            recent_demand_or_orders=self.downstream_orders_history[-history_limit:],
            incoming_shipments=[self.shipments_in_transit[1]],
            current_strategy=self.strategy,
            selling_price_per_unit=selling_price_per_unit,
            unit_cost_per_unit=unit_cost_per_unit,
            holding_cost_per_unit=kwargs.get('holding_cost_per_unit'),
            backlog_cost_per_unit=kwargs.get('backlog_cost_per_unit'),
            last_order_placed=self.last_order_placed,
            last_profit=self.last_profit,
            profit_history=self.profit_history,
            balance_history=self.balance_history,
            current_balance=self.balance,
            total_chain_inventory=kwargs.get('total_chain_inventory'),
            total_chain_backlog=kwargs.get('total_chain_backlog'),
            # Additional financial metrics
            cumulative_profit=sum(self.profit_history) if self.profit_history else 0.0,
            # Add hyperparameters
            safety_stock_target=kwargs.get('safety_stock_target'),
            backlog_clearance_rate=kwargs.get('backlog_clearance_rate'),
            demand_smoothing_factor=kwargs.get('demand_smoothing_factor'),
            # Keep for backward compatibility
            profit_per_unit_sold=profit_per_unit_sold
        )
        ctx = AgentContext(**ctx_kwargs)

        prompt = PromptEngine.build_prompt(
            role_name=self.role_name,
            phase=phase,
            ctx=ctx,
            comm_history=comm_history,
            orchestrator_advice=orchestrator_advice,
            history_limit=history_limit,
            other_agent_roles=other_agent_roles,
            round_index=round_index,
            longtermplanning_boolean=longtermplanning_boolean
        )

        system_prompt = self.prompts.get_system_prompt(self.role_name, enable_communication=(phase=="communication"))
        decision_type = f"{phase}_prompt"

        client = lite_client or get_default_client()
        # Get the current model name from the module
        from llm_calls_mitb_game import MODEL_NAME as current_model
        response_str = await client.chat_completion(
            model=current_model,
            system_prompt=system_prompt,
            user_prompt=prompt,
            temperature=temperature,
            agent_role=self.role_name,
            decision_type=decision_type
        )

        try:
            response = safe_parse_json(response_str)
        except Exception:
            response = {}

        # bookkeeping
        if phase == "decision":
            self.last_decision_prompt = prompt
            self.last_decision_system_prompt = system_prompt
            self.last_decision_output = response
        else:
            self.last_communication_prompt = prompt
            self.last_communication_system_prompt = system_prompt
            self.last_communication_output = response

        # Log immediately to human-readable file
        self.log_llm_call_immediately(phase, system_prompt, prompt, response, round_index)

        return response

    # ------------------------------
    # Deprecated wrappers (kept for compatibility)
    # ------------------------------
    async def initialize_strategy(self, *args, **kwargs):
        """Deprecated – use llm_decision('decision')."""
        return await self.llm_decision("decision", 
                                      temperature=kwargs.get('temperature', 0.7),
                                      selling_price_per_unit=kwargs.get('selling_price_per_unit'),
                                      unit_cost_per_unit=kwargs.get('unit_cost_per_unit'),
                                      holding_cost_per_unit=kwargs.get('holding_cost_per_unit'),
                                      backlog_cost_per_unit=kwargs.get('backlog_cost_per_unit'),
                                      safety_stock_target=kwargs.get('safety_stock_target'),
                                      backlog_clearance_rate=kwargs.get('backlog_clearance_rate'),
                                      demand_smoothing_factor=kwargs.get('demand_smoothing_factor'))

    async def update_strategy(self, *args, **kwargs):
        """Deprecated – use llm_decision('decision')."""
        return await self.llm_decision("decision", temperature=kwargs.get('temperature',0.7))

    async def decide_order_quantity(self, temperature=0.7, selling_price_per_unit=None, unit_cost_per_unit=None, holding_cost_per_unit=None, backlog_cost_per_unit=None, round_index=None, longtermplanning_boolean=False, profit_per_unit_sold=None, safety_stock_target=None, backlog_clearance_rate=None, demand_smoothing_factor=None, history_limit=20) -> dict:
        if self.logger:
            self.logger.log(f"[Agent {self.role_name}] Deciding order quantity. Inventory: {self.inventory}, Backlog: {self.backlog}, Downstream: {self.downstream_orders_history[-history_limit:]}, Shipments: {[self.shipments_in_transit[1]]}, Orders arriving: {[self.orders_in_transit[0]]}")
        last_order_placed = self.last_order_placed
        last_profit = self.last_profit
        prompt = self.prompts.get_decision_prompt(
            role_name=self.role_name,
            inventory=self.inventory,
            backlog=self.backlog,
            recent_demand_or_orders=self.downstream_orders_history[-history_limit:],
            incoming_shipments=[self.shipments_in_transit[1]],
            current_strategy=self.strategy,
            selling_price_per_unit=selling_price_per_unit,
            unit_cost_per_unit=unit_cost_per_unit,
            holding_cost_per_unit=holding_cost_per_unit,
            backlog_cost_per_unit=backlog_cost_per_unit,
            last_order_placed=last_order_placed,
            last_profit=last_profit,
            profit_history=self.profit_history,
            balance_history=self.balance_history,
            current_balance=self.balance,
            round_index=round_index,
            longtermplanning_boolean=longtermplanning_boolean,
            safety_stock_target=safety_stock_target,
            backlog_clearance_rate=backlog_clearance_rate,
            demand_smoothing_factor=demand_smoothing_factor
        )
        self.last_decision_prompt = prompt
        system_prompt = self.prompts.get_system_prompt(self.role_name)
        self.last_decision_system_prompt = system_prompt
        # if self.logger:
        #     self.logger.log(f"[LLM SYSTEM PROMPT]: {system_prompt}")
        #     self.logger.log(f"[LLM USER PROMPT]: {prompt}")
        try:
            client = lite_client or get_default_client()
            # Get the current model name from the module
            from llm_calls_mitb_game import MODEL_NAME as current_model
            response_str = await client.chat_completion(
                model=current_model,
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=temperature,
                agent_role=self.role_name,
                decision_type="order_decision"
            )
        except Exception as e:
            # print(f"❌ [Agent {self.role_name}] decide_order_quantity: LLM call failed. Error: {e}")  # Commented out
            response_str = "{}"
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
        
        # Log immediately to human-readable file
        self.log_llm_call_immediately("decision", system_prompt, prompt, response)
        
        return response

    async def generate_communication_message(self, round_index: int, other_agents: List['BeerGameAgent'], 
                                           message_history: List[Dict], temperature: float = 0.7, longtermplanning_boolean: bool = False, 
                                           selling_price_per_unit: float = None, unit_cost_per_unit: float = None,
                                           holding_cost_per_unit: float = None, backlog_cost_per_unit: float = None,
                                           profit_per_unit_sold: float = None) -> dict:
        """Deprecated wrapper – now delegates to llm_decision('communication')."""
        return await self.llm_decision(
            "communication",
            comm_history=message_history,
            other_agent_roles=[a.role_name for a in other_agents],
            round_index=round_index,
            temperature=temperature,
            selling_price_per_unit=selling_price_per_unit,
            unit_cost_per_unit=unit_cost_per_unit,
            holding_cost_per_unit=holding_cost_per_unit,
            backlog_cost_per_unit=backlog_cost_per_unit,
            longtermplanning_boolean=longtermplanning_boolean
        )

    async def decide_order_quantity_with_communication(self, temperature: float = 0.7, 
                                                     selling_price_per_unit: float = None,
                                                     unit_cost_per_unit: float = None,
                                                     holding_cost_per_unit: float = None,
                                                     backlog_cost_per_unit: float = None,
                                                     round_index: int = None,
                                                     profit_per_unit_sold: float = None,
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
                selling_price_per_unit=selling_price_per_unit,
                unit_cost_per_unit=unit_cost_per_unit,
                holding_cost_per_unit=holding_cost_per_unit,
                backlog_cost_per_unit=backlog_cost_per_unit,
                profit_per_unit_sold=profit_per_unit_sold,
                last_order_placed=self.last_order_placed,
                last_profit=self.last_profit,
                recent_communications=recent_communications,
                profit_history=self.profit_history,
                balance_history=self.balance_history,
                current_balance=self.balance,
                round_index=round_index
            )
            
            self.last_decision_prompt = prompt
            system_prompt = self.prompts.get_system_prompt(self.role_name, enable_communication=True)
            self.last_decision_system_prompt = system_prompt
            
            # if self.logger:
            #     self.logger.log(f"[LLM SYSTEM PROMPT]: {system_prompt}")
            #     self.logger.log(f"[LLM USER PROMPT]: {prompt}")
            
            try:
                response_str = await lite_client.chat_completion(
                    model=MODEL_NAME,
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    temperature=temperature,
                    agent_role=self.role_name,
                    decision_type="order_decision_with_communication"
                )
            except Exception as e:
                # print(f"❌ [Agent {self.role_name}] decide_order_quantity_with_communication: LLM call failed. Error: {e}")  # Commented out
                response_str = "{}"
            
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
    
    def load_memory_context(self) -> Dict[str, str]:
        """Retrieve relevant past experiences from memory if available."""
        if not self.memory:
            return {
                'decision_context': 'No memory context available.',
                'communication_context': 'No memory context available.'
            }
        
        return {
            'decision_context': self.memory.get_memory_context_for_decision(),
            'communication_context': self.memory.get_memory_context_for_communication()
        }
    
    def update_memory(self, round_number: int, decision_output: Dict[str, any] = None, 
                     communication_output: Dict[str, any] = None, 
                     performance_data: Dict[str, any] = None) -> None:
        """Store new experiences in memory after each decision."""
        if not self.memory:
            return
        
        if decision_output:
            self.memory.add_decision_memory(
                round_number=round_number,
                order_quantity=decision_output.get('order_quantity', 0),
                inventory=self.inventory,
                backlog=self.backlog,
                reasoning=decision_output.get('rationale', ''),
                confidence=decision_output.get('confidence', 0.5)
            )
            
            if 'strategy_description' in decision_output or self.strategy:
                self.memory.add_strategy_memory(
                    round_number=round_number,
                    strategy_description=decision_output.get('strategy_description', str(self.strategy)),
                    strategy_rationale=decision_output.get('rationale', ''),
                    expected_outcome=decision_output.get('expected_demand_next_round', 'Unknown')
                )
        
        if communication_output:
            self.memory.add_communication_memory(
                round_number=round_number,
                message=communication_output.get('message', ''),
                strategy_hint=communication_output.get('strategy_hint', ''),
                collaboration_proposal=communication_output.get('collaboration_proposal', ''),
                information_shared=communication_output.get('information_shared', '')
            )
        
        if performance_data:
            self.memory.add_performance_memory(
                round_number=round_number,
                profit=performance_data.get('profit', self.balance),
                inventory=self.inventory,
                backlog=self.backlog,
                units_sold=performance_data.get('units_sold', 0),
                holding_cost=performance_data.get('holding_cost', 0.0),
                backlog_cost=performance_data.get('backlog_cost', 0.0)
            )               

    def update_profit_history(self, round_profit: float, new_balance: float):
        """Update profit and balance history after each round"""
        self.profit_history.append(round_profit)
        self.balance_history.append(new_balance)
        # Keep only recent history to prevent memory bloat (last 20 rounds)
        if len(self.profit_history) > 20:
            self.profit_history = self.profit_history[-20:]
        if len(self.balance_history) > 20:
            self.balance_history = self.balance_history[-20:]

    def get_profit_trend(self) -> str:
        """Get a summary of recent profit trends for agent decision making"""
        if len(self.profit_history) < 2:
            return "Insufficient history"
        
        recent_profits = self.profit_history[-3:]  # Last 3 rounds
        if all(p > 0 for p in recent_profits):
            return "Profitable trend"
        elif all(p < 0 for p in recent_profits):
            return "Loss trend"
        elif recent_profits[-1] > recent_profits[0]:
            return "Improving"
        else:
            return "Declining"

    def get_balance_trend(self) -> str:
        """Get a summary of recent balance trends for agent decision making"""
        if len(self.balance_history) < 2:
            return "Insufficient history"
        
        recent_balances = self.balance_history[-3:]  # Last 3 rounds
        if recent_balances[-1] > recent_balances[0]:
            return "Growing balance"
        elif recent_balances[-1] < recent_balances[0]:
            return "Declining balance"
        else:
            return "Stable balance"

    # Compatibility alias so existing code using profit_accumulated continues to work
    @property
    def profit_accumulated(self):
        return self.balance

    @profit_accumulated.setter
    def profit_accumulated(self, value):
        self.balance = value               