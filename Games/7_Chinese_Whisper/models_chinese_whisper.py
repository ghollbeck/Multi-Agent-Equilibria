import asyncio
import json
from pydantic import BaseModel, Field
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, ClassVar
from prompts_chinese_whisper import ChineseWhisperPrompts
from llm_calls_chinese_whisper import lite_client, safe_parse_json, MODEL_NAME

@dataclass
class TransmissionData:
    """
    Records data from each information transmission in the Chinese Whisper chain.
    """
    generation: int
    agent_position: int
    information_type: str
    original_length: int
    processed_length: int
    confidence: float
    changes_made: str
    processing_time: float

@dataclass
class SimulationData:
    hyperparameters: dict
    transmissions_log: List[TransmissionData] = field(default_factory=list)
    original_information: str = ""
    final_information: str = ""

    def add_transmission_entry(self, entry: TransmissionData):
        self.transmissions_log.append(entry)

    def to_dict(self):
        return {
            'hyperparameters': self.hyperparameters,
            'transmissions_log': [asdict(t) for t in self.transmissions_log],
            'original_information': self.original_information,
            'final_information': self.final_information
        }

class ChineseWhisperLogger:
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

class ChineseWhisperAgent(BaseModel):
    agent_position: int
    total_agents: int
    information_received: str = ""
    information_processed: str = ""
    confidence: float = 0.0
    changes_made: str = ""
    key_elements_preserved: List[str] = Field(default_factory=list)
    processing_history: List[dict] = Field(default_factory=list)
    prompts: ClassVar[ChineseWhisperPrompts] = ChineseWhisperPrompts()
    logger: Optional['ChineseWhisperLogger'] = None
    last_processing_prompt: str = ""
    last_processing_output: dict = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    async def process_information(self, received_information: str, information_type: str = "general", 
                                temperature: float = 0.7) -> dict:
        """
        Process received information and reformulate it for the next agent in the chain.
        """
        if self.logger:
            self.logger.log(f"[Agent {self.agent_position}] Processing {information_type} information...")
        
        self.information_received = received_information
        
        if information_type == "factual":
            prompt = self.prompts.get_factual_processing_prompt(
                self.agent_position, self.total_agents, received_information)
        elif information_type == "narrative":
            prompt = self.prompts.get_narrative_processing_prompt(
                self.agent_position, self.total_agents, received_information)
        elif information_type == "technical":
            prompt = self.prompts.get_technical_processing_prompt(
                self.agent_position, self.total_agents, received_information)
        else:
            prompt = self.prompts.get_information_processing_prompt(
                self.agent_position, self.total_agents, received_information, information_type)
        
        self.last_processing_prompt = prompt
        system_prompt = "You are an expert at processing and reformulating information while preserving key elements. Return valid JSON only."
        
        try:
            response_str = await lite_client.chat_completion(
                model=MODEL_NAME,
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=temperature
            )
        except Exception as e:
            print(f"❌ [Agent {self.agent_position}] process_information: LLM call failed. Error: {e}")
            response_str = ''
        
        if self.logger:
            self.logger.log(f"[Agent {self.agent_position}] Processing response: {response_str}")
        
        default_response = {
            "agent_position": self.agent_position,
            "total_agents": self.total_agents,
            "information_type": information_type,
            "processed_information": received_information,  # Fallback to original
            "confidence": 0.5,
            "changes_made": "No changes due to processing error",
            "key_elements_preserved": []
        }
        
        try:
            response = safe_parse_json(response_str)
        except Exception:
            response = default_response
        
        self.information_processed = response.get('processed_information', received_information)
        self.confidence = response.get('confidence', 0.5)
        self.changes_made = response.get('changes_made', '')
        self.key_elements_preserved = response.get('key_elements_preserved', [])
        
        self.processing_history.append({
            'received': received_information,
            'processed': self.information_processed,
            'confidence': self.confidence,
            'changes': self.changes_made
        })
        
        self.last_processing_output = response
        return response

    def get_processed_information(self) -> str:
        """Return the processed information to pass to the next agent."""
        return self.information_processed

    def get_processing_summary(self) -> dict:
        """Return a summary of this agent's processing performance."""
        return {
            'agent_position': self.agent_position,
            'total_agents': self.total_agents,
            'confidence': self.confidence,
            'changes_made': self.changes_made,
            'key_elements_preserved': self.key_elements_preserved,
            'processing_count': len(self.processing_history)
        }
