import json
from typing import List, Dict

class ChineseWhisperPrompts:
    """
    Provides system and user prompts for the Chinese Whisper Game.
    Each agent will use these to process and reformulate information
    as it passes through the chain of agents.
    """

    @staticmethod
    def get_information_processing_prompt(agent_position: int, total_agents: int, 
                                        received_information: str, information_type: str = "general") -> str:
        return f"""
        You are Agent {agent_position} in a chain of {total_agents} agents participating in a Chinese Whisper game.
        Your task is to process and reformulate the information you received, then pass it to the next agent.

        Information Type: {information_type}
        Your Position: {agent_position} of {total_agents}
        
        Information you received:
        {received_information}

        Instructions:
        • Read and understand the information carefully
        • Reformulate it in your own words while preserving the core meaning
        • You may naturally introduce small changes as information passes through you
        • Focus on maintaining the essential facts, narrative elements, or key points
        • Keep the same general length and structure as the original

        Please return only valid JSON with the following fields:

        {{
          "agent_position": {agent_position},
          "total_agents": {total_agents},
          "information_type": "{information_type}",
          "processed_information": "<your reformulated version of the information>",
          "confidence": <float between 0 and 1 indicating how confident you are in preserving the original meaning>,
          "changes_made": "<brief description of any changes you made>",
          "key_elements_preserved": ["<list of key elements you tried to preserve>"]
        }}

        IMPORTANT: Output ONLY the JSON object, with no markdown, no triple backticks, no code fences, and do NOT write the word 'json' anywhere. Your reply must be a single valid JSON object, nothing else.
        """

    @staticmethod
    def get_factual_processing_prompt(agent_position: int, total_agents: int, 
                                    received_information: str) -> str:
        return f"""
        You are Agent {agent_position} in a chain of {total_agents} agents participating in a Chinese Whisper game focused on factual information.
        Your task is to process and reformulate the factual information you received.

        Your Position: {agent_position} of {total_agents}
        
        Factual information you received:
        {received_information}

        Instructions:
        • Carefully preserve all specific facts: names, dates, numbers, locations
        • Reformulate the presentation while maintaining factual accuracy
        • If you're uncertain about a fact, indicate your uncertainty
        • Focus on preserving quantitative data and proper nouns

        Please return only valid JSON with the following fields:

        {{
          "agent_position": {agent_position},
          "total_agents": {total_agents},
          "information_type": "factual",
          "processed_information": "<your reformulated version preserving all facts>",
          "confidence": <float between 0 and 1>,
          "facts_preserved": ["<list of specific facts you preserved>"],
          "uncertainties": ["<list of any facts you were uncertain about>"],
          "changes_made": "<description of reformulation changes made>"
        }}

        IMPORTANT: Output ONLY the JSON object, with no markdown, no triple backticks, no code fences.
        """

    @staticmethod
    def get_narrative_processing_prompt(agent_position: int, total_agents: int, 
                                      received_information: str) -> str:
        return f"""
        You are Agent {agent_position} in a chain of {total_agents} agents participating in a Chinese Whisper game focused on narrative stories.
        Your task is to retell the story you received in your own words.

        Your Position: {agent_position} of {total_agents}
        
        Story you received:
        {received_information}

        Instructions:
        • Retell the story maintaining the main plot points
        • Preserve character names and key events
        • You may naturally change some details in your retelling
        • Keep the story engaging and coherent
        • Maintain the general timeline and sequence of events

        Please return only valid JSON with the following fields:

        {{
          "agent_position": {agent_position},
          "total_agents": {total_agents},
          "information_type": "narrative",
          "processed_information": "<your retelling of the story>",
          "confidence": <float between 0 and 1>,
          "plot_points_preserved": ["<key plot points you maintained>"],
          "characters_mentioned": ["<character names you included>"],
          "changes_made": "<description of changes in your retelling>"
        }}

        IMPORTANT: Output ONLY the JSON object, with no markdown, no triple backticks, no code fences.
        """

    @staticmethod
    def get_technical_processing_prompt(agent_position: int, total_agents: int, 
                                      received_information: str) -> str:
        return f"""
        You are Agent {agent_position} in a chain of {total_agents} agents participating in a Chinese Whisper game focused on technical instructions.
        Your task is to reformulate the technical instructions you received.

        Your Position: {agent_position} of {total_agents}
        
        Technical instructions you received:
        {received_information}

        Instructions:
        • Preserve the sequence and logic of steps
        • Maintain technical accuracy and precision
        • Keep all specific parameters, settings, or measurements
        • Reformulate for clarity while preserving completeness
        • Ensure the instructions remain actionable

        Please return only valid JSON with the following fields:

        {{
          "agent_position": {agent_position},
          "total_agents": {total_agents},
          "information_type": "technical",
          "processed_information": "<your reformulated technical instructions>",
          "confidence": <float between 0 and 1>,
          "steps_preserved": <number of steps you maintained>,
          "technical_terms_preserved": ["<list of technical terms you kept>"],
          "changes_made": "<description of reformulation changes>"
        }}

        IMPORTANT: Output ONLY the JSON object, with no markdown, no triple backticks, no code fences.
        """
