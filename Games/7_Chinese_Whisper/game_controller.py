import asyncio
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import asdict
from models_chinese_whisper import ChineseWhisperAgent, SimulationData, TransmissionData, ChineseWhisperLogger
from information_generator import InformationGenerator, InformationSeed
from evaluation_engine import EvaluationEngine, EvaluationMetrics

class ChineseWhisperGameController:
    """
    Main controller for orchestrating Chinese Whisper game simulations.
    Manages agent chains, information flow, and evaluation processes.
    """
    
    def __init__(self, logger: Optional[ChineseWhisperLogger] = None):
        self.logger = logger or ChineseWhisperLogger()
        self.info_generator = InformationGenerator()
        self.evaluation_engine = EvaluationEngine()
        self.simulation_data = None
    
    async def run_single_chain(self, 
                              num_agents: int,
                              information_seed: InformationSeed,
                              model_name: str = "gpt-4o-mini",
                              temperature: float = 0.7,
                              generation_id: int = 1) -> Dict[str, Any]:
        """
        Run a single chain of agents processing information.
        
        Args:
            num_agents: Number of agents in the chain
            information_seed: Initial information to process
            model_name: LLM model to use
            temperature: Temperature for LLM calls
            generation_id: Identifier for this generation
            
        Returns:
            Dictionary containing chain results and metrics
        """
        if self.logger:
            self.logger.log(f"[Game Controller] Starting chain with {num_agents} agents")
            self.logger.log(f"[Game Controller] Information type: {information_seed.information_type}")
            self.logger.log(f"[Game Controller] Original length: {len(information_seed.content)} characters")
        
        agents = []
        for i in range(num_agents):
            agent = ChineseWhisperAgent(
                agent_position=i + 1,
                total_agents=num_agents,
                logger=self.logger
            )
            agents.append(agent)
        
        current_information = information_seed.content
        chain_data = []
        transmission_log = []
        
        for i, agent in enumerate(agents):
            start_time = time.time()
            
            if self.logger:
                self.logger.log(f"[Game Controller] Agent {i+1} processing information...")
            
            result = await agent.process_information(
                received_information=current_information,
                information_type=information_seed.information_type,
                temperature=temperature
            )
            
            processing_time = time.time() - start_time
            
            processed_info = agent.get_processed_information()
            chain_data.append(processed_info)
            
            transmission_entry = TransmissionData(
                generation=generation_id,
                agent_position=i + 1,
                information_type=information_seed.information_type,
                original_length=len(current_information),
                processed_length=len(processed_info),
                confidence=result.get('confidence', 0.0),
                changes_made=result.get('changes_made', ''),
                processing_time=processing_time
            )
            transmission_log.append(transmission_entry)
            
            current_information = processed_info
            
            if self.logger:
                self.logger.log(f"[Game Controller] Agent {i+1} completed. Confidence: {result.get('confidence', 0.0):.2f}")
        
        evaluation_metrics = self.evaluation_engine.evaluate_chain(
            original_info=information_seed.content,
            chain_data=chain_data,
            key_elements=information_seed.expected_key_elements,
            info_type=information_seed.information_type
        )
        
        threshold_analysis = self.evaluation_engine.analyze_critical_thresholds(
            chain_data=chain_data,
            original_info=information_seed.content
        )
        
        if self.logger:
            self.logger.log(f"[Game Controller] Chain completed. Final retention: {evaluation_metrics.retention_score:.2f}")
            self.logger.log(f"[Game Controller] Semantic similarity: {evaluation_metrics.semantic_similarity:.2f}")
            self.logger.log(f"[Game Controller] Factual accuracy: {evaluation_metrics.factual_accuracy:.2f}")
        
        return {
            'generation_id': generation_id,
            'num_agents': num_agents,
            'model_name': model_name,
            'temperature': temperature,
            'information_seed': asdict(information_seed),
            'chain_data': chain_data,
            'transmission_log': [asdict(t) for t in transmission_log],
            'evaluation_metrics': asdict(evaluation_metrics),
            'threshold_analysis': threshold_analysis,
            'agent_summaries': [agent.get_processing_summary() for agent in agents]
        }
    
    async def run_batch_experiment(self,
                                  agent_counts: List[int],
                                  information_types: List[str],
                                  complexity_levels: List[str],
                                  model_names: List[str],
                                  temperatures: List[float],
                                  runs_per_config: int = 1) -> Dict[str, Any]:
        """
        Run a batch of experiments with different configurations.
        
        Args:
            agent_counts: List of agent counts to test
            information_types: List of information types to test
            complexity_levels: List of complexity levels to test
            model_names: List of model names to test
            temperatures: List of temperatures to test
            runs_per_config: Number of runs per configuration
            
        Returns:
            Dictionary containing all experiment results
        """
        if self.logger:
            self.logger.log("[Game Controller] Starting batch experiment")
        
        all_results = []
        generation_id = 1
        
        total_configs = (len(agent_counts) * len(information_types) * 
                        len(complexity_levels) * len(model_names) * 
                        len(temperatures) * runs_per_config)
        
        if self.logger:
            self.logger.log(f"[Game Controller] Total configurations to run: {total_configs}")
        
        for agent_count in agent_counts:
            for info_type in information_types:
                for complexity in complexity_levels:
                    for model_name in model_names:
                        for temperature in temperatures:
                            for run in range(runs_per_config):
                                if self.logger:
                                    self.logger.log(f"[Game Controller] Config {generation_id}/{total_configs}: "
                                                  f"agents={agent_count}, type={info_type}, "
                                                  f"complexity={complexity}, model={model_name}, "
                                                  f"temp={temperature}, run={run+1}")
                                
                                info_seed = self.info_generator.generate_information(
                                    info_type=info_type,
                                    complexity=complexity
                                )
                                
                                try:
                                    result = await self.run_single_chain(
                                        num_agents=agent_count,
                                        information_seed=info_seed,
                                        model_name=model_name,
                                        temperature=temperature,
                                        generation_id=generation_id
                                    )
                                    all_results.append(result)
                                    
                                except Exception as e:
                                    if self.logger:
                                        self.logger.log(f"[Game Controller] Error in generation {generation_id}: {e}")
                                    continue
                                
                                generation_id += 1
        
        batch_summary = self._compile_batch_summary(all_results)
        
        return {
            'experiment_config': {
                'agent_counts': agent_counts,
                'information_types': information_types,
                'complexity_levels': complexity_levels,
                'model_names': model_names,
                'temperatures': temperatures,
                'runs_per_config': runs_per_config
            },
            'total_runs': len(all_results),
            'individual_results': all_results,
            'batch_summary': batch_summary
        }
    
    def _compile_batch_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compile summary statistics from batch results."""
        if not results:
            return {}
        
        config_groups = {}
        for result in results:
            config_key = (
                result['num_agents'],
                result['information_seed']['information_type'],
                result['information_seed']['complexity_level'],
                result['model_name'],
                result['temperature']
            )
            
            if config_key not in config_groups:
                config_groups[config_key] = []
            config_groups[config_key].append(result)
        
        config_summaries = {}
        for config_key, config_results in config_groups.items():
            metrics = [r['evaluation_metrics'] for r in config_results]
            
            avg_metrics = {
                'retention_score': sum(m['retention_score'] for m in metrics) / len(metrics),
                'semantic_similarity': sum(m['semantic_similarity'] for m in metrics) / len(metrics),
                'factual_accuracy': sum(m['factual_accuracy'] for m in metrics) / len(metrics),
                'structural_integrity': sum(m['structural_integrity'] for m in metrics) / len(metrics),
                'degradation_rate': sum(m['degradation_rate'] for m in metrics) / len(metrics),
                'key_elements_preserved_ratio': sum(m['key_elements_preserved'] / max(m['total_key_elements'], 1) for m in metrics) / len(metrics)
            }
            
            config_summaries[config_key] = {
                'config': {
                    'num_agents': config_key[0],
                    'information_type': config_key[1],
                    'complexity_level': config_key[2],
                    'model_name': config_key[3],
                    'temperature': config_key[4]
                },
                'num_runs': len(config_results),
                'average_metrics': avg_metrics
            }
        
        return {
            'total_configurations': len(config_groups),
            'configuration_summaries': list(config_summaries.values())
        }
    
    def save_results(self, results: Dict[str, Any], filepath: str):
        """Save experiment results to JSON file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            if self.logger:
                self.logger.log(f"[Game Controller] Results saved to {filepath}")
        except Exception as e:
            if self.logger:
                self.logger.log(f"[Game Controller] Error saving results: {e}")
