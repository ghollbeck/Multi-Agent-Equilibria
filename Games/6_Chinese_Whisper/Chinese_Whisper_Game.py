"""
Chinese Whisper Game Simulation with LLM-Based Information Processing
===================================================================

This script demonstrates how to run a Chinese Whisper game simulation using large 
language models (LLMs) for information processing and reformulation. It investigates
memory loss and information degradation across chains of LLM agent interactions.

Game Flow:
1. **Information Injection**: Start with original information (seed data)
2. **Agent Chain Processing**: Each agent in sequence:
   - Receives information from previous agent (or seed)
   - Processes and reformulates the information
   - Passes result to next agent in chain
3. **Evaluation**: Compare final output against original seed data
4. **Analysis**: Generate metrics on information retention and degradation patterns

Information Types:
- **Factual Data**: Names, dates, numbers, locations, specific facts
- **Narrative Stories**: Short stories with plot, characters, and timeline
- **Technical Instructions**: Step-by-step procedures or algorithms
- **Structured Data**: Lists, categorized information, hierarchical data

Evaluation Metrics:
- **Retention Score**: Percentage of original information preserved
- **Semantic Similarity**: Similarity between original and final information
- **Factual Accuracy**: Percentage of facts correctly preserved
- **Structural Integrity**: How well the original format/structure is maintained
- **Degradation Rate**: Information loss per agent in chain

Usage:
    python Chinese_Whisper_Game.py
"""

import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from game_controller import ChineseWhisperGameController
from models_chinese_whisper import ChineseWhisperLogger
from information_generator import InformationGenerator
from analytics import ChineseWhisperAnalytics

async def run_simple_experiment():
    """Run a simple Chinese Whisper experiment."""
    print("🎯 Starting Simple Chinese Whisper Experiment")
    
    log_file = "data/results/simple_experiment.log"
    Path("data/results").mkdir(parents=True, exist_ok=True)
    logger = ChineseWhisperLogger(log_to_file=log_file)
    
    controller = ChineseWhisperGameController(logger=logger)
    
    info_generator = InformationGenerator(seed=42)
    test_info = info_generator.generate_information("narrative", "simple")
    
    print(f"📝 Original Information ({len(test_info.content)} chars):")
    print(f"   {test_info.content[:100]}...")
    print()
    
    result = await controller.run_single_chain(
        num_agents=5,
        information_seed=test_info,
        model_name="gpt-4o-mini",
        temperature=0.7,
        generation_id=1
    )
    
    print("📊 Results:")
    metrics = result['evaluation_metrics']
    print(f"   Retention Score: {metrics['retention_score']:.2f}")
    print(f"   Semantic Similarity: {metrics['semantic_similarity']:.2f}")
    print(f"   Factual Accuracy: {metrics['factual_accuracy']:.2f}")
    print(f"   Degradation Rate: {metrics['degradation_rate']:.2f}")
    print()
    
    print(f"📝 Final Information ({len(result['chain_data'][-1])} chars):")
    print(f"   {result['chain_data'][-1][:100]}...")
    print()
    
    results_file = "data/results/simple_experiment_results.json"
    controller.save_results(result, results_file)
    print(f"💾 Results saved to {results_file}")
    
    logger.close()
    return result

async def run_batch_experiment():
    """Run a comprehensive batch experiment."""
    print("🎯 Starting Batch Chinese Whisper Experiment")
    
    log_file = "data/results/batch_experiment.log"
    Path("data/results").mkdir(parents=True, exist_ok=True)
    logger = ChineseWhisperLogger(log_to_file=log_file)
    
    controller = ChineseWhisperGameController(logger=logger)
    
    agent_counts = [2, 5, 8]
    information_types = ["factual", "narrative", "technical"]
    complexity_levels = ["simple", "medium"]
    model_names = ["gpt-4o-mini"]
    temperatures = [0.3, 0.7]
    runs_per_config = 2
    
    print(f"🔧 Configuration:")
    print(f"   Agent counts: {agent_counts}")
    print(f"   Information types: {information_types}")
    print(f"   Complexity levels: {complexity_levels}")
    print(f"   Models: {model_names}")
    print(f"   Temperatures: {temperatures}")
    print(f"   Runs per config: {runs_per_config}")
    print()
    
    start_time = time.time()
    results = await controller.run_batch_experiment(
        agent_counts=agent_counts,
        information_types=information_types,
        complexity_levels=complexity_levels,
        model_names=model_names,
        temperatures=temperatures,
        runs_per_config=runs_per_config
    )
    end_time = time.time()
    
    print(f"⏱️  Batch experiment completed in {end_time - start_time:.1f} seconds")
    print(f"📊 Total runs: {results['total_runs']}")
    print()
    
    results_file = "data/results/batch_experiment_results.json"
    controller.save_results(results, results_file)
    print(f"💾 Results saved to {results_file}")
    
    analytics = ChineseWhisperAnalytics(results)
    
    report = analytics.generate_summary_report()
    report_file = "data/results/batch_experiment_report.json"
    analytics.save_report(report, report_file)
    print(f"📈 Analysis report saved to {report_file}")
    
    print("🔍 Key Findings:")
    overall = report['overall_performance']
    print(f"   Average Retention Score: {overall['average_retention_score']:.3f}")
    print(f"   Average Semantic Similarity: {overall['average_semantic_similarity']:.3f}")
    print(f"   Average Factual Accuracy: {overall['average_factual_accuracy']:.3f}")
    print(f"   Average Degradation Rate: {overall['average_degradation_rate']:.3f}")
    print()
    
    if report['best_configurations']:
        best = report['best_configurations'][0]
        print("🏆 Best Configuration:")
        print(f"   Agents: {best['num_agents']}, Type: {best['information_type']}")
        print(f"   Complexity: {best['complexity_level']}, Model: {best['model_name']}")
        print(f"   Temperature: {best['temperature']}")
        print(f"   Composite Score: {best['composite_score']:.3f}")
    
    logger.close()
    return results

async def run_degradation_analysis():
    """Run focused analysis on degradation patterns."""
    print("🎯 Starting Degradation Pattern Analysis")
    
    log_file = "data/results/degradation_analysis.log"
    Path("data/results").mkdir(parents=True, exist_ok=True)
    logger = ChineseWhisperLogger(log_to_file=log_file)
    
    controller = ChineseWhisperGameController(logger=logger)
    
    info_generator = InformationGenerator(seed=123)
    test_info = info_generator.generate_information("factual", "medium")
    
    print(f"📝 Test Information: {test_info.information_type} - {test_info.complexity_level}")
    print(f"   Length: {len(test_info.content)} chars")
    print(f"   Key elements: {len(test_info.expected_key_elements)}")
    print()
    
    chain_lengths = [2, 4, 6, 8, 10, 12]
    degradation_results = []
    
    for chain_length in chain_lengths:
        print(f"🔗 Testing chain length: {chain_length}")
        
        result = await controller.run_single_chain(
            num_agents=chain_length,
            information_seed=test_info,
            model_name="gpt-4o-mini",
            temperature=0.5,
            generation_id=chain_length
        )
        
        metrics = result['evaluation_metrics']
        degradation_results.append({
            'chain_length': chain_length,
            'retention_score': metrics['retention_score'],
            'semantic_similarity': metrics['semantic_similarity'],
            'factual_accuracy': metrics['factual_accuracy'],
            'degradation_rate': metrics['degradation_rate'],
            'critical_point': result['threshold_analysis'].get('critical_point')
        })
        
        print(f"   Retention: {metrics['retention_score']:.3f}, "
              f"Similarity: {metrics['semantic_similarity']:.3f}, "
              f"Accuracy: {metrics['factual_accuracy']:.3f}")
    
    print()
    print("📊 Degradation Analysis Summary:")
    for result in degradation_results:
        print(f"   Chain {result['chain_length']:2d}: "
              f"Retention {result['retention_score']:.3f}, "
              f"Rate {result['degradation_rate']:.3f}")
    
    analysis_file = "data/results/degradation_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump({
            'test_information': {
                'type': test_info.information_type,
                'complexity': test_info.complexity_level,
                'length': len(test_info.content),
                'key_elements_count': len(test_info.expected_key_elements)
            },
            'degradation_results': degradation_results
        }, f, indent=2)
    
    print(f"💾 Degradation analysis saved to {analysis_file}")
    
    logger.close()
    return degradation_results

async def main():
    """Main function to run Chinese Whisper experiments."""
    print("🎮 Chinese Whisper Game - LLM Memory Loss Research")
    print("=" * 55)
    print()
    
    Path("data/seed_information").mkdir(parents=True, exist_ok=True)
    Path("data/results").mkdir(parents=True, exist_ok=True)
    
    try:
        print("1️⃣  Running Simple Experiment...")
        await run_simple_experiment()
        print()
        
        print("2️⃣  Running Batch Experiment...")
        await run_batch_experiment()
        print()
        
        print("3️⃣  Running Degradation Analysis...")
        await run_degradation_analysis()
        print()
        
        print("✅ All experiments completed successfully!")
        print("📁 Check the 'data/results/' directory for output files")
        
    except Exception as e:
        print(f"❌ Error during experiments: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
