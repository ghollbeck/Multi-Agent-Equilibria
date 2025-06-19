import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
from pathlib import Path

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    sns = None

class ChineseWhisperAnalytics:
    """
    Analytics and visualization tools for Chinese Whisper game results.
    Provides comprehensive analysis of information degradation patterns.
    """
    
    def __init__(self, results_data: Optional[Dict[str, Any]] = None):
        self.results_data = results_data
        self.df = None
        if results_data:
            self.df = self._create_dataframe(results_data)
    
    def load_results(self, filepath: str):
        """Load results from JSON file."""
        with open(filepath, 'r') as f:
            self.results_data = json.load(f)
        self.df = self._create_dataframe(self.results_data)
    
    def _create_dataframe(self, results_data: Dict[str, Any]) -> pd.DataFrame:
        """Convert results data to pandas DataFrame for analysis."""
        rows = []
        
        if 'individual_results' in results_data:
            for result in results_data['individual_results']:
                base_row = {
                    'generation_id': result['generation_id'],
                    'num_agents': result['num_agents'],
                    'model_name': result['model_name'],
                    'temperature': result['temperature'],
                    'information_type': result['information_seed']['information_type'],
                    'complexity_level': result['information_seed']['complexity_level'],
                    'original_length': len(result['information_seed']['content']),
                    'final_length': len(result['chain_data'][-1]) if result['chain_data'] else 0,
                    'total_key_elements': result['information_seed']['expected_key_elements'].__len__()
                }
                
                metrics = result['evaluation_metrics']
                base_row.update({
                    'retention_score': metrics['retention_score'],
                    'semantic_similarity': metrics['semantic_similarity'],
                    'factual_accuracy': metrics['factual_accuracy'],
                    'structural_integrity': metrics['structural_integrity'],
                    'information_completeness': metrics['information_completeness'],
                    'degradation_rate': metrics['degradation_rate'],
                    'key_elements_preserved': metrics['key_elements_preserved']
                })
                
                if 'threshold_analysis' in result:
                    base_row['critical_point'] = result['threshold_analysis'].get('critical_point')
                
                rows.append(base_row)
        
        return pd.DataFrame(rows)
    
    def plot_degradation_by_chain_length(self, save_path: Optional[str] = None) -> matplotlib.figure.Figure:
        """Plot how information degrades with chain length."""
        if self.df is None:
            raise ValueError("No data loaded. Use load_results() first.")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Information Degradation by Chain Length', fontsize=16)
        
        if HAS_SEABORN:
            sns.boxplot(data=self.df, x='num_agents', y='retention_score', ax=axes[0,0])
        else:
            self.df.boxplot(column='retention_score', by='num_agents', ax=axes[0,0])
        axes[0,0].set_title('Retention Score vs Chain Length')
        axes[0,0].set_ylabel('Retention Score')
        
        if HAS_SEABORN:
            sns.boxplot(data=self.df, x='num_agents', y='semantic_similarity', ax=axes[0,1])
        else:
            self.df.boxplot(column='semantic_similarity', by='num_agents', ax=axes[0,1])
        axes[0,1].set_title('Semantic Similarity vs Chain Length')
        axes[0,1].set_ylabel('Semantic Similarity')
        
        if HAS_SEABORN:
            sns.boxplot(data=self.df, x='num_agents', y='factual_accuracy', ax=axes[1,0])
        else:
            self.df.boxplot(column='factual_accuracy', by='num_agents', ax=axes[1,0])
        axes[1,0].set_title('Factual Accuracy vs Chain Length')
        axes[1,0].set_ylabel('Factual Accuracy')
        
        if HAS_SEABORN:
            sns.boxplot(data=self.df, x='num_agents', y='degradation_rate', ax=axes[1,1])
        else:
            self.df.boxplot(column='degradation_rate', by='num_agents', ax=axes[1,1])
        axes[1,1].set_title('Degradation Rate vs Chain Length')
        axes[1,1].set_ylabel('Degradation Rate')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_information_type_comparison(self, save_path: Optional[str] = None) -> matplotlib.figure.Figure:
        """Compare degradation across different information types."""
        if self.df is None:
            raise ValueError("No data loaded. Use load_results() first.")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Information Degradation by Information Type', fontsize=16)
        
        if HAS_SEABORN:
            sns.boxplot(data=self.df, x='information_type', y='retention_score', ax=axes[0,0])
        else:
            self.df.boxplot(column='retention_score', by='information_type', ax=axes[0,0])
        axes[0,0].set_title('Retention Score by Information Type')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        if HAS_SEABORN:
            sns.boxplot(data=self.df, x='information_type', y='factual_accuracy', ax=axes[0,1])
        else:
            self.df.boxplot(column='factual_accuracy', by='information_type', ax=axes[0,1])
        axes[0,1].set_title('Factual Accuracy by Information Type')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        if HAS_SEABORN:
            sns.boxplot(data=self.df, x='information_type', y='structural_integrity', ax=axes[1,0])
        else:
            self.df.boxplot(column='structural_integrity', by='information_type', ax=axes[1,0])
        axes[1,0].set_title('Structural Integrity by Information Type')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        if HAS_SEABORN:
            sns.boxplot(data=self.df, x='information_type', y='degradation_rate', ax=axes[1,1])
        else:
            self.df.boxplot(column='degradation_rate', by='information_type', ax=axes[1,1])
        axes[1,1].set_title('Degradation Rate by Information Type')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_model_performance_comparison(self, save_path: Optional[str] = None) -> Optional[matplotlib.figure.Figure]:
        """Compare performance across different models."""
        if self.df is None:
            raise ValueError("No data loaded. Use load_results() first.")
        
        if 'model_name' not in self.df.columns or self.df['model_name'].nunique() < 2:
            print("Not enough model variety for comparison")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        if HAS_SEABORN:
            sns.boxplot(data=self.df, x='model_name', y='retention_score', ax=axes[0,0])
        else:
            self.df.boxplot(column='retention_score', by='model_name', ax=axes[0,0])
        axes[0,0].set_title('Retention Score by Model')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        if HAS_SEABORN:
            sns.boxplot(data=self.df, x='model_name', y='semantic_similarity', ax=axes[0,1])
        else:
            self.df.boxplot(column='semantic_similarity', by='model_name', ax=axes[0,1])
        axes[0,1].set_title('Semantic Similarity by Model')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        if HAS_SEABORN:
            sns.boxplot(data=self.df, x='model_name', y='factual_accuracy', ax=axes[1,0])
        else:
            self.df.boxplot(column='factual_accuracy', by='model_name', ax=axes[1,0])
        axes[1,0].set_title('Factual Accuracy by Model')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        self.df['consistency'] = 1 - self.df['degradation_rate']
        if HAS_SEABORN:
            sns.boxplot(data=self.df, x='model_name', y='consistency', ax=axes[1,1])
        else:
            self.df.boxplot(column='consistency', by='model_name', ax=axes[1,1])
        axes[1,1].set_title('Processing Consistency by Model')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_critical_thresholds(self, save_path: Optional[str] = None) -> matplotlib.figure.Figure:
        """Plot analysis of critical degradation thresholds."""
        if self.df is None:
            raise ValueError("No data loaded. Use load_results() first.")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Critical Threshold Analysis', fontsize=16)
        
        critical_points = self.df['critical_point'].dropna()
        if len(critical_points) > 0:
            axes[0].hist(critical_points, bins=range(1, int(critical_points.max()) + 2), alpha=0.7)
            axes[0].set_title('Distribution of Critical Points')
            axes[0].set_xlabel('Agent Position Where Degradation Accelerates')
            axes[0].set_ylabel('Frequency')
        else:
            axes[0].text(0.5, 0.5, 'No critical points detected', ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title('Distribution of Critical Points')
        
        for info_type in self.df['information_type'].unique():
            type_data = self.df[self.df['information_type'] == info_type]
            axes[1].scatter(type_data['num_agents'], type_data['retention_score'], 
                          label=info_type, alpha=0.6)
        
        axes[1].set_title('Retention Score vs Chain Length by Information Type')
        axes[1].set_xlabel('Number of Agents')
        axes[1].set_ylabel('Retention Score')
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary report."""
        if self.df is None:
            raise ValueError("No data loaded. Use load_results() first.")
        
        report = {
            'dataset_overview': {
                'total_experiments': len(self.df),
                'unique_configurations': len(self.df.groupby(['num_agents', 'information_type', 'complexity_level', 'model_name', 'temperature'])),
                'agent_counts_tested': sorted(self.df['num_agents'].unique().tolist()),
                'information_types_tested': self.df['information_type'].unique().tolist(),
                'complexity_levels_tested': self.df['complexity_level'].unique().tolist(),
                'models_tested': self.df['model_name'].unique().tolist()
            },
            'overall_performance': {
                'average_retention_score': self.df['retention_score'].mean(),
                'average_semantic_similarity': self.df['semantic_similarity'].mean(),
                'average_factual_accuracy': self.df['factual_accuracy'].mean(),
                'average_structural_integrity': self.df['structural_integrity'].mean(),
                'average_degradation_rate': self.df['degradation_rate'].mean()
            },
            'chain_length_analysis': self._analyze_chain_length_impact(),
            'information_type_analysis': self._analyze_information_type_impact(),
            'critical_threshold_analysis': self._analyze_critical_thresholds(),
            'best_configurations': self._find_best_configurations(),
            'worst_configurations': self._find_worst_configurations()
        }
        
        return report
    
    def _analyze_chain_length_impact(self) -> Dict[str, Any]:
        """Analyze impact of chain length on performance."""
        if self.df is None:
            return {}
        
        chain_analysis = {}
        
        for agent_count in sorted(self.df['num_agents'].unique()):
            subset = self.df[self.df['num_agents'] == agent_count]
            chain_analysis[f'{agent_count}_agents'] = {
                'count': len(subset),
                'avg_retention': subset['retention_score'].mean(),
                'avg_semantic_similarity': subset['semantic_similarity'].mean(),
                'avg_factual_accuracy': subset['factual_accuracy'].mean(),
                'avg_degradation_rate': subset['degradation_rate'].mean()
            }
        
        return chain_analysis
    
    def _analyze_information_type_impact(self) -> Dict[str, Any]:
        """Analyze impact of information type on performance."""
        if self.df is None:
            return {}
        
        type_analysis = {}
        
        for info_type in self.df['information_type'].unique():
            subset = self.df[self.df['information_type'] == info_type]
            type_analysis[info_type] = {
                'count': len(subset),
                'avg_retention': subset['retention_score'].mean(),
                'avg_semantic_similarity': subset['semantic_similarity'].mean(),
                'avg_factual_accuracy': subset['factual_accuracy'].mean(),
                'avg_structural_integrity': subset['structural_integrity'].mean(),
                'most_vulnerable_metric': self._find_most_vulnerable_metric(subset)
            }
        
        return type_analysis
    
    def _analyze_critical_thresholds(self) -> Dict[str, Any]:
        """Analyze critical threshold patterns."""
        if self.df is None:
            return {'critical_points_detected': 0}
        
        if 'critical_point' not in self.df.columns:
            return {'critical_points_detected': 0, 'message': 'No critical point data available'}
        
        critical_points = self.df['critical_point'].dropna()
        
        if len(critical_points) == 0:
            return {'critical_points_detected': 0}
        
        return {
            'critical_points_detected': len(critical_points),
            'average_critical_point': critical_points.mean(),
            'most_common_critical_point': critical_points.mode().iloc[0] if len(critical_points.mode()) > 0 else None,
            'critical_point_range': [critical_points.min(), critical_points.max()]
        }
    
    def _find_most_vulnerable_metric(self, subset: pd.DataFrame) -> str:
        """Find which metric degrades most for a given subset."""
        metrics = ['retention_score', 'semantic_similarity', 'factual_accuracy', 'structural_integrity']
        metric_means = {metric: subset[metric].mean() for metric in metrics}
        return min(metric_means.keys(), key=lambda k: metric_means[k])
    
    def _find_best_configurations(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """Find best performing configurations."""
        if self.df is None:
            return []
        
        self.df['composite_score'] = (
            self.df['retention_score'] * 0.3 +
            self.df['semantic_similarity'] * 0.3 +
            self.df['factual_accuracy'] * 0.25 +
            self.df['structural_integrity'] * 0.15
        )
        
        best_configs = self.df.nlargest(top_n, 'composite_score')
        
        return [
            {
                'rank': i + 1,
                'num_agents': row['num_agents'],
                'information_type': row['information_type'],
                'complexity_level': row['complexity_level'],
                'model_name': row['model_name'],
                'temperature': row['temperature'],
                'composite_score': row['composite_score'],
                'retention_score': row['retention_score'],
                'semantic_similarity': row['semantic_similarity'],
                'factual_accuracy': row['factual_accuracy']
            }
            for i, (_, row) in enumerate(best_configs.iterrows())
        ]
    
    def _find_worst_configurations(self, bottom_n: int = 5) -> List[Dict[str, Any]]:
        """Find worst performing configurations."""
        if self.df is None:
            return []
        
        if 'composite_score' not in self.df.columns:
            self.df['composite_score'] = (
                self.df['retention_score'] * 0.3 +
                self.df['semantic_similarity'] * 0.3 +
                self.df['factual_accuracy'] * 0.25 +
                self.df['structural_integrity'] * 0.15
            )
        
        worst_configs = self.df.nsmallest(bottom_n, 'composite_score')
        
        return [
            {
                'rank': i + 1,
                'num_agents': row['num_agents'],
                'information_type': row['information_type'],
                'complexity_level': row['complexity_level'],
                'model_name': row['model_name'],
                'temperature': row['temperature'],
                'composite_score': row['composite_score'],
                'retention_score': row['retention_score'],
                'semantic_similarity': row['semantic_similarity'],
                'factual_accuracy': row['factual_accuracy']
            }
            for i, (_, row) in enumerate(worst_configs.iterrows())
        ]
    
    def save_report(self, report: Dict[str, Any], filepath: str):
        """Save analysis report to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Report saved to {filepath}")
