# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 04:38:12 2025

@author: EMRE
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy import stats
from typing import List, Dict, Tuple
import seaborn as sns
from scipy.stats import norm

class UnifiedCausalOptimizer:
    def __init__(self, causal_graph: nx.DiGraph, target_variables: List[str], 
                 n_iterations: int = 100, alpha: float = 0.05):
        """
        Initialize the unified causal optimizer.
        
        Args:
            causal_graph: NetworkX DiGraph representing causal structure
            target_variables: List of variables to optimize
            n_iterations: Number of optimization iterations
            alpha: Significance level
        """
        self.graph = causal_graph
        self.target_variables = target_variables
        self.n_iterations = n_iterations
        self.alpha = alpha
        
        # Initialize history tracking
        self.history = {
            'pdc_scores': np.zeros(n_iterations),
            'bayes_factors': np.zeros((n_iterations, len(target_variables))),
            'intervention_effects': np.zeros((n_iterations, len(target_variables))),
            'cumulative_regret': np.zeros(n_iterations),
            'estimation_errors': np.zeros(n_iterations),
            'posterior_probs': np.zeros((n_iterations, len(target_variables)))
        }
        
        # Initialize GP model parameters
        self.gp_lengthscale = 1.0
        self.gp_variance = 1.0
        self.noise_variance = 0.1

    def optimize(self, initial_data: Dict):
        """Run the optimization process."""
        current_data = initial_data.copy()
        
        for t in range(self.n_iterations):
            # Compute PDC and intervention effects
            pdc, effects = self._compute_iteration_metrics(current_data, t)
            
            # Update history
            self.history['pdc_scores'][t] = pdc
            self.history['intervention_effects'][t] = effects
            
            # Compute Bayes factors
            bf = self._compute_bayes_factors(current_data, t)
            self.history['bayes_factors'][t] = bf
            
            # Update posterior probabilities
            post_probs = self._update_posterior_probs(bf, t)
            self.history['posterior_probs'][t] = post_probs
            
            # Compute regret and estimation error
            self.history['cumulative_regret'][t] = self._compute_regret(t)
            self.history['estimation_errors'][t] = self._compute_estimation_error(t)
            
            # Generate next intervention
            new_intervention = self._select_next_intervention(current_data, t)
            current_data = self._simulate_intervention(new_intervention)

    def visualize_results(self):
        """Create comprehensive visualization of results."""
        fig = plt.figure(figsize=(15, 10))
        
        # 1. PDC Score Evolution
        ax1 = plt.subplot(231)
        ax1.plot(self.history['pdc_scores'], 'b-', label='PDC Score')
        ax1.set_title('PDC Score Evolution')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('PDC Score')
        ax1.grid(True)
        
        # 2. Bayes Factors
        ax2 = plt.subplot(232)
        for i, target in enumerate(self.target_variables):
            ax2.plot(self.history['bayes_factors'][:, i], 
                    label=f'Target {target}')
        ax2.set_title('Bayes Factors')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('log(BF)')
        ax2.legend()
        ax2.grid(True)
        
        # 3. Intervention Effects
        ax3 = plt.subplot(233)
        for i, target in enumerate(self.target_variables):
            ax3.plot(self.history['intervention_effects'][:, i], 
                    label=f'Target {target}')
        ax3.set_title('Intervention Effects')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Effect Size')
        ax3.legend()
        ax3.grid(True)
        
        # 4. Cumulative Regret
        ax4 = plt.subplot(234)
        ax4.plot(self.history['cumulative_regret'], 'r-', label='Regret')
        ax4.set_title('Cumulative Regret')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Regret')
        ax4.grid(True)
        
        # 5. Estimation Error
        ax5 = plt.subplot(235)
        ax5.plot(self.history['estimation_errors'], 'g-', label='Error')
        ax5.set_title('Estimation Error')
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Error')
        ax5.grid(True)
        
        # 6. Posterior Probabilities
        ax6 = plt.subplot(236)
        for i, target in enumerate(self.target_variables):
            ax6.plot(self.history['posterior_probs'][:, i], 
                    label=f'Target {target}')
        ax6.set_title('Posterior Probabilities')
        ax6.set_xlabel('Iteration')
        ax6.set_ylabel('P(H1|Data)')
        ax6.legend()
        ax6.grid(True)
        
        plt.tight_layout()
        return fig

    def get_summary_statistics(self):
        """Return summary statistics of the optimization."""
        summary = {
            'final_pdc': self.history['pdc_scores'][-1],
            'avg_pdc': np.mean(self.history['pdc_scores']),
            'final_regret': self.history['cumulative_regret'][-1],
            'final_error': self.history['estimation_errors'][-1],
            'convergence_iteration': self._find_convergence_iteration()
        }
        return summary

    def _compute_iteration_metrics(self, data: Dict, t: int) -> Tuple[float, np.ndarray]:
        """Compute PDC and intervention effects for current iteration."""
        # Simulate computation of PDC and effects
        pdc = 1.0 / (1 + np.exp(-t/20))  # Example computation
        effects = np.random.normal(0, 1, len(self.target_variables))
        effects *= np.exp(-t/50)  # Decay over time
        return pdc, effects

    def _compute_bayes_factors(self, data: Dict, t: int) -> np.ndarray:
        """Compute Bayes factors for each target variable."""
        bf = np.zeros(len(self.target_variables))
        for i in range(len(self.target_variables)):
            # Simulate Bayes factor computation
            bf[i] = np.exp(-0.1 * t) * np.random.normal(0, 1)
        return bf

    def _update_posterior_probs(self, bf: np.ndarray, t: int) -> np.ndarray:
        """Update posterior probabilities using Bayes factors."""
        post_probs = 1 / (1 + np.exp(-bf))
        return post_probs

    def _compute_regret(self, t: int) -> float:
        """Compute regret for current iteration."""
        return np.exp(-0.05 * t) * np.random.gamma(2, 0.5)

    def _compute_estimation_error(self, t: int) -> float:
        """Compute estimation error for current iteration."""
        return np.exp(-0.03 * t) * np.random.gamma(1, 0.3)

    def _select_next_intervention(self, data: Dict, t: int) -> Dict:
        """Select next intervention using acquisition function."""
        # Simplified intervention selection
        return {'intervention_value': np.random.normal(0, 1)}

    def _simulate_intervention(self, intervention: Dict) -> Dict:
        """Simulate intervention results."""
        return {'outcome': np.random.normal(intervention['intervention_value'], 0.1)}

    def _find_convergence_iteration(self) -> int:
        """Find iteration where optimization converged."""
        # Simple convergence criterion
        threshold = 0.01
        for t in range(1, self.n_iterations):
            if abs(self.history['pdc_scores'][t] - self.history['pdc_scores'][t-1]) < threshold:
                return t
        return self.n_iterations

# Example usage
def run_example():
    # Create simple causal graph
    G = nx.DiGraph()
    G.add_edges_from([('X1', 'Y1'), ('X2', 'Y2'), ('X1', 'Y2')])
    
    # Initialize optimizer
    optimizer = UnifiedCausalOptimizer(
        causal_graph=G,
        target_variables=['Y1', 'Y2'],
        n_iterations=100
    )
    
    # Run optimization
    initial_data = {'X1': 0, 'X2': 0, 'Y1': 0, 'Y2': 0}
    optimizer.optimize(initial_data)
    
    # Plot results
    fig = optimizer.visualize_results()
    
    # Get summary statistics
    summary = optimizer.get_summary_statistics()
    print("\nOptimization Summary:")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")
    
    return optimizer, fig

if __name__ == "__main__":
    optimizer, fig = run_example()
    plt.show()