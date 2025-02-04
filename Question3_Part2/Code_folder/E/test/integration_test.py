# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 04:43:18 2025

@author: EMRE
"""
import unittest
import networkx as nx
import numpy as np
from multiple_variables import UnifiedCausalOptimizer

class TestUnifiedCausalOptimizerIntegration(unittest.TestCase):
    
    def setUp(self):
        """Set up a causal graph and optimizer instance for integration testing."""
        self.graph = nx.DiGraph()
        self.graph.add_edges_from([('X1', 'Y1'), ('X2', 'Y2'), ('X1', 'Y2')])
        
        self.optimizer = UnifiedCausalOptimizer(
            causal_graph=self.graph,
            target_variables=['Y1', 'Y2'],
            n_iterations=10  # Reduced for faster testing
        )
        
        self.initial_data = {'X1': 0, 'X2': 0, 'Y1': 0, 'Y2': 0}
    
    def test_full_pipeline(self):
        """Ensure the entire optimization process runs without errors."""
        try:
            self.optimizer.optimize(self.initial_data)
        except Exception as e:
            self.fail(f"Integration test failed during optimization: {e}")
        
        summary = self.optimizer.get_summary_statistics()
        fig = self.optimizer.visualize_results()
        
        # Check if the summary statistics are generated
        self.assertIn('final_pdc', summary)
        self.assertIn('avg_pdc', summary)
        self.assertIn('final_regret', summary)
        self.assertIn('final_error', summary)
        self.assertIn('convergence_iteration', summary)
        
        # Ensure visualization is generated
        self.assertIsNotNone(fig)
        
    def test_consistency_between_components(self):
        """Ensure consistency across different parts of the optimization."""
        self.optimizer.optimize(self.initial_data)
        
        pdc_scores = self.optimizer.history['pdc_scores']
        bayes_factors = self.optimizer.history['bayes_factors']
        intervention_effects = self.optimizer.history['intervention_effects']
        
        # Ensure PDC scores are non-decreasing (expected behavior in many cases)
        self.assertTrue(np.all(np.diff(pdc_scores) >= -0.1))  # Allow small fluctuations
        
        # Ensure Bayes factors and intervention effects have reasonable values
        self.assertFalse(np.isnan(bayes_factors).any())
        self.assertFalse(np.isnan(intervention_effects).any())
    
if __name__ == "__main__":
    unittest.main()
