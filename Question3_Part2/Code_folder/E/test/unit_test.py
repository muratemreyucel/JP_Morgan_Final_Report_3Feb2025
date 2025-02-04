# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 04:38:27 2025

@author: EMRE
"""
import unittest
import networkx as nx
import numpy as np
from multiple_variables import UnifiedCausalOptimizer

class TestUnifiedCausalOptimizer(unittest.TestCase):
    
    def setUp(self):
        """Set up a simple causal graph and optimizer instance."""
        self.graph = nx.DiGraph()
        self.graph.add_edges_from([('X1', 'Y1'), ('X2', 'Y2'), ('X1', 'Y2')])
        
        self.optimizer = UnifiedCausalOptimizer(
            causal_graph=self.graph,
            target_variables=['Y1', 'Y2'],
            n_iterations=10  # Reduce iterations for faster testing
        )
        
        self.initial_data = {'X1': 0, 'X2': 0, 'Y1': 0, 'Y2': 0}
    
    def test_initialization(self):
        """Test if the optimizer initializes correctly."""
        self.assertEqual(len(self.optimizer.target_variables), 2)
        self.assertEqual(self.optimizer.n_iterations, 10)
        self.assertEqual(self.optimizer.history['pdc_scores'].shape, (10,))
    
    def test_optimization_runs(self):
        """Test if the optimization process runs without errors."""
        try:
            self.optimizer.optimize(self.initial_data)
        except Exception as e:
            self.fail(f"Optimization process failed with error: {e}")
    
    def test_history_integrity(self):
        """Ensure history data has the correct shape after optimization."""
        self.optimizer.optimize(self.initial_data)
        
        self.assertEqual(self.optimizer.history['pdc_scores'].shape, (10,))
        self.assertEqual(self.optimizer.history['bayes_factors'].shape, (10, 2))
        self.assertEqual(self.optimizer.history['intervention_effects'].shape, (10, 2))
        self.assertEqual(self.optimizer.history['cumulative_regret'].shape, (10,))
        self.assertEqual(self.optimizer.history['estimation_errors'].shape, (10,))
        self.assertEqual(self.optimizer.history['posterior_probs'].shape, (10, 2))
    
    def test_summary_statistics(self):
        """Test if summary statistics are computed properly."""
        self.optimizer.optimize(self.initial_data)
        summary = self.optimizer.get_summary_statistics()
        
        self.assertIn('final_pdc', summary)
        self.assertIn('avg_pdc', summary)
        self.assertIn('final_regret', summary)
        self.assertIn('final_error', summary)
        self.assertIn('convergence_iteration', summary)
        
        self.assertIsInstance(summary['final_pdc'], float)
        self.assertIsInstance(summary['avg_pdc'], float)
        self.assertIsInstance(summary['final_regret'], float)
        self.assertIsInstance(summary['final_error'], float)
        self.assertIsInstance(summary['convergence_iteration'], int)
    
    def test_visualization(self):
        """Test if visualization function runs without errors."""
        self.optimizer.optimize(self.initial_data)
        try:
            fig = self.optimizer.visualize_results()
        except Exception as e:
            self.fail(f"Visualization failed with error: {e}")
        
        self.assertIsNotNone(fig)
    
if __name__ == "__main__":
    unittest.main()


