# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 04:41:18 2025

@author: EMRE
"""

import unittest
import os
import pickle
import numpy as np
import networkx as nx
from multiple_variables import UnifiedCausalOptimizer

BASELINE_FILE = "regression_baseline.pkl"

class TestUnifiedCausalOptimizerRegression(unittest.TestCase):
    
    def setUp(self):
        """Set up a simple causal graph and optimizer instance."""
        self.graph = nx.DiGraph()
        self.graph.add_edges_from([('X1', 'Y1'), ('X2', 'Y2'), ('X1', 'Y2')])
        
        self.optimizer = UnifiedCausalOptimizer(
            causal_graph=self.graph,
            target_variables=['Y1', 'Y2'],
            n_iterations=10  # Reduced for faster testing
        )
        
        self.initial_data = {'X1': 0, 'X2': 0, 'Y1': 0, 'Y2': 0}
    
    def test_regression(self):
        """Ensure the optimizer produces consistent results over time."""
        self.optimizer.optimize(self.initial_data)
        new_output = {
            'pdc_scores': self.optimizer.history['pdc_scores'],
            'bayes_factors': self.optimizer.history['bayes_factors'],
            'intervention_effects': self.optimizer.history['intervention_effects'],
            'cumulative_regret': self.optimizer.history['cumulative_regret'],
            'estimation_errors': self.optimizer.history['estimation_errors'],
            'posterior_probs': self.optimizer.history['posterior_probs']
        }
        
        if os.path.exists(BASELINE_FILE):
            with open(BASELINE_FILE, "rb") as f:
                baseline_output = pickle.load(f)
            
            for key in new_output:
                np.testing.assert_array_almost_equal(
                    new_output[key], baseline_output[key], decimal=6,
                    err_msg=f"Mismatch in {key} between new and baseline outputs"
                )
        else:
            with open(BASELINE_FILE, "wb") as f:
                pickle.dump(new_output, f)
            print("Baseline regression data saved.")

if __name__ == "__main__":
    unittest.main()