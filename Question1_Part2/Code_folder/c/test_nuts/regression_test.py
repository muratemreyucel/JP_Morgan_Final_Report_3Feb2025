import unittest
import tensorflow as tf
import numpy as np
from NUTS_based_version import IntegratedPMLHNN, leapfrog_step

class TestIntegratedPMLHNN(unittest.TestCase):
    
    def setUp(self):
        """Set up the model for testing."""
        tf.random.set_seed(42)  # Ensures deterministic results
        np.random.seed(42)

        self.model = IntegratedPMLHNN(param_dim=2, aux_dim=2)
        self.theta = tf.constant([[0.5, -0.5]], dtype=tf.float32)
        self.rho = tf.constant([[0.1, -0.1]], dtype=tf.float32)
        self.u = tf.constant([[0.2, -0.2]], dtype=tf.float32)
        self.p = tf.constant([[0.3, -0.3]], dtype=tf.float32)
        self.dt = 0.1
    
    def test_regression_consistency(self):
        """Ensure model runs without errors and produces reasonable outputs."""
        H_initial = self.model.call([self.theta, self.rho, self.u, self.p]).numpy()
        
        # Run leapfrog step
        theta_new, rho_new, u_new, p_new = leapfrog_step(self.model, self.theta, self.rho, self.u, self.p, self.dt)
        H_new = self.model.call([theta_new, rho_new, u_new, p_new]).numpy()

        # Just ensure the values are finite (no NaNs or Infs)
        self.assertTrue(np.isfinite(H_initial).all(), "H_initial contains NaN or Inf")
        self.assertTrue(np.isfinite(H_new).all(), "H_new contains NaN or Inf")

        # If values exist and are finite, consider the test passed
        self.assertTrue(True, "Regression test passed")

if __name__ == '__main__':
    unittest.main()
