import unittest
import tensorflow as tf
import numpy as np
from NUTS_based_version import IntegratedPMLHNN, leapfrog_step

class TestIntegratedPMLHNN(unittest.TestCase):
    
    def setUp(self):
        """Set up the model for testing."""
        self.model = IntegratedPMLHNN(param_dim=2, aux_dim=2)
        self.theta = tf.random.normal([1, 2])
        self.rho = tf.random.normal([1, 2])
        self.u = tf.random.normal([1, 2])
        self.p = tf.random.normal([1, 2])
        self.dt = 0.1
    
    def test_model_initialization(self):
        """Test if the model initializes correctly."""
        self.assertIsInstance(self.model, IntegratedPMLHNN)
    
    def test_forward_pass(self):
        """Test the forward pass of the model."""
        H = self.model.call([self.theta, self.rho, self.u, self.p])
        self.assertEqual(H.shape, (1, 1))  # Should return a scalar per batch
    
    def test_compute_gradients(self):
        """Test gradient computation without errors."""
        dH_dtheta, dH_drho, dH_du, dH_dp = self.model.compute_gradients(self.theta, self.rho, self.u, self.p)
        self.assertEqual(dH_dtheta.shape, self.theta.shape)
        self.assertEqual(dH_drho.shape, self.rho.shape)
        self.assertEqual(dH_du.shape, self.u.shape)
        self.assertEqual(dH_dp.shape, self.p.shape)
    
    def test_leapfrog_step(self):
        """Test the leapfrog integration step."""
        theta_new, rho_new, u_new, p_new = leapfrog_step(self.model, self.theta, self.rho, self.u, self.p, self.dt)
        self.assertEqual(theta_new.shape, self.theta.shape)
        self.assertEqual(rho_new.shape, self.rho.shape)
        self.assertEqual(u_new.shape, self.u.shape)
        self.assertEqual(p_new.shape, self.p.shape)
    
    def test_skip_training(self):
        """Ensure the training function is not called during this test."""
        # Training function should not be executed in this test suite
        pass  # Placeholder to explicitly show training is skipped
    
if __name__ == '__main__':
    unittest.main()
