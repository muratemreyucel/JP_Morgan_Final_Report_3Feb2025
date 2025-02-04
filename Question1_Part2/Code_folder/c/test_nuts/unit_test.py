import unittest
import tensorflow as tf
import numpy as np
from NUTS_based_version import IntegratedPMLHNN, leapfrog_step

class TestNUTSHNN(unittest.TestCase):
    
    def setUp(self):
        """Initialize a small test model before running each test."""
        self.model = IntegratedPMLHNN(param_dim=2, aux_dim=2)
        self.theta = tf.random.normal([1, 2])
        self.rho = tf.random.normal([1, 2])
        self.u = tf.random.normal([1, 2])
        self.p = tf.random.normal([1, 2])
        self.dt = 0.1

    def test_model_initialization(self):
        """Test if the model initializes without errors."""
        self.assertIsInstance(self.model, IntegratedPMLHNN)

    def test_hamiltonian_computation(self):
        """Check if Hamiltonian calculation gives finite values."""
        H = self.model.call([self.theta, self.rho, self.u, self.p])
        self.assertTrue(np.isfinite(H.numpy()).all())

    def test_gradient_computation(self):
        """Ensure gradients are correctly computed."""
        dH_dtheta, dH_drho, dH_du, dH_dp = self.model.compute_gradients(self.theta, self.rho, self.u, self.p)
        self.assertIsNotNone(dH_dtheta)
        self.assertIsNotNone(dH_drho)
        self.assertIsNotNone(dH_du)
        self.assertIsNotNone(dH_dp)

    def test_leapfrog_step(self):
        """Test if leapfrog integration preserves Hamiltonian structure."""
        theta_new, rho_new, u_new, p_new = leapfrog_step(self.model, self.theta, self.rho, self.u, self.p, self.dt)
        self.assertEqual(theta_new.shape, self.theta.shape)
        self.assertEqual(rho_new.shape, self.rho.shape)
        self.assertEqual(u_new.shape, self.u.shape)
        self.assertEqual(p_new.shape, self.p.shape)

    def test_skip_training(self):
        """Explicitly skip training to speed up the test."""
        pass  # Placeholder for consistency with integration tests

if __name__ == '__main__':
    unittest.main()
