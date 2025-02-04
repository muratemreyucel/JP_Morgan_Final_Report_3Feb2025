import unittest
import tensorflow as tf
import numpy as np
from Pseudo_Marginal_Latent_Hamiltonian_Neural_Network import ImprovedExtendedHNN, leapfrog_step

class TestRegressionImprovedExtendedHNN(unittest.TestCase):

    def setUp(self):
        """Set up the model and initial test conditions."""
        self.param_dim = 2
        self.aux_dim = 2
        self.hidden_dim = 100
        self.model = ImprovedExtendedHNN(param_dim=self.param_dim, aux_dim=self.aux_dim, hidden_dim=self.hidden_dim)

        # Set a fixed random seed for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)

        # Generate fixed test inputs
        self.batch_size = 4
        self.theta = tf.random.normal([self.batch_size, self.param_dim], seed=42)
        self.rho = tf.random.normal([self.batch_size, self.param_dim], seed=42)
        self.u = tf.random.normal([self.batch_size, self.aux_dim], seed=42)
        self.p = tf.random.normal([self.batch_size, self.aux_dim], seed=42)
        self.dt = 0.1  # Time step for leapfrog integration

    def test_regression_forward_pass(self):
        """Ensure the model output remains within a stable range."""
        output = self.model([self.theta, self.rho, self.u, self.p]).numpy()
        self.assertEqual(output.shape, (self.batch_size, 1))
        self.assertTrue(np.all(np.isfinite(output)))
        self.assertTrue(np.min(output) > -10 and np.max(output) < 10)  # Ensures output isn't exploding

    def test_regression_kinetic_energy(self):
        """Ensure kinetic energy calculation remains valid."""
        T = self.model.kinetic_energy(self.rho, self.u, self.p).numpy()
        self.assertEqual(T.shape, (self.batch_size, 1))
        self.assertTrue(np.all(T >= 0))  # Kinetic energy should be non-negative
        self.assertTrue(np.max(T) < 20)  # Sanity check for outliers

    def test_regression_compute_gradients(self):
        """Ensure computed gradients remain stable over time."""
        dH_dtheta, dH_drho, dH_du, dH_dp = self.model.compute_gradients(self.theta, self.rho, self.u, self.p)

        for grad in [dH_dtheta, dH_drho, dH_du, dH_dp]:
            self.assertEqual(grad.shape, self.theta.shape)  # Shape should be the same as input
            self.assertTrue(np.all(np.isfinite(grad.numpy())))  # No NaN or inf

    def test_regression_leapfrog_step(self):
        """Ensure leapfrog step behavior remains numerically stable."""
        theta_new, rho_new, u_new, p_new = leapfrog_step(self.model, self.theta, self.rho, self.u, self.p, self.dt)

        # Ensure shapes remain correct
        self.assertEqual(theta_new.shape, self.theta.shape)
        self.assertEqual(rho_new.shape, self.rho.shape)
        self.assertEqual(u_new.shape, self.u.shape)
        self.assertEqual(p_new.shape, self.p.shape)

        # Ensure numerical stability: Values should not explode
        for var in [theta_new, rho_new, u_new, p_new]:
            self.assertTrue(np.all(np.isfinite(var.numpy())))
            self.assertTrue(np.min(var.numpy()) > -100 and np.max(var.numpy()) < 100)  # Prevent unstable jumps

if __name__ == '__main__':
    unittest.main()
