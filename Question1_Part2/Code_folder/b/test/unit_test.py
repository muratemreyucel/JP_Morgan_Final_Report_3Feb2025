import unittest
import tensorflow as tf
import numpy as np
from Pseudo_Marginal_Latent_Hamiltonian_Neural_Network import ImprovedExtendedHNN, leapfrog_step

class TestImprovedExtendedHNN(unittest.TestCase):

    def setUp(self):
        """Set up a model and test inputs for unit tests"""
        self.param_dim = 2
        self.aux_dim = 2
        self.hidden_dim = 100
        self.model = ImprovedExtendedHNN(param_dim=self.param_dim, aux_dim=self.aux_dim, hidden_dim=self.hidden_dim)

        # Create dummy input data
        self.batch_size = 4
        self.theta = tf.random.normal([self.batch_size, self.param_dim])
        self.rho = tf.random.normal([self.batch_size, self.param_dim])
        self.u = tf.random.normal([self.batch_size, self.aux_dim])
        self.p = tf.random.normal([self.batch_size, self.aux_dim])

    def test_model_initialization(self):
        """Test if the model initializes correctly"""
        self.assertIsInstance(self.model, ImprovedExtendedHNN)
        self.assertEqual(self.model.total_dim, self.param_dim + self.aux_dim)

    def test_forward_pass(self):
        """Test if the model can compute a Hamiltonian value"""
        output = self.model([self.theta, self.rho, self.u, self.p])
        self.assertEqual(output.shape, (self.batch_size, 1))
        self.assertTrue(np.all(np.isfinite(output.numpy())))  # Ensure no NaN/inf values

    def test_kinetic_energy(self):
        """Test kinetic energy computation"""
        T = self.model.kinetic_energy(self.rho, self.u, self.p)
        self.assertEqual(T.shape, (self.batch_size, 1))
        self.assertTrue(np.all(T.numpy() >= 0))  # Kinetic energy should be non-negative

    def test_compute_gradients(self):
        """Test if the model correctly computes gradients"""
        dH_dtheta, dH_drho, dH_du, dH_dp = self.model.compute_gradients(self.theta, self.rho, self.u, self.p)
        
        self.assertEqual(dH_dtheta.shape, self.theta.shape)
        self.assertEqual(dH_drho.shape, self.rho.shape)
        self.assertEqual(dH_du.shape, self.u.shape)
        self.assertEqual(dH_dp.shape, self.p.shape)

    def test_leapfrog_step(self):
        """Test the leapfrog integration step"""
        dt = 0.1
        theta_new, rho_new, u_new, p_new = leapfrog_step(self.model, self.theta, self.rho, self.u, self.p, dt)

        self.assertEqual(theta_new.shape, self.theta.shape)
        self.assertEqual(rho_new.shape, self.rho.shape)
        self.assertEqual(u_new.shape, self.u.shape)
        self.assertEqual(p_new.shape, self.p.shape)

if __name__ == '__main__':
    unittest.main()
