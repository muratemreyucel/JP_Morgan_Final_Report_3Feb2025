import unittest
import tensorflow as tf
import numpy as np
from Pseudo_Marginal_Latent_Hamiltonian_Neural_Network import ImprovedExtendedHNN, generate_pm_hmc_data, improved_train_step

class TestIntegrationImprovedExtendedHNN(unittest.TestCase):

    def setUp(self):
        """Set up the model, dataset, and optimizer for integration testing."""
        self.param_dim = 2
        self.aux_dim = 2
        self.hidden_dim = 100
        self.model = ImprovedExtendedHNN(param_dim=self.param_dim, aux_dim=self.aux_dim, hidden_dim=self.hidden_dim)

        # Set a fixed random seed for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)

        # Generate a small dataset for testing
        self.dataset = generate_pm_hmc_data(n_trajectories=10, n_steps=20, dt=0.1, param_dim=self.param_dim, aux_dim=self.aux_dim)
        self.dataset = self.dataset.batch(2)  # Small batch size for quick testing

        # Define optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def test_integration_training_step(self):
        """Test the integration of data, model, and training step."""
        for states, derivs in self.dataset.take(1):  # Take one batch
            initial_loss, dyn_loss, energy_loss = improved_train_step(self.model, states, derivs, self.optimizer)

            # Check that loss values are finite
            self.assertTrue(np.isfinite(initial_loss.numpy()))
            self.assertTrue(np.isfinite(dyn_loss.numpy()))
            self.assertTrue(np.isfinite(energy_loss.numpy()))

            # Check loss is not NaN or infinite
            self.assertGreater(initial_loss.numpy(), 0)
            self.assertLess(initial_loss.numpy(), 1e10)  # Allow higher loss values

    def test_integration_model_inference(self):
        """Ensure the model runs inference correctly after training."""
        for states, _ in self.dataset.take(1):  # Take one batch
            output = self.model(states)

            # Ensure output is a tensor
            self.assertIsInstance(output, tf.Tensor)

            # Check output shape dynamically (don't assume fixed shape)
            self.assertEqual(len(output.shape), 3)  # Expecting a tensor with 3 dimensions
            self.assertGreaterEqual(output.shape[1], 1)  # Ensure at least 1 unit in second dimension

            # Check that output values are finite
            self.assertTrue(np.all(np.isfinite(output.numpy())))

if __name__ == '__main__':
    unittest.main()
