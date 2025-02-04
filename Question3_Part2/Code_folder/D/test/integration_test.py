import unittest
import tensorflow as tf
import numpy as np
import sys
import os
import pickle

# Ensure the script can find the main module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Question3_Part_2_replicating_synthetic_data_generations import SyntheticDataGenerator

BASELINE_FILE = "regression_baseline.pkl"

class TestSyntheticDataGenerator(unittest.TestCase):
    
    def setUp(self):
        self.generator = SyntheticDataGenerator(seed=42)
        self.sample_size = 5000
    
    def test_generate_data_H1(self):
        X, Y = self.generator.generate_data_H1(self.sample_size)
        self.assertEqual(X.shape, (self.sample_size,))
        self.assertEqual(Y.shape, (self.sample_size,))
        self.assertTrue(np.all(np.isfinite(X.numpy())))
        self.assertTrue(np.all(np.isfinite(Y.numpy())))
    
    def test_generate_data_H0_reverse(self):
        X, Y = self.generator.generate_data_H0_reverse(self.sample_size)
        self.assertEqual(X.shape, (self.sample_size,))
        self.assertEqual(Y.shape, (self.sample_size,))
        self.assertTrue(np.all(np.isfinite(X.numpy())))
        self.assertTrue(np.all(np.isfinite(Y.numpy())))
    
    def test_generate_data_H0_confounded(self):
        X, Y, U = self.generator.generate_data_H0_confounded(self.sample_size)
        self.assertEqual(X.shape, (self.sample_size,))
        self.assertEqual(Y.shape, (self.sample_size,))
        self.assertEqual(U.shape, (self.sample_size,))
        self.assertTrue(np.all(np.isfinite(X.numpy())))
        self.assertTrue(np.all(np.isfinite(Y.numpy())))
        self.assertTrue(np.all(np.isfinite(U.numpy())))
    
    def test_seed_consistency(self):
        tf.random.set_seed(42)  # Reset seed before creating generator
        np.random.seed(42)  # Reset NumPy seed to ensure full reproducibility
        
        gen1 = SyntheticDataGenerator(seed=42)
        X1, Y1 = gen1.generate_data_H1(self.sample_size)
        
        tf.random.set_seed(42)  # Reset seed again
        np.random.seed(42)
        
        gen2 = SyntheticDataGenerator(seed=42)
        X2, Y2 = gen2.generate_data_H1(self.sample_size)
        
        np.testing.assert_array_almost_equal(X1.numpy(), X2.numpy(), decimal=6)
        np.testing.assert_array_almost_equal(Y1.numpy(), Y2.numpy(), decimal=6)
    
    def test_regression(self):
        tf.random.set_seed(42)
        np.random.seed(42)
        
        gen = SyntheticDataGenerator(seed=42)
        X, Y = gen.generate_data_H1(self.sample_size)
        
        new_output = (X.numpy(), Y.numpy())
        
        if os.path.exists(BASELINE_FILE):
            with open(BASELINE_FILE, "rb") as f:
                baseline_output = pickle.load(f)
            np.testing.assert_array_almost_equal(new_output[0], baseline_output[0], decimal=6)
            np.testing.assert_array_almost_equal(new_output[1], baseline_output[1], decimal=6)
        else:
            with open(BASELINE_FILE, "wb") as f:
                pickle.dump(new_output, f)
            print("Baseline regression data saved.")
    
    def test_integration(self):
        tf.random.set_seed(42)
        np.random.seed(42)
        
        gen = SyntheticDataGenerator(seed=42)
        X_h1, Y_h1 = gen.generate_data_H1(self.sample_size)
        X_h0r, Y_h0r = gen.generate_data_H0_reverse(self.sample_size)
        X_h0c, Y_h0c, U = gen.generate_data_H0_confounded(self.sample_size)
        
        # Ensure data shapes are correct
        self.assertEqual(X_h1.shape, (self.sample_size,))
        self.assertEqual(Y_h1.shape, (self.sample_size,))
        self.assertEqual(X_h0r.shape, (self.sample_size,))
        self.assertEqual(Y_h0r.shape, (self.sample_size,))
        self.assertEqual(X_h0c.shape, (self.sample_size,))
        self.assertEqual(Y_h0c.shape, (self.sample_size,))
        self.assertEqual(U.shape, (self.sample_size,))
        
        # Ensure values are finite
        self.assertTrue(np.all(np.isfinite(X_h1.numpy())))
        self.assertTrue(np.all(np.isfinite(Y_h1.numpy())))
        self.assertTrue(np.all(np.isfinite(X_h0r.numpy())))
        self.assertTrue(np.all(np.isfinite(Y_h0r.numpy())))
        self.assertTrue(np.all(np.isfinite(X_h0c.numpy())))
        self.assertTrue(np.all(np.isfinite(Y_h0c.numpy())))
        self.assertTrue(np.all(np.isfinite(U.numpy())))
        
        # Check that changing seed generates different results
        gen_diff = SyntheticDataGenerator(seed=123)
        X_diff, Y_diff = gen_diff.generate_data_H1(self.sample_size)
        
        self.assertFalse(np.array_equal(X_h1.numpy(), X_diff.numpy()))
        self.assertFalse(np.array_equal(Y_h1.numpy(), Y_diff.numpy()))

if __name__ == "__main__":
    unittest.main()