import unittest
import numpy as np
from generalized_replication import PseudoMarginalGLMM, generate_glmm_data

class TestPseudoMarginalGLMM(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.n_subjects = 10
        cls.n_covariates = 5
        cls.n_obs_per_subject = 6
        cls.n_particles = 10
        
        # Generate synthetic data for testing
        X, y, _, _ = generate_glmm_data(n_subjects=cls.n_subjects, n_obs_per_subject=cls.n_obs_per_subject, n_covariates=cls.n_covariates)
        
        cls.model = PseudoMarginalGLMM(X, y, cls.n_subjects, cls.n_covariates, cls.n_obs_per_subject, cls.n_particles)
    
    def test_initialization(self):
        """Test if the model initializes with correct shapes and default values."""
        self.assertEqual(self.model.X.shape[1], self.n_covariates)
        self.assertEqual(len(self.model.y), self.n_subjects * self.n_obs_per_subject)
        self.assertEqual(self.model.beta.shape[0], self.n_covariates)
        self.assertEqual(len(self.model.subject_indices), self.n_subjects * self.n_obs_per_subject)
        self.assertEqual(self.model.mixture_weights.shape[0], 2)
        self.assertEqual(self.model.mixture_means.shape[0], 2)
        self.assertEqual(self.model.mixture_precisions.shape[0], 2)
    
    def test_log_prior(self):
        """Test log-prior computation for beta."""
        beta = np.random.randn(self.n_covariates)
        log_prior_value = self.model.log_prior(beta)
        expected_value = -0.5 * np.sum(beta**2 / 10000)
        self.assertAlmostEqual(log_prior_value, expected_value, places=6)
    
    def test_importance_sampling_likelihood(self):
        """Test that the importance sampling likelihood returns a finite value."""
        beta = np.random.randn(self.n_covariates)
        random_effects = np.random.randn(self.n_subjects)
        log_likelihood = self.model.importance_sampling_likelihood(beta, random_effects)
        self.assertTrue(np.isfinite(log_likelihood))
    
    def test_log_random_effects_prior(self):
        """Test the log prior computation for random effects."""
        random_effects = np.random.randn(self.n_subjects)
        log_re_prior = self.model.log_random_effects_prior(random_effects)
        self.assertTrue(np.isfinite(log_re_prior))
    
    def test_pm_hmc_sample(self):
        """Test that the PM-HMC sampling runs without errors and produces reasonable output shapes."""
        beta_samples, re_samples = self.model.pm_hmc_sample(n_iter=100, n_warmup=20, step_size=0.01)
        self.assertEqual(beta_samples.shape[1], self.n_covariates)
        self.assertEqual(re_samples.shape[1], self.n_subjects)
    
    def test_regression_output_consistency(self):
        """Regression test to ensure model outputs remain within a reasonable range over time."""
        np.random.seed(42)
        beta_samples, re_samples = self.model.pm_hmc_sample(n_iter=100, n_warmup=20, step_size=0.01)
        
        beta_mean = np.mean(beta_samples, axis=0)
        re_mean = np.mean(re_samples, axis=0)
        
        print("Beta means:", beta_mean)
        print("Random effects means:", re_mean)
        
        self.assertTrue(np.all(np.abs(beta_mean) < 3))  # Allowing reasonable range
        self.assertTrue(np.all(np.abs(re_mean) < 3))  # Allowing reasonable range

if __name__ == '__main__':
    unittest.main()