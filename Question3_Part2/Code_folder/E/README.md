# README

## Project Overview
This project contains code for optimizing causal interventions using a Bayesian approach.

### Files in the Project
- **`multiple_variables.py`** - The main script containing the `UnifiedCausalOptimizer` class.
- **`multiple_variables.ipynb`** - A Jupyter Notebook version of the script.

### Required Libraries
Ensure the following libraries are installed:
```bash
pip install tensorflow tensorflow-probability numpy matplotlib networkx seaborn
```

### Tests Structure
The **`tests`** folder should be in the same directory as `multiple_variables.py`. It contains:
- **`unit_test.py`** - Unit tests for verifying individual functions.
- **`regression_test.py`** - Ensures that future changes do not break existing functionality.
- **`integration_test.py`** - Validates that multiple components work together correctly.

### Running Tests
To execute tests, navigate to the project directory and run:
```bash
python -m unittest discover tests
```
