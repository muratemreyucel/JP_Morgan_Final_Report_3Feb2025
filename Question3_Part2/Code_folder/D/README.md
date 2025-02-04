# README

## Project Overview
This project contains code for synthetic data generation using TensorFlow and TensorFlow Probability.

### Files in the Project
- **`Question3_Part_2_replicating_synthetic_data_generations.py`**
- **`Question3_Part_2_replicating_synthetic_data_generations.ipynb`**
  - These two files are the same, one as a Python script and the other as a Jupyter Notebook.

### Required Libraries
Ensure the following libraries are installed:
```bash
pip install tensorflow tensorflow-probability numpy matplotlib
```

### Tests Structure
The **Three tests files below(py files)** should be in the same folder with `Question3_Part_2_replicating_synthetic_data_generations.py`. 
- **`unit_test.py`** - Unit tests for verifying individual functions.
- **`regression_test.py`** - Ensures that future changes do not break existing functionality.
- **`integration_test.py`** - Validates that multiple components work together correctly.

### Running Tests
To execute tests, navigate to the project directory and run:
```bash
python -m unittest discover tests
```
or you can run in cmd one by one such as(go to the folder using cmd):

python unit_test.py
python regression_test.py
python integration_test.py
