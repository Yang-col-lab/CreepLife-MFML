Coupling physics-based creep calculations with multi-fidelity learning for creep rupture life prediction of nickel-based superalloys

This repository provides a multi-fidelity machine learning framework for predicting the creep rupture life of nickel-based superalloys. Four models are implemented: a single-fidelity Gaussian process regression model (SF_GPR), a multi-fidelity co-kriging model (MFCK), and two Δ-machine learning models (DELTA1 and DELTA2).

The repository includes the following files:

1. data_file

raw_data.xlsx: Contains the dataset, including alloy compositions, manufacturing processes, testing conditions, calculated creep rupture life (simcreep), and experimental creep rupture life (expcreep). Both calculated and experimental creep rupture lives are provided on the log10 scale.

2. prediction_model

models.py: Trains and evaluates one single-fidelity baseline model and three multi-fidelity learning models using 10-fold cross-validation with different proportions of available experimental (high-fidelity) training data.

3. result

0.05.xlsx / 0.3.xlsx: Contain prediction results obtained from a single 10-fold cross-validation run with experimental data ratios of 0.05 and 0.30, respectively, including model performance, individual predictions, and uncertainty estimates.

Note: The data_file contains a subset of the original dataset. The full dataset is available from the corresponding author upon reasonable request. The script supports additional experimental data ratios beyond 0.05 and 0.30.
