# QST-Honours-Project

This project implements and compares 3 different statistical reconstruction regimes for Quantum State Tomography. These are Maximum Likelihood Estimation (MLE), Bayesian Mean Estimation (BME), and a Deep Neural Network (DNN). The code is designed to be modular, easy to extend, and suitable for benchmarking different QST methods under controlled noise models.

## Overview

The goal of this project is to reconstruct unknown quantum states represented by density matrices $\rho$ from simulated measurement data. The pipeline supports:

### State sampling:
- Haar-random and Hilbert–Schmidt–random state generation
- GHZ-family state generation with adjustable phase and dephasing
- Pauli-tensor measurement simulation with optional noise

### Statistical Reconstruction methods:
- MLE reconstruction using a Cholesky parameterisation
- Bayesian sampling using Metropolis–Hastings
- DNN-based QST and transfer learning
- Fidelity and eigenvalue-based evaluation tools


## Scientific Investigation Findings

The aim of this work was to evaluate each of the reconstruction methods under a range of realistic noise
conditions. The MLE approach was found to frequently produce density matrices with zero eigenvalues,
potentially leading experimentalists to wrongfully conclude that certain outcomes may be impossible.
In contrast, the BME methodology avoids this pathology by averaging over the posterior distribution,
which yields more sensible predictions. However, both MLE and BME degrade significantly in the pres-
ence of sampling noise and measurement miscalibration. The DNN model demonstrates superior ro-
bustness under these noise sources, while also offering improved computational scalability. Finally, a
novel transfer-learning extension was developed for reconstructing Greenberger–Horne–Zeilinger (GHZ)
states, achievingenhanced computational efficiency and the best reconstructiona ccuracy across all tested
methods.

## Author
Steven Richard Glass
25880586@sun.ac.za
Stellenbosch University
