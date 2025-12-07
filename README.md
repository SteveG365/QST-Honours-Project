# QST-Honours-Project

This project implements and compares 3 different statistical reconstruction regimes for Quantum State Tomography. These are Maximum Likelihood Estimation (MLE), Bayesian Mean Estimation (BME), and a Deep Neural Network (DNN). The code is designed to be modular, easy to extend, and suitable for benchmarking different QST methods under controlled noise models.

## Overview

The goal of this project is to reconstruct unknown quantum states from simulated measurement data. The pipeline supports:

### State sampling:
- Haar-random and Hilbert–Schmidt–random state generation
- GHZ-family state generation with adjustable phase and dephasing
- Pauli-tensor measurement simulation with optional noise

### Statistical Reconstruction methods:
- MLE reconstruction using a Cholesky parameterisation
- Bayesian sampling using Metropolis–Hastings
- DNN-based QST and transfer learning
- Fidelity and eigenvalue-based evaluation tools


## Author
Steven Richard Glass
25880586@sun.ac.za
Stellenbosch University
