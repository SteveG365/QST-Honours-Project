from dataclasses import dataclass
from typing import Tuple
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import sqrtm
from itertools import product
from functools import reduce


# Utilities
def fidelity(rho1, rho2):
    sqrt_rho1 = sqrtm(rho1)
    F = np.trace(sqrtm(sqrt_rho1 @ rho2 @ sqrt_rho1))
    return np.real(F)**2

def kron_all(mats):
    return reduce(np.kron, mats)

def params_to_rho(d, params):
    # params: length 16
    L = np.zeros((d,d), dtype=complex)
    idx = 0
    # diagonal entries (real, positive)
    for i in range(d):
        L[i, i] = params[idx]
        idx += 1
    # lower-triangular off-diagonals (real + imag)
    for i in range(1, d):
        for j in range(i):
            re = params[idx]; im = params[idx+1]
            L[i, j] = re + 1j * im
            idx += 2
    rho = L @ L.conj().T
    return rho / np.trace(rho)

# POVM Builder 
def build_single_qubit_Us():
    X_cols = [np.array([1, 1])/np.sqrt(2),
              np.array([1,-1])/np.sqrt(2)]
    Y_cols = [np.array([1, 1j])/np.sqrt(2),
              np.array([1,-1j])/np.sqrt(2)] 
    Z_cols = [np.array([1,0]), np.array([0,1])]
    return {'X': np.column_stack(X_cols),
            'Y': np.column_stack(Y_cols),
            'Z': np.column_stack(Z_cols)}


def computational_projectors(d: int):
    proj = []
    for m in range(d):
        P = np.zeros((d, d), dtype=complex)
        P[m, m] = 1.0
        proj.append(P)
    return proj


def build_povm(n_qubits: int):
    Us1 = build_single_qubit_Us()
    settings = []
    for bases in product(['X','Y','Z'], repeat=n_qubits):
        U = kron_all([Us1[b] for b in bases])
        settings.append(U)

    # computational-basis projectors
    proj = []
    for m in range(2**n_qubits):
        P = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
        P[m, m] = 1.0
        proj.append(P)

    # rotated projectors
    E = []
    for U in settings:
        U_dag = U.conj().T
        for P in proj:
            E.append(U @ P @ U_dag)
    E = np.asarray(E)
    E = np.ascontiguousarray(E)
    return E

def probs_from_rho(rho: np.ndarray, E: np.ndarray) -> np.ndarray:
    """
    Compute the probabilities from the density matrix and POVM elements.
    """
    return np.einsum('kij,ji->k', E, rho).real


def neg_log_likelihood(params: np.ndarray, counts: np.ndarray, d: int, E: np.ndarray) -> float:
    rho = params_to_rho(d, params)
    # probs for each rotated projector
    probs = probs_from_rho(rho, E)
    # numerical safety
    probs = np.clip(probs, 1e-15, 1.0)
    return -np.sum(counts * np.log(probs))


# MLE Wrapper

@dataclass
class MLEQST:
    n_qubits: int
    maxiter: int = 500
    ftol: float = 1e-7
    gtol: float = 1e-7
    method: str = 'L-BFGS-B'

    def __post_init__(self):
        self.d = 2 ** self.n_qubits
        self.E = build_povm(self.n_qubits)
        self.n_params = self.d * self.d
        self.K = 2**self.n_qubits * 3**self.n_qubits

    def _init_params(self) -> np.ndarray:
        """Cholesky init close to maximally mixed: diag = 1/sqrt(d), others = 0."""
        init = np.zeros(self.d * self.d, dtype=float)
        init[:self.d] = 1.0 / np.sqrt(self.d)
        return init
    def fit_single(self, counts: np.ndarray) -> np.ndarray:
        """
        MLE for a single experiment:
          counts: shape (K,) matched to the POVM (from my generator thats Pauli Tensor POVM)
        Returns: rho_hat (d x d).
        """
        init = self._init_params()
        res = minimize(
            neg_log_likelihood,
            init,
            args=(counts, self.d, self.E),
            method=self.method,
            options={'maxiter': self.maxiter, 'ftol': self.ftol, 'gtol': self.gtol}
        )
        return params_to_rho(self.d, res.x)

    def fit_batch(self, counts_batch: np.ndarray) -> np.ndarray:
        """
        Vector of experiments:
          counts_batch: shape (N, K) or (N, S, O).
        Returns: rho_est (N, d, d).
        """
        counts_batch = np.asarray(counts_batch)
        if counts_batch.ndim == 2 and counts_batch.shape[1] != self.K:
            # maybe given as (N, S*O)? else could be (N, S, O)
            pass
        N = counts_batch.shape[0]
        rho_est = np.zeros((N, self.d, self.d), dtype=complex)
        for i in range(N):
            rho_est[i] = self.fit_single(counts_batch[i])
        return rho_est
    
    def avg_fidelity(self, counts_batch: np.ndarray, y_true: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """
        Reconstruct all states and compute fidelities to y_true for a batch of states
        Returns (mean_fid, std_fid, per_example_fidelities)
        """
        rho_est = self.fit_batch(counts_batch)
        N = rho_est.shape[0]
        fids = np.empty(N, dtype=float)
        for i in range(N):
            fids[i] = fidelity(rho_est[i], y_true[i])
        return float(fids.mean()), float(fids.std(ddof=0)), fids

    
