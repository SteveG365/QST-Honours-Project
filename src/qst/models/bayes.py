import numpy as np
from itertools import product
from functools import reduce
from dataclasses import dataclass
from typing import Optional, Tuple


def kron_all(mats):
    return reduce(np.kron, mats)

def _single_qubit_eigenbases() -> dict:
    X_cols = [np.array([1, 1]) / np.sqrt(2),
              np.array([1,-1]) / np.sqrt(2)]
    Y_cols = [np.array([1, 1j]) / np.sqrt(2),
              np.array([1,-1j]) / np.sqrt(2)]
    Z_cols = [np.array([1, 0]), np.array([0, 1])]
    return {
        "X": np.column_stack(X_cols),
        "Y": np.column_stack(Y_cols),
        "Z": np.column_stack(Z_cols),
    }

def _computational_projectors(d: int):
    """Projectors in the computational basis for dimension d=2^n."""
    proj = []
    for m in range(d):
        P = np.zeros((d, d), dtype=complex)
        P[m, m] = 1.0
        proj.append(P)
    return proj

def cholesky_param_dim(d: int) -> int:
    """
    Number of real parameters for Cholesky parameterization:
    - d real diagonals
    - strictly lower triangle: d(d-1)/2 complex entries = d(d-1) reals
    Total = d + d(d-1) = d^2
    """
    return d * d

def params_to_rho(params: np.ndarray, d: int) -> np.ndarray:
    """Map d^2 real params -> valid dxd density matrix via lower-triangular Cholesky."""
    if params.size != d*d:
        raise ValueError(f"Expected {d*d} params for d={d}, got {params.size}")
    L = np.zeros((d, d), dtype=complex)
    idx = 0
    # Diagonals (real, non-negative when squared)
    for i in range(d):
        L[i, i] = params[idx]
        idx += 1
    # Lower-triangular off-diagonals (complex)
    for i in range(1, d):
        for j in range(i):
            re = params[idx]; im = params[idx + 1]
            L[i, j] = re + 1j * im
            idx += 2
    rho = L @ L.conj().T
    tr = np.real_if_close(np.trace(rho))
    if tr <= 0:
        # Guard against pathological initializations
        rho = rho + 1e-12 * np.eye(d)
        tr = np.trace(rho)
    return rho / tr

def _as_counts_matrix(counts: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    Ensure counts has shape (3^n, 2^n). If 1D of length 6^n, reshape.
    """
    counts = np.asarray(counts)
    S = 3 ** n_qubits
    O = 2 ** n_qubits
    if counts.ndim == 1:
        if counts.size != S * O:
            raise ValueError(f"Flat counts must have length {S*O} (got {counts.size}).")
        return counts.reshape(S, O)
    elif counts.ndim == 2:
        if counts.shape != (S, O):
            raise ValueError(f"Counts matrix must be shape {(S, O)} (got {counts.shape}).")
        return counts
    else:
        raise ValueError("counts must be 1D or 2D array.")


def build_pauli_povm(n_qubits: int) -> Tuple[np.ndarray, int, int]:
    """
    Build the full Pauli-tensor POVM for n_qubits.
    Returns:
        E_blocks: ndarray with shape (S, O, d, d) where
                  S = 3^n settings, O = 2^n outcomes per setting, d=2^n
        S: number of settings
        O: number of outcomes per setting
    """
    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1")
    d = 2 ** n_qubits
    S = 3 ** n_qubits
    O = 2 ** n_qubits

    eig = _single_qubit_eigenbases()
    settings = list(product(["X", "Y", "Z"], repeat=n_qubits))
    comp_proj = _computational_projectors(d)
    E_blocks = np.zeros((S, O, d, d), dtype=complex)
    for s_idx, setting in enumerate(settings):
        U_local = [eig[b] for b in setting]
        U = kron_all(U_local)
        U_dag = U.conj().T
        for m, P in enumerate(comp_proj):
            E_blocks[s_idx, m] = U @ P @ U_dag
    return E_blocks, S, O


# Prior, likelihood, posterior

def log_prior(params: np.ndarray) -> float:
    """Uninformative prior"""
    return 0.0

def probs_per_setting(rho: np.ndarray, E_blocks: np.ndarray) -> np.ndarray:
    """
    Compute probabilities for each setting block; shape (S, O).
    Each row sums to ~1 by construction (numerical eps aside).
    """
    S, O, d, _ = E_blocks.shape
    p = np.real(np.einsum("sodc,cd->so", E_blocks, rho))
    p = np.clip(p, 1e-12, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    return p

def log_likelihood(params: np.ndarray, counts: np.ndarray, E_blocks: np.ndarray, n_qubits: int) -> float:
    """
    Multinomial log-likelihood across all settings
    """
    d = 2 ** n_qubits
    rho = params_to_rho(params, d)
    p_s = probs_per_setting(rho, E_blocks) 
    Counts = _as_counts_matrix(counts, n_qubits) 
    return float((Counts * np.log(p_s)).sum())

def log_posterior(params: np.ndarray, counts: np.ndarray, E_blocks: np.ndarray, n_qubits: int) -> float:
    return log_prior(params) + log_likelihood(params, counts, E_blocks, n_qubits)


@dataclass
class BayesQST:
    n_qubits: int
    E_blocks: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.E_blocks is None:
            self.E_blocks, _, _ = build_pauli_povm(self.n_qubits)
        self.d = 2 ** self.n_qubits
        self.D = self.d * self.d

    def init_params_maximally_mixed(self) -> np.ndarray:
        """
        Cholesky for maximally mixed: choose L diagonal entries = 1/sqrt(d).
        Off-diagonals = 0.
        """
        params = np.zeros(self.D, dtype=float)
        params[:self.d] = 1.0 / np.sqrt(self.d)
        return params

    def metropolis_sampler(
        self,
        counts: np.ndarray,
        n_samples: int = 2000,
        proposal_std: float = 0.01,
        burn_in: int = 1000,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """
        Basic Metropolis-Hastings over unconstrained Cholesky params
        Returns array of shape (n_samples, D), and the acceptance ratio
        """
        rng = np.random.default_rng(random_state)
        
        current_params = self.init_params_maximally_mixed()
        curr_post = log_posterior(current_params, counts, self.E_blocks, self.n_qubits)
        samples = np.zeros((n_samples, self.D), dtype=float)
        kept = 0
        total = burn_in + n_samples
        accepted_counter = 0
        
        for t in range(total):
            proposal_params = current_params + rng.normal(scale=proposal_std, size=self.D)
            
            prop_post = log_posterior(proposal_params, counts, self.E_blocks, self.n_qubits)
            r = np.exp(prop_post - curr_post)
            if (rng.uniform() < r) :
                current_params = proposal_params
                curr_post = prop_post
                accepted_counter +=1

            if t >= burn_in:
                samples[kept] = current_params
                kept += 1

        return samples, accepted_counter / total


    def batch_metropolis_sampler(
            self,
            counts: np.ndarray,
            n_samples : int = 1000,
            proposal_std : float = 0.01 ,
            burn_in : int = 100,
            random_state : int = 42):
        """
        Function for performing metropolis hastings sampling on a batch of data
        counts: shape = (N, D) 
        returns tensor of shape (N, n_samples, D)
        """
        N = counts.shape[0]
        D = self.D
        accept_counter = 0
        rng = np.random.default_rng(random_state)
        samples = np.zeros(shape=(N, n_samples, D), dtype=float)
        
        total = n_samples + burn_in

        for i in range(N):
            kept = 0
            current_params = self.init_params_maximally_mixed()
            curr_post = log_posterior(current_params, counts[i], self.E_blocks, self.n_qubits)
            for t in range(total):

                proposal_params = current_params + rng.normal(scale=proposal_std, size=self.D)

                prop_post = log_posterior(proposal_params, counts[i], self.E_blocks, self.n_qubits)

                r = np.exp(prop_post - curr_post)
                if (rng.uniform() < r):
                    #accept
                    current_params = proposal_params
                    curr_post = prop_post
                    accept_counter += 1
                
                if (t >= burn_in):
                    samples[i, kept] = current_params
                    kept += 1

        return samples, accept_counter / (total * N)



    def params_to_rho(self, params: np.ndarray) -> np.ndarray:
        return params_to_rho(params, self.d)
