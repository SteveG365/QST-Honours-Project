import numpy as np
import itertools
from collections import defaultdict
import argparse
from functools import reduce
from qiskit.quantum_info import random_statevector, random_density_matrix
np.random.seed(42)

def generate_states(n_qubits, n_states, pure_fraction=0.5, seed=42):
    """
    Generates a mix of pure and mixed states for the generalised training dataset
    Pure: generated using Qiskit's random_statevector, uses the Haar random Unitary
    Mixed: Uses Qiskit's random_density_matrix, and generates according to the Hilbert-Schidt measure
    """
    np.random.seed(seed)
    n_pure = int(round(n_states * pure_fraction))
    n_mixed = n_states - n_pure
    states = []
    dim = 2**n_qubits

    #generate pure state
    for i in range(n_pure):
        psi = random_statevector(dims= dim)
        psi = np.asarray(psi.data, dtype=complex).reshape(dim)
        rho = np.outer(psi, psi.conj())
        states.append(rho)

    #generate mixed states
    for i in range(n_mixed):
        rho = random_density_matrix(dims=dim)
        rho = np.asarray(rho.data, dtype=complex)
        states.append(rho)
    return states


def generate_mixed_ghz_states(n_qubits, num_states, 
                              phi_range=(0, 2*np.pi), p_range=(0.0, 0.5),
                              seed=42):
    """
    Generate a list of mixed GHZ density matrices for n_qubits.
    """
    rng = np.random.default_rng(seed)
    dim = 2**n_qubits
    states = []

    for i in range(num_states):
        # Sample random phase and dephasing strength
        phi = rng.uniform(*phi_range)
        p = rng.uniform(*p_range)

        # Build pure GHZ 
        psi = np.zeros(dim, dtype=complex)
        psi[0] = 1/np.sqrt(2)
        psi[-1] = np.exp(1j*phi)/np.sqrt(2)

        rho_pure = np.outer(psi, psi.conj())

        # Build dephased mixture
        rho_diag = np.zeros((dim, dim), dtype=complex)
        rho_diag[0,0] = 0.5
        rho_diag[-1,-1] = 0.5

        rho_mixed = (1-p)*rho_pure + p*rho_diag

        states.append(rho_mixed)

    return states

def generate_mixed_purity_states(n_qubits, n_states, purity_range=(0.0, 1.0), seed=42):
    """
    Generate a list of mixed states with specified purity range for n_qubits.
    """
    rng = np.random.default_rng(seed)
    dim = 2**n_qubits
    states = []

    P_min_valid = 1.0 / dim
    P_low = max(purity_range[0], P_min_valid)
    P_high = min(purity_range[1], 1.0)
    
    purities = np.linspace(P_low, P_high, n_states)
    for P in purities:
        
        psi = random_statevector(dim, seed=rng)
        psi = np.asarray(psi.data, dtype=complex).reshape(dim)
        rho = np.outer(psi, psi.conj())
        p = np.sqrt((P - 1/dim)/(1 - 1/dim))

        rho_p = p * rho + (1-p)/dim * np.eye(dim)
        states.append(rho_p)


    return states, P_low, P_high

_PAULI_EIGENVECTORS = {
    'X': {
        0: np.array([[1/np.sqrt(2)], [ 1/np.sqrt(2)]]),
        1: np.array([[1/np.sqrt(2)], [-1/np.sqrt(2)]])
    },
    'Y': {
        0: np.array([[1/np.sqrt(2)], [ 1j/np.sqrt(2)]]),
        1: np.array([[1/np.sqrt(2)], [-1j/np.sqrt(2)]])
    },
    'Z': {
        0: np.array([[1.0], [0.0]]),
        1: np.array([[0.0], [1.0]])
    }
}

def single_qubit_projector(pauli, outcome):
    """
    Return the projector |i><i| for the given Pauli basis and outcome i
    """
    v = _PAULI_EIGENVECTORS[pauli][outcome]
    return v @ v.conj().T


def generate_pauli_povm(n_qubits):
    """
    Build the full tensor-product Pauli 'cube' POVM for n_qubits.

    Returns:
      povm_elems: list of projectors (length = 3^n * 2^n)
      labels: list of (basis_tuple, outcome_tuple)
      bases: sorted list of unique basis tuples (each basis has 2^n outcomes)
    """
    bases = list(itertools.product(['X', 'Y', 'Z'], repeat=n_qubits))
    povm_elems = []
    labels = []
    for basis in bases:
        for outcome in itertools.product([0, 1], repeat=n_qubits):
            ops = [single_qubit_projector(basis[i], outcome[i]) for i in range(n_qubits)]
            E = ops[0]
            for op in ops[1:]:
                E = np.kron(E, op)
            povm_elems.append(E)
            labels.append((basis, outcome))
    return povm_elems, labels, bases

def group_indices_by_basis(labels):
    """
    Map basis_tuple -> list of indices into the flat POVM list for that basis.
    Each basis has 2^n outcomes.
    """
    idx = defaultdict(list)
    for i, (basis, outcome) in enumerate(labels):
        idx[basis].append(i)
    # Ensure deterministic order of outcomes within a basis
    for b in idx:
        idx[b] = sorted(idx[b], key=lambda i: labels[i][1])
    return idx

def simulate_counts_grouped_by_basis(rho, povm, labels, bases, shots_per_basis):
    """
    For each basis, compute probabilities over its 2^n outcomes, then draw a
    multinomial with 'shots_per_basis' shots. Concatenate counts across all bases.
    This matches Ma's shots per measurement set' idea.
    """
    basis_to_indices = group_indices_by_basis(labels)

    all_counts = []
    for basis in bases:
        indices = basis_to_indices[basis]  # length = 2^n
        probs = np.array([np.real(np.trace(rho @ povm[i])) for i in indices], dtype=float)
        probs = np.clip(probs, 0, None)
        probs = probs / probs.sum()
        counts = np.random.multinomial(shots_per_basis, probs)
        all_counts.append(counts)
    return np.concatenate(all_counts, axis=0)

def sample_Ue(xi, n, seed):
    np.random.seed(seed)
    angles = np.random.normal(0, xi, 3)
    w1 = angles[0]; w2 = angles[1]; w3 = angles[2]
    Ue = np.array([[np.exp(1j * w1/2) * np.cos(w3),     -1j * np.exp(1j * w2) * np.sin(w3)],
                   [-1j * np.exp(-1j * w2) * np.sin(w3), np.exp(-1j * w1/2) * np.cos(w3) ]], dtype=complex)
    Ue_t = reduce(np.kron, [Ue] * n)
    return Ue_t

def rotate_state(rho, xi, n, seed):
    Ue = sample_Ue(xi, n, seed)
    rho_eff = Ue @ rho @ Ue.conj().T
    return rho_eff

def rotate_povm(povm, xi, n, seed):
    Ue = sample_Ue(xi, n, seed)
    povm_rotated = [Ue.conj().T @ E @ Ue for E in povm]
    return povm_rotated

def generate_dataset(
    n_qubits,
    n_states,
    shots_per_basis,
    pure_fraction=0.5,
    purity_range=None,
    noise=False,
    xi=0.0,
    noise_mode ='state',
    ghz=False,
    seed=42):
    """
    Generate dataset of measurement counts and true states.

    Args:
      n_qubits: number of qubits
      n_states: number of states
      shots_per_basis: S (copies per measurement basis)
      pure_fraction: fraction of pure (Haar) states vs Ginibre mixed
      purity_range: tuple (min, max) purity for generating mixed states
      noise: bool, whether to add coherent measurement noise
      xi: float noise ratio (ignored if noise=False)
      noise_mode: 'state' or 'basis', whether to misalign the state or the measurement basis
      ghz: bool, whether to generate GHZ states
      seed: random seed, default 42 
    """
    np.random.seed(seed)

    if (ghz == True):
        states = generate_mixed_ghz_states(n_qubits, n_states, seed=seed)
    elif (purity_range != None):
        states, P_low, P_high = generate_mixed_purity_states(n_qubits, n_states, purity_range, seed=seed)
    else:
        states = generate_states(n_qubits, n_states, pure_fraction, seed=seed)

    povm, labels, bases = generate_pauli_povm(n_qubits)

    counts_list = []

    if noise and noise_mode == 'basis':
        #misalign the measurement basis once
        povm = rotate_povm(povm, xi, n_qubits, seed=seed)

    for rho in states:

        if noise and noise_mode == 'state':
            #misalign rho
            rho_eff = rotate_state(rho, xi, n_qubits, seed=seed)
            c = simulate_counts_grouped_by_basis(rho_eff, povm, labels, bases, shots_per_basis)
        else:
            c = simulate_counts_grouped_by_basis(rho, povm, labels, bases, shots_per_basis)
        counts_list.append(c)

    counts = np.vstack(counts_list)
    return {
        'counts': counts,
        'states': np.stack(states),
        'labels': labels,
        'bases': bases,
        'n_qubits': n_qubits,
        'n_states': n_states,
        'shots_per_basis': shots_per_basis,
        'pure_fraction': pure_fraction,
        'purity_range': (P_low, P_high),
        'noise_enabled': noise,
        'xi': xi,
        'noise_mode': noise_mode,
        'ghz': ghz,
        'seed': seed
    }



if __name__ == "__main__":
    """
    How to run: python pure_mixed_build.py --n_qubits 2 --n_states 500 --shots_per_basis 100 --pure_fraction 0.5 
                                            --noise --xi 0.05 --noise_mode state --ghz False --output dataset.npz

    """
    parser = argparse.ArgumentParser(description="Generate QST dataset")
    parser.add_argument('--n_qubits', type=int, default=2)
    parser.add_argument('--n_states', type=int, default=500)
    parser.add_argument('--shots_per_basis', type=int, default=100)
    parser.add_argument('--pure_fraction', type=float, default=0.5)
    parser.add_argument('--purity_range', type=float, nargs=2, default=None, help="Tuple (min, max) purity for mixed states")
    
    parser.add_argument('--noise', action='store_true', help="Enable coherent measurement noise")
    parser.add_argument('--xi', type=float, default=0.05, help="Noise ratio (ignored if --noise not set)")
    parser.add_argument('--noise_mode', type=str, choices=['state', 'basis'], default='state', help="Noise mode")
    parser.add_argument('--ghz', type=bool, default=False, help="Generate GHZ states if True, else generalised states")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--output', type=str, default='tomography_dataset.npz')
    args = parser.parse_args()

    data = generate_dataset(
        n_qubits=args.n_qubits,
        n_states=args.n_states,
        shots_per_basis=args.shots_per_basis,
        pure_fraction=args.pure_fraction,
        purity_range=args.purity_range,
        noise=args.noise,
        xi=args.xi,
        noise_mode=args.noise_mode,
        ghz=args.ghz,
        seed=args.seed
    )

    np.savez_compressed(
        args.output,
        counts=data['counts'],
        states=data['states'],
        labels=np.array(data['labels'], dtype=object),
        bases=np.array(data['bases'], dtype=object),
        n_qubits=data['n_qubits'],
        n_states=data['n_states'],
        shots_per_basis=data['shots_per_basis'],
        pure_fraction=data['pure_fraction'],
        purity_range=data['purity_range'],
        noise_enabled=data['noise_enabled'],
        xi=data['xi'],
        noise_mode=data['noise_mode'],
        ghz=data['ghz'],
        seed=data['seed']
    )
    print(f"Saved {args.output}")
    print(f"States: {data['n_states']} | Qubits: {data['n_qubits']}")
    print(f"Measurements per state: {len(data['labels'])}")
    print(f"seed: {data['seed']}")
    if data['purity_range'] is not None:
        print(f"Purity range for mixed states: {data['purity_range']}")
    else:
        print("No purity range specified, using pure_fraction =", data['pure_fraction'])

    if data['noise_enabled']:
        print(f"Noise, xi={data['xi']}, mode={data['noise_mode']}")
    else:
        print("no noise present")
    if data['ghz']:
        print("GHZ states generated")
    else:
        print("Generalised states generated")
