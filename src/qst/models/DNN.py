from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple
import numpy as np
import tensorflow as tf
from scipy.linalg import sqrtm

# Utilities

def rho_to_alpha(rho):
    """
    Convert a (d x d) density matrix rho into its "Cholesky parameters" alpha.
    """
    L = np.linalg.cholesky(rho + 1e-14 * np.eye(rho.shape[0]))
    d = rho.shape[0]
    alpha = []
    # diagonal (real, >0)
    for i in range(d):
        alpha.append(np.real(L[i, i]))
    for i in range(1, d):
        for j in range(i):
            alpha.append(np.real(L[i, j]))
            alpha.append(np.imag(L[i, j]))
    return np.array(alpha, dtype=np.float32)

def fidelity(rho1, rho2):
    A = sqrtm(rho1)
    return np.real(np.trace(sqrtm(A @ rho2 @ A)))**2


def tf_sqrtm_psd(A):
    """
    Compute principal sqrt of Hermitian PSD A (batch,d,d) via eigendecomposition.
    """
    
    eigvals, eigvecs = tf.linalg.eigh(A)

    eigvals = tf.math.real(eigvals)
    eigvals = tf.clip_by_value(eigvals, 0.0, tf.reduce_max(eigvals))
    sqrtvals = tf.sqrt(eigvals)
    D = tf.cast(tf.linalg.diag(sqrtvals), tf.complex64)
    
    return eigvecs @ D @ tf.linalg.adjoint(eigvecs)

def tf_alpha_to_rho(alpha, d):
    """
    Map real alpha (batch, N_alpha) -> complex density matrices (batch, d, d).
    Enforces positivity via a softplus on the Cholesky diag.
    """
    batch = tf.shape(alpha)[0]
    # split diag vs off-diag
    raw_diag = alpha[:, :d] 
    off_vals  = alpha[:, d:] 

    # start zero L
    L = tf.zeros((batch, d, d), tf.complex64)

    diag_pos = tf.nn.softplus(raw_diag) + 1e-6 
    diag_c   = tf.cast(diag_pos, tf.complex64)
    L = tf.linalg.set_diag(L, diag_c)

    idx = 0
    for i in range(1, d):
        for j in range(i):
            re = off_vals[:, idx]
            im = off_vals[:, idx+1]
            idx += 2
            cij = (tf.cast(re, tf.complex64)
                   + 1j * tf.cast(im, tf.complex64))
            cij = tf.reshape(cij, (batch, 1, 1))

            # mask with a one-hot at (i,j)
            flat = tf.one_hot(i*d + j, d*d, dtype=tf.complex64)
            mask = tf.reshape(flat, (d, d))[None, :, :]

            L = L + cij * mask

    rho = L @ tf.linalg.adjoint(L)  
    tr  = tf.linalg.trace(rho)
    return rho / tr[:, None, None]

def make_fidelity_loss(d):
    def fidelity_loss(alpha_true, alpha_pred):
        rho_t = tf_alpha_to_rho(alpha_true, d)
        rho_p = tf_alpha_to_rho(alpha_pred, d)

        # tiny regularizer to guard numeric issues
        I = tf.eye(d, dtype=tf.complex64)[None, :, :]
        rho_t = rho_t + 1e-8 * I
        rho_p = rho_p + 1e-8 * I

        sqrt_t = tf_sqrtm_psd(rho_t)
        inter  = sqrt_t @ (rho_p @ sqrt_t)
        s_mat  = tf_sqrtm_psd(inter)

        tr_s = tf.linalg.trace(s_mat)
        F = tf.abs(tr_s)**2
        return tf.reduce_mean(1.0 - F)
    return fidelity_loss

def make_hybrid_loss(d, lam=0.8):
    """
    Hybrid loss = lam * MSE + (1 - lam) * (1 - fidelity).
    """
    fid_loss_fn = make_fidelity_loss(d)

    def hybrid_loss(alpha_true, alpha_pred):
        # MSE on the Cholesky parameters
        mse = tf.reduce_mean(tf.square(alpha_true - alpha_pred))

        # fidelity loss already = mean(1 - F)
        phys = fid_loss_fn(alpha_true, alpha_pred)

        return lam * mse + (1.0 - lam) * phys

    return hybrid_loss

def alpha_to_rho_batch(d, alpha):
    """Convert batch of alpha vectors to density matrices using Cholesky."""
    N = alpha.shape[0]
    rho = np.zeros((N, d, d), dtype=np.complex64)
    for i in range(N):
        a = alpha[i]
        L = np.zeros((d, d), dtype=np.complex64)
        idx = 0
        for j in range(d):
            L[j, j] = a[idx]
            idx += 1
        for j in range(1, d):
            for k in range(j):
                re = a[idx]
                im = a[idx + 1]
                L[j, k] = re + 1j * im
                idx += 2
        rho_i = L @ L.conj().T
        rho[i] = rho_i / np.trace(rho_i)
    return rho



#  Build Neural network model 

def build_model(
    input_dim: int,
    d: int,
    hidden_sizes: Iterable[int] = (512, 512, 512),
    dropout: float = 0.2,
    lr: float = 1e-3,
    lam: float = 0.8) -> tf.keras.Model:
    """
    Construct and Compile the DNN for QST using TensorFlow/Keras, and the Hybrid Loss function
    """
    n_alpha = d * d

    layers = [
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout),
    ]
    for h in hidden_sizes:
        layers += [
            tf.keras.layers.Dense(h, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout),
        ]
    layers += [tf.keras.layers.Dense(n_alpha, activation=None)]

    model = tf.keras.Sequential(layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=make_hybrid_loss(d, lam=lam),
    )
    return model



# Wrapper

@dataclass
class DNNQST:
    d: int
    model: Optional[tf.keras.Model] = None
    config: Optional[Dict[str, Any]] = None

    def build(self, input_dim: int, **kwargs):
        cfg = dict(hidden_sizes=(512, 512, 512), dropout=0.2, lr=1e-3, lam=0.8)
        cfg.update(kwargs)
        self.config = cfg
        self.model = build_model(input_dim=input_dim, d=self.d, **cfg)

    def fit(
        self,
        X_train: np.ndarray,
        Y_rho_train: np.ndarray,
        epochs: int = 20,
        batch_size: int = 64,
        validation_split: float = 0.1,
        callbacks: Optional[list] = None,
        verbose: int = 1,
    ) -> tf.keras.callbacks.History:
        """
        Train on (X, rho) pairs
        """
        if self.model is None:
            raise RuntimeError("Call build(input_dim=...) before fit().")
        # Convert labels
        alphas_train = np.stack([rho_to_alpha(r) for r in Y_rho_train], axis=0).astype(np.float32)
        history = self.model.fit(
            X_train, alphas_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
        )
        return history

    #  Inference

    def predict_alpha(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not built.")
        alpha_pred = self.model.predict(X, batch_size=batch_size, verbose=0)
        return alpha_pred.astype(np.float32)

    def predict_rho(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        alpha_pred = self.predict_alpha(X, batch_size=batch_size)
        rho_pred = alpha_to_rho_batch(self.d, alpha_pred)
        return rho_pred

    #  Evaluation

    def evaluate_fidelity(
        self, X: np.ndarray, Y_rho_true: np.ndarray, batch_size: int = 256
    ) -> Tuple[float, float, np.ndarray]:
        """
        Return (mean_fidelity, std_fidelity, per_example_fidelities)
        """
        rho_pred = self.predict_rho(X, batch_size=batch_size)
        N = Y_rho_true.shape[0]
        fidelities = np.empty(N, dtype=np.float64)
        for i in range(N):
            fidelities[i] = fidelity(rho_pred[i], Y_rho_true[i])
        return float(fidelities.mean()), float(fidelities.std(ddof=0)), fidelities

    # Svaing the Model

    def save(self, path: str):
        if self.model is None:
            raise RuntimeError("Nothing to save; build the model first.")
        self.model.save(path)

    def load(self, path: str):
        self.model = tf.keras.models.load_model(
            path,
            compile=True)
        
        