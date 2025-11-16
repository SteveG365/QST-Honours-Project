# src/models/qst_transfer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple, List

import numpy as np
import tensorflow as tf
from scipy.linalg import sqrtm

def n_alpha_from_dim(d: int) -> int:
    """Number of real parameters in complex Cholesky (equals d**2)."""
    return d + 2 * (d * (d - 1) // 2)

def rho_to_alpha(rho: np.ndarray) -> np.ndarray:
    """
    Convert a (d x d) density matrix rho into its 'Cholesky parameters' alpha.
    """
    d = rho.shape[0]
    # tiny jitter for numerical PSD
    L = np.linalg.cholesky(rho + 1e-14 * np.eye(d))
    alpha = []
    # diagonal (real, >0)
    for i in range(d):
        alpha.append(np.real(L[i, i]))
    # strictly lower triangle (real + imag)
    for i in range(1, d):
        for j in range(i):
            alpha.append(np.real(L[i, j]))
            alpha.append(np.imag(L[i, j]))
    return np.array(alpha, dtype=np.float32)

def fidelity(rho1: np.ndarray, rho2: np.ndarray) -> float:
    A = sqrtm(rho1)
    return float(np.real(np.trace(sqrtm(A @ rho2 @ A)))**2)

def alpha_to_rho_batch(d: int, alpha: np.ndarray) -> np.ndarray:
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
        tr = np.real(np.trace(rho_i))
        rho[i] = rho_i / (tr if tr > 0 else 1.0)
    return rho

def tf_sqrtm_psd(A: tf.Tensor) -> tf.Tensor:
    """
    Principal sqrt of Hermitian PSD A (..., d, d) via eigendecomposition.
    """
    eigvals, eigvecs = tf.linalg.eigh(A)
    eigvals = tf.math.real(eigvals)
    maxv = tf.reduce_max(eigvals, axis=-1, keepdims=True)
    eigvals = tf.clip_by_value(eigvals, 0.0, maxv)
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


def make_fidelity_loss(d: int) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    def fidelity_loss(alpha_true, alpha_pred):
        rho_t = tf_alpha_to_rho(alpha_true, d)
        rho_p = tf_alpha_to_rho(alpha_pred, d)
        I = tf.eye(d, dtype=tf.complex64)[None, :, :]
        rho_t = rho_t + 1e-8 * I
        rho_p = rho_p + 1e-8 * I

        sqrt_t = tf_sqrtm_psd(rho_t)
        inter  = sqrt_t @ (rho_p @ sqrt_t)
        s_mat  = tf_sqrtm_psd(inter)

        tr_s = tf.linalg.trace(s_mat)
        F = tf.abs(tr_s) ** 2
        return 1.0 - tf.reduce_mean(F)
    return fidelity_loss

def make_hybrid_loss(d: int, lam: float = 0.8) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    fid_loss_fn = make_fidelity_loss(d)
    def hybrid_loss(alpha_true, alpha_pred):
        mse  = tf.reduce_mean(tf.square(alpha_true - alpha_pred))
        phys = fid_loss_fn(alpha_true, alpha_pred)
        return lam * mse + (1.0 - lam) * phys
    return hybrid_loss

def make_fidelity_metric(d: int) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    def fidelity_metric(alpha_true, alpha_pred):
        rho_t = tf_alpha_to_rho(alpha_true, d)
        rho_p = tf_alpha_to_rho(alpha_pred, d)
        I = tf.eye(d, dtype=tf.complex64)[None, :, :]
        rho_t = rho_t + 1e-8 * I
        rho_p = rho_p + 1e-8 * I
        sqrt_t = tf_sqrtm_psd(rho_t)
        inter  = sqrt_t @ (rho_p @ sqrt_t)
        s_mat  = tf_sqrtm_psd(inter)
        tr_s   = tf.linalg.trace(s_mat)
        F      = tf.abs(tr_s) ** 2
        return tf.reduce_mean(F)
    fidelity_metric.__name__ = "fidelity_metric"
    return fidelity_metric





@dataclass
class QSTConfig:
    """Configuration for the QST MLP model and training."""
    hidden_sizes: Tuple[int, ...] = (512, 512, 512)
    dropout: float = 0.2
    use_batchnorm: bool = True
    # compile/training
    lr_backbone: float = 1e-3       # LR when training backbone
    lr_head: float = 1e-3           # LR when training head-only phase
    batch_size: int = 64
    epochs: int = 30
    val_split: float = 0.1
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0



class QSTTransferModel:
    """
    Keras MLP for QST that predicts alpha-parameter vectors from features X.
    Supports:
      - backbone (feature extractor) + task head
      - freezing/unfreezing backbone layers
      - head replacement for different d (n_alpha)
      - two-phase transfer
    """
    def __init__(
        self,
        input_dim: int,
        n_alpha: int,
        d: int,
        lam: float,
        config: Optional[QSTConfig] = None,
        name: str = "qst_transfer_mlp",
    ):
        self.input_dim = input_dim
        self.n_alpha = n_alpha
        self.d = d
        self.lam = lam
        self.cfg = config if config is not None else QSTConfig()
        self.name = name
        self._build_model()


    def _build_backbone(self, x: tf.Tensor) -> tf.Tensor:
        h = x
        for i, width in enumerate(self.cfg.hidden_sizes, start=1):
            h = tf.keras.layers.Dense(width, activation="relu", name=f"feat_dense_{i}")(h)
            if self.cfg.use_batchnorm:
                h = tf.keras.layers.BatchNormalization(name=f"feat_bn_{i}")(h)
            if self.cfg.dropout and self.cfg.dropout > 0:
                h = tf.keras.layers.Dropout(self.cfg.dropout, name=f"feat_do_{i}")(h)
        return h

    def _build_head(self, features: tf.Tensor, n_alpha: Optional[int] = None) -> tf.Tensor:
        out_dim = n_alpha if n_alpha is not None else self.n_alpha
        y = tf.keras.layers.Dense(out_dim, name="head_dense")(features)
        return y

    def _build_model(self):
        norm = tf.keras.layers.Normalization(name="input_norm")
        inputs = tf.keras.layers.Input(shape=(self.input_dim,), name="input")
        x = norm(inputs)
        features = self._build_backbone(x)
        outputs  = self._build_head(features, self.n_alpha)
        self.model: tf.keras.Model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        self.compile()

    def compile(self, lr_backbone: Optional[float] = None, lr_head: Optional[float] = None):
        if lr_backbone is None:
            lr_backbone = self.cfg.lr_backbone
        if lr_head is None:
            lr_head = self.cfg.lr_head
        loss_fn = make_hybrid_loss(self.d, lam=self.lam)
        metric = make_fidelity_metric(self.d)
        opt = tf.keras.optimizers.Adam(learning_rate=lr_backbone)
        self.model.compile(optimizer=opt, loss=loss_fn, metrics=[metric], run_eagerly=False)

    def adapt_normalizer(self, X: np.ndarray):
        """
        Adapt input normalization on source/general features, then freeze it.
        """
        layer = self.model.get_layer("input_norm")
        layer.adapt(X.astype(np.float32))
        layer.trainable = False


    def backbone_layers(self) -> List[tf.keras.layers.Layer]:
        return [L for L in self.model.layers if L.name.startswith("feat_")]

    def head_layers(self) -> List[tf.keras.layers.Layer]:
        return [L for L in self.model.layers if L.name.startswith("head_")]

    def freeze_backbone(self):
        for L in self.backbone_layers():
            L.trainable = False

    def unfreeze_backbone(self, up_to: Optional[int] = None):
        """
        If up_to is None: unfreeze all backbone layers.
        Else: unfreeze blocks whose trailing index <= up_to (feat_*_1, feat_*_2, ...).
        """
        def parse_block_idx(name: str) -> int:
            parts = name.split("_")
            for p in reversed(parts):
                if p.isdigit():
                    return int(p)
            return 10**9
        for L in self.backbone_layers():
            idx = parse_block_idx(L.name)
            L.trainable = (up_to is None) or (idx <= up_to)

    def replace_head(self, new_n_alpha: int):
        """
        Replace the task head (for different output size); keeps backbone weights.
        """
        features_layer = self.model.get_layer(name=f"feat_dense_{len(self.cfg.hidden_sizes)}").output
        new_head = self._build_head(features_layer, n_alpha=new_n_alpha)
        self.n_alpha = new_n_alpha
        self.model = tf.keras.Model(inputs=self.model.input, outputs=new_head, name=self.name)
        self.compile()

    def fit(
        self,
        X_train: np.ndarray,
        y_alpha_train: np.ndarray,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        val_split: Optional[float] = None,
        callbacks: Optional[list] = None,
        verbose: int = 1,
    ):
        if epochs is None:
            epochs = self.cfg.epochs
        if batch_size is None:
            batch_size = self.cfg.batch_size
        if val_split is None:
            val_split = self.cfg.val_split
        if callbacks is None:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.cfg.early_stopping_patience,
                    min_delta=self.cfg.early_stopping_min_delta,
                    restore_best_weights=True,
                )
            ]
        return self.model.fit(
            X_train,
            y_alpha_train,
            validation_split=val_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
        )

    def predict_alpha(self, X: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        if batch_size is None:
            batch_size = self.cfg.batch_size
        return self.model.predict(X, batch_size=batch_size)

    def predict_rho(self, alpha_pred: np.ndarray) -> np.ndarray:
        return alpha_to_rho_batch(self.d, alpha_pred)

    def save_weights(self, path: str):
        self.model.save_weights(path)

    def load_weights(self, path: str):
        import inspect
        sig = inspect.signature(self.model.load_weights)
        params = sig.parameters
        if "by_name" in params:
            return self.model.load_weights(path, by_name=False, skip_mismatch=True)
        else:
            return self.model.load_weights(path)

    def rhos_to_alphas(self, Y_rho: Iterable[np.ndarray]) -> np.ndarray:
        return np.stack([rho_to_alpha(rho) for rho in Y_rho], axis=0)

    def leave_one_out_train(
        self,
        X: np.ndarray,
        Y_rho: np.ndarray,
        idx: int,
        verbose: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Trains on all but a single held-out index. Returns (rho_pred, alpha_pred) for the held-out example.
        """
        X_test = X[idx]
        y_test = Y_rho[idx]
        X_train = np.delete(X, idx, axis=0)
        Y_train = np.delete(Y_rho, idx, axis=0)
        alphas_train = self.rhos_to_alphas(Y_train)
        self.fit(X_train, alphas_train, verbose=verbose)
        alpha_pred = self.predict_alpha(X_test[None, :])[0]
        rho_pred = self.predict_rho(alpha_pred[None, :])[0]
        return rho_pred, alpha_pred


    def pretrain_general(self, X_gen: np.ndarray, Y_gen_rho: np.ndarray, verbose: int = 1):
        """
        Pretrain on large, general dataset. Call adapt_normalizer(X_gen) before this.
        """
        y_gen_alpha = self.rhos_to_alphas(Y_gen_rho)
        self.unfreeze_backbone()
        self.compile(lr_backbone=self.cfg.lr_backbone)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.cfg.early_stopping_patience,
                min_delta=self.cfg.early_stopping_min_delta,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=verbose),
        ]
        return self.model.fit(
            X_gen, y_gen_alpha,
            validation_split=self.cfg.val_split,
            epochs=self.cfg.epochs,
            batch_size=self.cfg.batch_size,
            callbacks=callbacks,
            verbose=verbose,
        )

    def transfer_to_target(
        self,
        X_tgt: np.ndarray,
        Y_tgt_rho: np.ndarray,
        last_k_to_unfreeze: int = 1,
        lambda_l2sp: float = 0.0,
        head_only_epochs: int = 20,
        finetune_epochs: int = 30,
        verbose: int = 1,
    ):
        """
        Transfer in two phases:
          Freeze backbone, train head only
          Unfreeze last k backbone blocks (if k > 0)
        """
        y_tgt_alpha = self.rhos_to_alphas(Y_tgt_rho)

        self.freeze_backbone()
        self.compile(lr_backbone=self.cfg.lr_head)
        cb_A = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=max(5, self.cfg.early_stopping_patience // 2),
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=verbose),
        ]
        self.model.fit(
            X_tgt, y_tgt_alpha,
            validation_split=min(0.25, max(0.1, self.cfg.val_split)),
            epochs=head_only_epochs,
            batch_size=min(32, self.cfg.batch_size),
            callbacks=cb_A,
            verbose=verbose,
        )

        if last_k_to_unfreeze > 0:
            block_ids = []
            for L in self.backbone_layers():
                try:
                    block_ids.append(int(L.name.split("_")[-1]))
                except Exception:
                    pass
            max_block = max(block_ids) if block_ids else 0

            # freeze all then unfreeze last k
            self.freeze_backbone()
            for L in self.backbone_layers():
                try:
                    idx = int(L.name.split("_")[-1])
                except Exception:
                    idx = -1
                if idx > max_block - last_k_to_unfreeze:
                    L.trainable = True



            # smaller LR for fine-tuning
            self.compile(lr_backbone=min(5e-4, 0.5 * self.cfg.lr_backbone))
            cb_B = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.cfg.early_stopping_patience,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=verbose),
            ]
            self.model.fit(
                X_tgt, y_tgt_alpha,
                validation_split=min(0.3, max(0.1, self.cfg.val_split)),
                epochs=finetune_epochs,
                batch_size=min(32, self.cfg.batch_size),
                callbacks=cb_B,
                verbose=verbose,
            )


