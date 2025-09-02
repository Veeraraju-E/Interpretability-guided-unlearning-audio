1) Replace compute_whitening + apply_whitening and add inverse_whiten

Replace your compute_whitening / apply_whitening with the following (and add inverse_whiten).

# Replace existing compute_whitening and apply_whitening with these:
def compute_whitening(embeddings: np.ndarray, n_components: int = None, eps: float = 1e-12):
    """
    Compute *symmetric* whitener W = V diag(lambda^{-1/2}) V^T
    and its inverse W_inv = V diag(lambda^{+1/2}) V^T.
    Returns (mu, W, W_inv). No row-normalization.
    """
    X = np.asarray(embeddings, dtype=np.float64)
    N, d = X.shape
    mu = np.mean(X, axis=0)
    Xc = X - mu

    # empirical covariance
    cov = (Xc.T @ Xc) / max(1, (N - 1))

    # eigen-decomposition (symmetric)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
    # sort descending and optionally truncate
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    if n_components is None:
        n_components = d
    n_components = min(n_components, d)
    eigvals = eigvals[:n_components]
    eigvecs = eigvecs[:, :n_components]

    # clamp
    eigvals = np.maximum(eigvals, eps)
    # symmetric whitener and inverse-whitener in the original d-dimensional ambient space
    # build full-rank symmetric W (d x d) using truncated eigenspace
    V = eigvecs
    Lambda_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
    Lambda_sqrt = np.diag(np.sqrt(eigvals))

    # W maps (X - mu) -> whitened coordinates Z (d x d). Use the reconstructed symmetric operators:
    W = V @ Lambda_inv_sqrt @ V.T       # shape (d, d)
    W_inv = V @ Lambda_sqrt @ V.T       # shape (d, d)

    return mu.astype(np.float32), W.astype(np.float32), W_inv.astype(np.float32)

def apply_whitening(embeddings: np.ndarray, mu: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Z = (E - mu) @ W
    NOTE: do NOT row-normalize here. Keep covariance structure intact.
    """
    X = np.asarray(embeddings, dtype=np.float32)
    Xc = X - mu
    Z = np.dot(Xc, W)
    return Z.astype(np.float32)

def inverse_whiten(Z: np.ndarray, mu: np.ndarray, W_inv: np.ndarray) -> np.ndarray:
    """
    E' = Z @ W_inv + mu
    """
    Z = np.asarray(Z, dtype=np.float32)
    E = np.dot(Z, W_inv) + mu
    return E.astype(np.float32)


Why: This implements the symmetric whitener from the PDF and returns its inverse so edited whitened embeddings can be mapped back correctly. Do not l2-normalize rows here (that breaks Σ computations). 

2) Remove row-normalization from SAE encoding / decoding pipeline

Where: remove l2_norm_rows calls applied to Z_all, Z_edit, E_edit inside the main flow. Concretely, replace:

Z_all = l2_norm_rows(Z_all.astype(np.float32)) → remove normalization and just keep Z_all = Z_all.astype(np.float32)

Z_edit = l2_norm_rows(Z_edit) → remove

E_edit_norm = l2_norm_rows(E_edit) → remove; we will normalize for scoring later (compute_scores already L2-normalizes before dot product).

Example replacement (in main after collecting activations):

# after encoding:
Z_all = np.vstack(activations).astype(np.float32)    # DO NOT L2 normalize here


Why: LEACE math depends on empirical covariances of the whitened ambient coordinates; L2-normalising rows ruins those covariances.

3) Add SAE feature-mining → build forget-subspace U (decoder columns)

New helper (add after SAE class / train helper):

from sklearn.metrics import roc_auc_score

def mine_forget_subspace_from_sae(sae: TopKSAE, Z_activations: np.ndarray, labels: List[str], mask: np.ndarray,
                                  forget_labels: List[str], top_k: int = 64):
    """
    1) compute per-latent discriminative score (AUC) between forget vs retain on the masked subset
    2) pick top_k latent indices F
    3) collect corresponding decoder atoms (columns of sae.dec.weight) -> DF (d x |F|)
    4) orthonormalize DF (SVD) to return U (d x r) with orthonormal columns (r <= top_k)
    Returns: U (d x r), selected_indices list, scores
    """
    labels_arr = np.array([1 if (lab in forget_labels) else 0 for lab in labels], dtype=int)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        raise ValueError("Empty mask for feature mining.")
    # evaluate AUC per latent unit
    m = Z_activations.shape[1]
    scores = np.zeros(m, dtype=float)
    for j in range(m):
        try:
            scores[j] = float(roc_auc_score(labels_arr[idx], Z_activations[idx, j]))
        except Exception:
            scores[j] = 0.5
    # pick top_k
    k = min(top_k, m)
    selected = np.argsort(scores)[::-1][:k]

    # decoder weight: shape (d, m) in your TopKSAE (nn.Linear(m, d, bias=False) stores weight (d,m))
    D = sae.dec.weight.detach().cpu().numpy()  # (d, m)
    DF = D[:, selected]                         # (d, k)
    # SVD / orthonormalize to get U (d x r)
    U_svd, svals, Vt = np.linalg.svd(DF, full_matrices=False)
    r = np.sum(svals > 1e-12)
    if r == 0:
        raise ValueError("Selected decoder atoms are numerically rank-deficient.")
    U = U_svd[:, :r].astype(np.float32)        # orthonormal columns (d x r)

    return U, selected.tolist(), scores.tolist()


Why: This implements the pipeline's "Feature mining and subspace construction" step: choose discriminative latent units (based on activations), map to decoder atoms, orthonormalize to produce U. The PDF insists U is built from decoder columns. 
 

4) Implement LEACE operator in the reduced frame and apply the affine-centered edit

Add these two helpers (compute + apply):

def compute_leace_reduced(Z_whitened: np.ndarray, U: np.ndarray, labels: List[str],
                          forget_labels: List[str], mask: np.ndarray, gamma: float = 1e-3):
    """
    Computes Pr (r x r) as in the PDF (reduced frame).
    Also returns Zbar_U (1 x r) computed on masked subset (train/dev used to form Σ).
    """
    # reduced coords
    ZU = np.dot(Z_whitened, U)          # (N x r)
    labels_arr = np.array([1 if (lab in forget_labels) else 0 for lab in labels], dtype=int)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        raise ValueError("Empty mask for LEACE projector computation.")
    ZU_tr = ZU[idx, :]                  # (n_tr x r)
    ytr = labels_arr[idx].reshape(-1, 1)  # (n_tr x 1)
    # centering
    ZbarU = np.mean(ZU_tr, axis=0, keepdims=True)   # (1 x r)
    Ybar = np.mean(ytr, axis=0, keepdims=True)      # (1 x 1)
    ZU_tilde = ZU_tr - ZbarU                        # (n_tr x r)
    Y_tilde = ytr - Ybar                            # (n_tr x 1)

    n_tr = max(1, ZU_tr.shape[0])
    Sigma = (ZU_tilde.T @ ZU_tilde) / max(1, (n_tr - 1))    # (r x r)
    SigmaZY = (ZU_tilde.T @ Y_tilde) / max(1, (n_tr - 1))   # (r x 1)

    # regularize
    Sigma_gamma = Sigma + gamma * np.eye(Sigma.shape[0])
    # symmetric sqrt and inv-sqrt via eigh
    eigvals, eigvecs = np.linalg.eigh(Sigma_gamma)
    eigvals = np.maximum(eigvals, 1e-12)
    sqrt_S = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    invsqrt_S = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    M = invsqrt_S @ SigmaZY          # (r x 1) -> predictive direction(s)
    # projector onto column space of M: SVD -> take left singular vectors
    U_m, s_m, Vt_m = np.linalg.svd(M, full_matrices=False)
    rank = np.sum(s_m > 1e-12)
    if rank == 0:
        PC = np.zeros((M.shape[0], M.shape[0]))
    else:
        Uc = U_m[:, :rank]   # (r x rank)
        PC = Uc @ Uc.T       # (r x r)

    # Pr = I_r - sqrt_S @ PC @ invsqrt_S   (same as I - W+ PC Wr)
    Pr = np.eye(Sigma.shape[0]) - (sqrt_S @ PC @ invsqrt_S)

    return Pr.astype(np.float32), ZbarU.squeeze(0).astype(np.float32)

def apply_leace_edit_and_mapback(Z_whitened: np.ndarray, U: np.ndarray, Pr: np.ndarray, ZbarU: np.ndarray,
                                 W_inv: np.ndarray, mu: np.ndarray):
    """
    Apply affine-centered reduced-frame LEACE edit (matrix form), map back to original embedding frame.
    Returns E_edit (N x d) in original embedding coordinates.
    Mathematics matches Section 6 of the PDF.
    """
    # ZU
    ZU = np.dot(Z_whitened, U)               # (N x r)
    # center, apply Pr^T per PDF (they used P_r^T in formula)
    ZU_centered = ZU - ZbarU[None, :]
    ZU_edited = np.dot(ZU_centered, Pr.T) + ZbarU[None, :]   # (N x r)
    # replace projected part in ambient whitened frame
    Z_prime = Z_whitened - np.dot(ZU, U.T) + np.dot(ZU_edited, U.T)   # (N x d)
    # inverse whiten to original embedding space
    E_edit = inverse_whiten(Z_prime, mu, W_inv)
    return E_edit.astype(np.float32)


Why: This is a nearly line-by-line implementation of Section 5–6 (LEACE in reduced frame + mapping back) from your PDF. It computes Σ, ΣZY, Σγ, invsqrt and sqrt, forms M, gets projector onto col(M), forms Pr, then performs the affine-centered edit and inverse-whitens back to original embedding frame. 

5) Fix the classifier train/dev mask bug

Replace this block in main():

# old (incorrect)
cls_train_mask = (M_retain_dev | M_forget_dev)
cls_val_mask = (M_retain_test | M_forget_test)


with this (correct):

# correct masks: train -> dev (validation) -> test (final eval)
cls_train_mask = (M_retain_train | M_forget_train)
cls_val_mask = (M_retain_dev | M_forget_dev)
cls_test_mask = (M_retain_test | M_forget_test)


And use cls_val_mask for validation during training, and cls_test_mask for final classifier evaluation.

Why: training-on-dev was the likely cause of empty training sets and incorrect validation logic. 

6) Fix build_verification_trials return bug

Replace the problematic return line inside that function with a deterministic consistent return:

def build_verification_trials(labels: List[str], target_speaker: str, mask: np.ndarray) -> Tuple[List[Tuple[int,int]], np.ndarray]:
    labels = np.array(labels)
    idx_in_mask = np.where(mask)[0].tolist()
    pos_idx = [i for i in idx_in_mask if labels[i] == target_speaker]
    ok_idx = [i for i in idx_in_mask if labels[i] != target_speaker]
    trials = []
    trial_labels = []

    for (i,j) in combinations(pos_idx, 2):
        trials.append((i,j))
        trial_labels.append(1)

    n_pos = max(1, len(trials))
    if len(ok_idx) == 0 or len(pos_idx) == 0:
        # always return consistent types
        return trials, (np.array(trial_labels, dtype=int) if len(trial_labels)>0 else np.array([], dtype=int))

    rng = np.random.RandomState(1337)
    neg_pairs = []
    for _ in range(n_pos):
        a = rng.choice(pos_idx)
        b = rng.choice(ok_idx)
        neg_pairs.append((a,b))

    trials.extend(neg_pairs)
    trial_labels.extend([0]*len(neg_pairs))
    return trials, np.array(trial_labels, dtype=int)


Why: guarantee the function always returns (list_of_pairs, np.ndarray_of_labels) — callers expect that exact contract. 

7) Changes to the main pipeline (pseudo-diff)

Change the overall sequence in main() (high-level):

compute mu, W, W_inv = compute_whitening(E_raw[sets=='train'], ...)

E_whitened = apply_whitening(E_raw, mu, W) (no row-norm)

train SAE on E_whitened[train_mask] (same mask you used) — unchanged

encode to get Z_activations = sae.encode(E_whitened) (do NOT normalize activations)

mine U, selected, scores = mine_forget_subspace_from_sae(sae, Z_activations, labels, project_mask, [forget_speaker], top_k=...)

compute Pr, ZbarU = compute_leace_reduced(E_whitened, U, labels, [forget_speaker], mask=project_mask, gamma=gamma)

apply E_edit = apply_leace_edit_and_mapback(E_whitened, U, Pr, ZbarU, W_inv, mu)

For scoring and classifier evaluation: use the same normalization before scoring. compute_scores() already L2-normalizes inside, so you can pass E_raw and E_edit (or normalized variants) consistently.

Important: Remove these previous calls which you had placed earlier:

Z_all = l2_norm_rows(Z_all) (remove)

Z_edit = l2_norm_rows(Z_edit) (remove)

E_edit_norm = l2_norm_rows(E_edit) (remove)

do not use compute_leace_projector (the old logistic-reg based function) for the main edit — either remove it or keep it as a debugging baseline.

8) Save W_inv, U and selected_indices to artifacts

When saving NPZ at the end, add:

np.savez(out_npz,
         mu=mu, W=W, W_inv=W_inv,
         embeddings_before=E_raw, embeddings_after=E_edit,
         labels=np.array(labels, dtype=object), segment_ids=np.array(ids, dtype=object),
         sets=np.array(sets, dtype=object), results=results,
         per_class_test=json.dumps(per_class_test),
         U=U, selected_decoder_indices=np.array(selected_indices, dtype=int))


Why: Saves everything for reproducibility and diagnosis (you will want U and W_inv to re-run tests later). 

Quick tests / sanity checks (run these immediately after changes)

Whiten sanity

On the training set used to compute whitening: np.cov(Z_train, rowvar=False) should be approximately identity (check max deviation < 1e-6 for well-conditioned data).

Example check:

Ztr = E_whitened[sets=='train']
C = np.cov(Ztr, rowvar=False)
print("max abs(C - I) = ", np.max(np.abs(C - np.eye(C.shape[0]))))


LEACE identity test (per PDF Section 5.3 identity)

After computing Pr and performing the reduced-frame edit, compute the empirical cross-covariance in reduced coordinates:

ZU_prime = (E_whitened @ U)  # before/after as needed
# compute centered reduced coords for edited set (use same subset used to compute Sigma)
# then check (1/(N-1)) ZU_prime.T @ Y_tilde  is approximately zero for the training subset


If Pr is correct, (1/(N-1)) Z̃'_U^T Ỹ should be (near) zero.

Classifier masks

Confirm sum(cls_train_mask) > 0 and sum(cls_val_mask) > 0 before training; otherwise raise a helpful error.

Verification trial types

After build_verification_trials, assert isinstance(trials, list) and isinstance(labels_arr, np.ndarray) and len(trials) == len(labels_arr).

Save and reload check

Immediately reload the saved .npz and verify W_inv @ W ≈ I on the subspace and U has orthonormal columns U.T @ U ≈ I_r.