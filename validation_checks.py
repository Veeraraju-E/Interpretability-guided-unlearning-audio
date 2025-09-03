from __future__ import annotations
import numpy as np, json, os, time
from numpy.linalg import svd, norm
from scipy.stats import entropy
from sklearn.cluster import AgglomerativeClustering
from diagnostics import (check_whitening, decoder_metrics, activation_sparsity_metrics,
                         feature_discriminability, check_U, leace_invariants,
                         probe_lens_auc, cka_matrix, polysemanticity_index, energy_share_in_U
                         )
from plotting import (plot_probe_lens, plot_cka_heatmap, plot_polysemanticity,
                      plot_singular_value_spectrum, plot_decoder_overlap_matrix,
                      plot_label_direction_energy, plot_atom_contribution_scores)

from common import sae_encode_batch, invert_whitener

def check_projector_properties(P_r, tol=1e-6):
    idemp_err = norm(P_r @ P_r - P_r) / (norm(P_r) + 1e-12)
    sym_err = norm(P_r - P_r.T) / (norm(P_r) + 1e-12)
    eigs = np.linalg.eigvalsh(P_r) if P_r.size else np.array([])
    return {"idempotence_err": float(idemp_err), "symmetry_err": float(sym_err),
            "eigs_min": float(eigs.min()) if eigs.size else None,
            "eigs_max": float(eigs.max()) if eigs.size else None}

def atom_class_entropy(A_dev, labels_dev):
    labels_u = np.array(labels_dev)
    classes = np.unique(labels_u)
    m = A_dev.shape[1]
    entropies = np.zeros(m)
    for j in range(m):
        totals = np.array([np.abs(A_dev[labels_u==c, j]).sum() for c in classes])
        p = totals / (totals.sum() + 1e-12)
        entropies[j] = entropy(p, base=2)
    return entropies

def remove_atoms_from_Z(Z, D_cols):
    # Orthonormalize columns of D_cols via SVD and subtract projection
    if D_cols.shape[1] == 0:
        return Z.copy()
    U_d, s, Vt = svd(D_cols, full_matrices=False)
    Q = U_d[:, :D_cols.shape[1]]  # [d, r_sel]
    Z_proj = (Z @ Q) @ Q.T
    return Z - Z_proj

# def atom_ablation_curve(Z_all, dec_w, feat_idx, mu, W, test_trials, test_labels, top_k=100):
#     # Compute label direction in whitened ambient coordinates (ZU mean diff projected to ambient)
#     # Here we pick top atoms by projection of label direction onto atoms
#     # label_direction in ambient whitened frame: mean(forget dev) - mean(retain dev)
#     # For simplicity user must supply indices; this function returns (k, EER)
#     raise NotImplementedError("Use script in voxceleb.py to adapt to your variables")

def run_validation_checks(save_dir,
                          Z_all, Z_edit, E_all_norm, E_edit_norm,
                          ZU_mean, U, P_r, mu_r, dec_w, sae,
                          feat_idx, sets, labels,
                          mask_dev, mask_test,
                          y_dev_bin, leace_ridge,
                          mu, W,
                          device='cpu'):
    os.makedirs(save_dir, exist_ok=True)
    out = {}
    t0 = time.time()

    # A) Whitening
    try:
        Z_ret_dev = Z_all[np.array([s == 'retain:dev' for s in sets])]
        whiten_diag = check_whitening(Z_ret_dev)
        out['whiten_diag'] = whiten_diag
    except Exception as e:
        out['whiten_diag_error'] = str(e)

    # B) SAE/decoder diagnostics
    dec_d = decoder_metrics(dec_w)
    out['decoder_metrics'] = dec_d
    try:
        mask_dev_bool = np.array([s.endswith(':dev') for s in sets])
        A_dev = sae_encode_batch(sae, Z_all[mask_dev_bool], device=device, batch=8192)
        spars = activation_sparsity_metrics(A_dev, k_expected=getattr(sae,'k',None))
        out['sparsity'] = spars
    except Exception as e:
        out['sparsity_error'] = str(e)

    # C) Feature discriminability (sanity on selected feat_idx)
    try:
        A_ret = sae_encode_batch(sae, Z_all[np.array([s == 'retain:dev' for s in sets])], device=device, batch=8192)
        A_fgt = sae_encode_batch(sae, Z_all[np.array([s == 'forget:dev' for s in sets])], device=device, batch=8192)
        feat_disc = feature_discriminability(A_fgt, A_ret)
        out['feature_discriminability_summary'] = feat_disc.get('feature_auc_summary', {})
        # selected feature AUCs
        feat_aucs = np.array(feat_disc['feature_auc'])
        if len(feat_aucs) >= max(feat_idx)+1:
            out['selected_feat_aucs'] = feat_aucs[feat_idx].tolist()
    except Exception as e:
        out['feature_disc_error'] = str(e)

    # D) U checks
    try:
        D_selected = dec_w[:, feat_idx] if len(feat_idx) > 0 else dec_w[:,:0]
        diagU = check_U(U, D_selected if D_selected.shape[1] > 0 else None)
        out['U_diag'] = diagU
        Eshare = energy_share_in_U(Z_all, U)
        out['energy_share_in_U'] = float(Eshare)
    except Exception as e:
        out['U_diag_error'] = str(e)

    # E) LEACE invariants
    try:
        leace_diag = leace_invariants(Z_all, sets, U, np.array([1 if 'forget' in s else 0 for s in sets])[mask_dev],
                                      P_r, ZU_mean)
        out['leace_diag'] = leace_diag
        proj_props = check_projector_properties(P_r)
        out['P_r_props'] = proj_props
    except Exception as e:
        out['leace_diag_error'] = str(e)

    # F) Probe lens
    try:
        stages = {
            "E": invert_whitener(Z_all[mask_dev], mu, W),
            "Z": Z_all[mask_dev],
            "ZU": Z_all[mask_dev] @ U,
            "ZU_center": (Z_all[mask_dev] @ U) - (ZU_mean),
            "ZU_edit": ((Z_all[mask_dev] @ U) - ZU_mean) @ P_r.T + ZU_mean,
        }
        lens = probe_lens_auc(stages, y_dev_bin)
        out['probe_lens'] = lens
        plot_probe_lens(lens, save_dir)
    except Exception as e:
        out['probe_lens_error'] = str(e)

    # G) CKA
    try:
        Z_dev = Z_all[mask_dev]; Z_edit_dev = Z_edit[mask_dev]
        reps = {
            "Z(dev)": Z_dev,
            "Z_edit(dev)": Z_edit_dev,
            "ZU(dev)": Z_dev @ U,
            "ZU_edit(dev)": Z_edit_dev @ U,
        }
        cka = cka_matrix(reps)
        out['cka'] = {f"{a}|{b}": v for (a,b), v in cka.items()}
        plot_cka_heatmap(cka, save_dir)
    except Exception as e:
        out['cka_error'] = str(e)

    # H) Polysemanticity
    try:
        A_dev_full = sae_encode_batch(sae, Z_all[mask_dev], device=device, batch=8192)
        poly = polysemanticity_index(A_dev_full, [("forget" if (s.startswith('forget')) else "retain") for s in np.array(sets)[mask_dev]], top_q=0.1)
        out['polysemanticity'] = poly
        ents = atom_class_entropy(A_dev_full, [("forget" if (s.startswith('forget')) else "retain") for s in np.array(sets)[mask_dev]])
        out['atom_entropies_summary'] = {"mean": float(np.mean(ents)), "std": float(np.std(ents))}
        plot_polysemanticity(list(ents), save_dir)
    except Exception as e:
        out['polysemanticity_error'] = str(e)

    # I) Save JSON
    with open(os.path.join(save_dir, "validation_checks_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Validation checks finished in", time.time()-t0, "s. Output:", os.path.join(save_dir, "validation_checks_summary.json"))
    return out
