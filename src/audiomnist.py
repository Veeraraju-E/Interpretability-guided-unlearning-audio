from __future__ import annotations
import os
import time
import json
import random
import csv
import traceback
from pathlib import Path
from typing import List, Dict, Optional
import argparse
import numpy as np
import torch

from common import (
    seed_all, human_time, extract_embeddings, fit_whitener, apply_whitener,
    invert_whitener, l2_normalize, SAE, train_sae, sae_encode_batch,
    prune_and_normalize_decoder, feature_stats, select_features_from_stats, build_U_from_decoder,
    leace_projector_in_U, apply_subspace_leace, apply_sauce_gating, inject_noise_in_subspace,
    compute_scores, eer_from_scores, build_verification_trials,
    train_classifier, evaluate_classifier, per_class_eers, avg_eers, safe_linear_probe,
    # monosemanticity functions left available for single-run usage
    calculate_monosemanticity_score, compute_monosemanticity_statistics
)


def build_manifest(data_root: str, embeddings_path: str = None, dev_frac_per_digit: float = 0.75, seed: int = 1337) -> List[Dict]:
    """
    Build manifest entries for AudioMNIST dataset.
    Filenames look like: '0_george_0.wav' => digit=0, speaker=george
    Returns list of dicts with: {"path", "speaker", "digit", "split", "segment_id"}
    """
    if embeddings_path is not None:
        data = np.load(embeddings_path, allow_pickle=True)
        labels = data['labels'].tolist() if 'labels' in data else (data['y'].tolist() if 'y' in data else None)
        ids = data['segment_ids'].tolist() if 'segment_ids' in data else (data['ids'].tolist() if 'ids' in data else None)
        if labels is None:
            raise RuntimeError("embeddings .npz must contain 'labels' or 'y' array when embeddings_path is used")
        raw_entries = [{'path': f'dummy_{i}', 'digit': labels[i], 'segment_id': ids[i] if ids is not None else None} for i in range(len(labels))]
    else:
        wavs = sorted(Path(data_root).glob("*.wav"))
        if not wavs:
            raise RuntimeError(f"No wav files found in {data_root}")

        raw_entries = []
        for p in wavs:
            fname = p.stem
            parts = fname.split("_")
            if len(parts) < 2:
                continue
            digit, spk = parts[0], parts[1]
            raw_entries.append({
                "path": str(p),
                "digit": digit,
                "segment_id": str(p)  # full path used for embedding alignment
            })

        if not raw_entries:
            raise RuntimeError("No valid AudioMNIST entries parsed")

    # Split by digit
    by_digit = {}
    for i, e in enumerate(raw_entries):
        digit = e['digit']
        by_digit.setdefault(digit, []).append(i)

    rng = random.Random(seed)
    manifest = []
    for digit, indices in by_digit.items():
        rng.shuffle(indices)
        n_dev = max(1, int(len(indices) * dev_frac_per_digit))
        if len(indices) > 1 and n_dev >= len(indices):
            n_dev = len(indices) - 1
        dev_indices = indices[:n_dev]
        test_indices = indices[n_dev:]
        for idx in dev_indices:
            entry = raw_entries[idx].copy()
            entry['split'] = 'dev'
            manifest.append(entry)
        for idx in test_indices:
            entry = raw_entries[idx].copy()
            entry['split'] = 'test'
            manifest.append(entry)
    if len(manifest) == 0:
        raise RuntimeError("No entries found in manifest")
    return manifest


# ----------------------------
# Precompute / shared steps
# ----------------------------
def precompute_embeddings_and_whitener(data_root: str, embeddings_path: Optional[str], dev_frac_per_digit: float,
                                       manifest_seed: int, encoder: str, device: str,
                                       wav2vec2_model: str, hubert_model: str, conformer_model: str):
    """
    Build manifest (dev/test splits), extract/load embeddings aligned to manifest order,
    fit a whitener on *dev* samples (using all dev samples), return a dict of precomputed objects.
    This is run exactly once for --run_all to avoid repeating extractor work.
    """
    print("[PRECOMP] Building manifest and splits (once)...")
    entries = build_manifest(data_root, embeddings_path, dev_frac_per_digit, manifest_seed)
    by = {}
    for e in entries:
        by.setdefault(e["digit"], []).append(e)
    digits = sorted(by.keys())

    dev_manifest = [m for m in entries if m['split'] == 'dev']
    test_manifest = [m for m in entries if m['split'] == 'test']
    manifest_ordered = list(dev_manifest) + list(test_manifest)

    # load or extract embeddings aligned to manifest_ordered
    if embeddings_path:
        print("[PRECOMP] Loading precomputed embeddings and aligning to manifest order...")
        data = np.load(embeddings_path, allow_pickle=True)
        embs = data['embeddings']
        seg_ids = None
        if 'segment_ids' in data:
            seg_ids = list(map(str, data['segment_ids']))
        elif 'ids' in data:
            seg_ids = list(map(str, data['ids']))

        if seg_ids is not None:
            id_to_idx = {sid: i for i, sid in enumerate(seg_ids)}
            E_all = np.zeros((len(manifest_ordered), embs.shape[1]), dtype=np.float32)
            for i, entry in enumerate(manifest_ordered):
                sid = entry.get('segment_id', None)
                if sid is None:
                    raise RuntimeError("Embeddings provided but manifest entries lack 'segment_id' to align")
                sid_s = str(sid)
                if sid_s not in id_to_idx and os.path.basename(sid_s) in id_to_idx:
                    sid_s = os.path.basename(sid_s)
                if sid_s not in id_to_idx:
                    raise RuntimeError(f"segment_id {sid} not found in embeddings file keys")
                E_all[i] = embs[id_to_idx[sid_s]]
            L_all = [m['digit'] for m in manifest_ordered]
            P_all = [m['path'] for m in manifest_ordered]
        else:
            print("  Warning: embeddings file lacks 'segment_ids' - assuming ordering already matches manifest")
            if embs.shape[0] != len(manifest_ordered):
                raise RuntimeError("Embeddings count does not match manifest length and no segment_ids provided for alignment")
            E_all = embs.astype(np.float32)
            L_all = [m['digit'] for m in manifest_ordered]
            P_all = [m['path'] for m in manifest_ordered]
        print(f"  Loaded {E_all.shape[0]} embeddings, shape {E_all.shape}")
    else:
        print("[PRECOMP] Extracting embeddings (this may take time; done once)...")
        E_all, L_all, P_all = extract_embeddings(manifest_ordered, encoder=encoder, device=device,
                                                 wav2vec2_model=wav2vec2_model, hubert_model=hubert_model,
                                                 conformer_model=conformer_model, label_key='digit')

    labels = L_all
    sets = [f"?:{m['split']}" for m in manifest_ordered]  # placeholder sets; specific forget assignment is per-run

    # dev/test masks by split only (independent of forget digit)
    mask_dev = np.array([m['split'] == 'dev' for m in manifest_ordered], dtype=bool)
    mask_test = np.array([m['split'] == 'test' for m in manifest_ordered], dtype=bool)

    # Fit whitener on dev samples (dev union)
    print("[PRECOMP] Fitting whitener on DEV (all dev samples)...")
    E_dev = E_all[mask_dev]
    if E_dev.shape[0] < 2:
        raise RuntimeError("Not enough dev samples to fit whitener")
    mu, W = fit_whitener(E_dev)
    Z_all = apply_whitener(E_all, mu, W)

    precomp = {
        "entries": entries,
        "manifest": manifest_ordered,
        "digits": digits,
        "E_all": E_all,
        "labels": labels,
        "paths": P_all,
        "mask_dev": mask_dev,
        "mask_test": mask_test,
        "mu": mu,
        "W": W,
        "Z_all": Z_all
    }
    return precomp


# ----------------------------
# Train SAE (once per seed)
# ----------------------------
def train_sae_for_seed(precomp, seed: int, sae_multiplier: float, sae_variant: str, sae_topk: int,
                       lambda_sparsity: float, sae_epochs: int, device: str):
    """
    Train SAE on Z_train (dev union) for the given seed. Returns trained SAE, training_time_seconds,
    and A_all (activations) computed on Z_all.
    """
    seed_all(seed)
    Z_all = precomp["Z_all"]
    mask_dev = precomp["mask_dev"]

    print(f"[SAE] Training SAE for seed {seed} on all dev samples (once per seed)...")
    Z_train = Z_all[mask_dev]
    perm = np.random.permutation(Z_train.shape[0])
    cut = int(0.8 * len(perm)) if len(perm) > 1 else int(0.5 * len(perm))
    if cut <= 0:  # fallback
        Z_sae_train, Z_sae_val = Z_train, Z_train
    else:
        Z_sae_train, Z_sae_val = Z_train[perm[:cut]], Z_train[perm[cut:]]

    sae = SAE(input_dim=Z_all.shape[1], hidden_mult=sae_multiplier, k=sae_topk, variant=sae_variant)
    t_sae_start = time.time()
    sae = train_sae(sae, Z_sae_train, Z_sae_val, epochs=sae_epochs, batch_size=64, lr=1e-3, device=device, lambda_sparsity=lambda_sparsity)
    t_sae_end = time.time()
    sae_training_time = float(t_sae_end - t_sae_start)

    # encode all activations once for this seed (reused across digits)
    print(f"[SAE] Encoding SAE activations for all samples (seed {seed})...")
    A_all = sae_encode_batch(sae, Z_all, device=device, batch=8192)

    # get decoder weights for U building later
    dec_w = sae.dec.weight.detach().cpu().numpy()

    return sae, sae_training_time, A_all, dec_w


# ----------------------------
# Per-digit run (uses precomputed and trained SAE)
# ----------------------------
def single_forget_run(precomp, sae, dec_w, A_all, clf, seed: int, forget_digit: str, sae_params: Dict, edit_params: Dict, compute_ms: bool = True, save_artifacts: bool = True, save_dir: Optional[str] = None):
    """
    Run the per-forget digit steps (feature stats, select features, build U, LEACE,
    SAUCE gating, editing, evaluation). Returns `results` (dict), editing_time_seconds.
    If save_artifacts==False, nothing is written to disk.
    """
    manifest = precomp["manifest"]
    labels = precomp["labels"]
    E_all = precomp["E_all"]
    Z_all = precomp["Z_all"]
    mu = precomp["mu"]
    W = precomp["W"]
    mask_dev_global = precomp["mask_dev"]
    mask_test_global = precomp["mask_test"]

    # Build per-run sets (forget vs retain)
    sets = [f"{'forget' if m['digit'] == forget_digit else 'retain'}:{m['split']}" for m in manifest]

    M_ret_dev = np.array([s == 'retain:dev' for s in sets], dtype=bool)
    M_fgt_dev = np.array([s == 'forget:dev' for s in sets], dtype=bool)
    M_ret_test = np.array([s == 'retain:test' for s in sets], dtype=bool)
    M_fgt_test = np.array([s == 'forget:test' for s in sets], dtype=bool)

    # Feature mining (depends on forget digit)
    Z_ret_dev = Z_all[M_ret_dev]
    Z_fgt_dev = Z_all[M_fgt_dev]
    A_ret = A_all[M_ret_dev]
    A_fgt = A_all[M_fgt_dev]

    # optionally compute Monosemanticity scores (skip for --run_all to save time)
    ms_original_stats, ms_sae_stats, ms_final_stats = None, None, None
    if compute_ms:
        print("[MS] Computing Monosemanticity scores (optional, single-run mode)...")
        E_all_norm = l2_normalize(E_all)
        ms_original = calculate_monosemanticity_score(E_all_norm, E_all_norm, top_m=20, min_activations=5)
        ms_original_stats = compute_monosemanticity_statistics(ms_original)
        A_all_local = A_all
        ms_sae = calculate_monosemanticity_score(A_all_local, E_all_norm, top_m=20, min_activations=5)
        ms_sae_stats = compute_monosemanticity_statistics(ms_sae)

    print("[FEATURE] Computing feature stats and selecting features for this forget digit...")
    stats = feature_stats(A_fgt, A_ret)
    feat_idx_global = select_features_from_stats(stats, top_k=sae_params.get("build_U_top", 200), fdr=0.05)
    feat_idx = feat_idx_global

    # Build U subspace from decoder using selected features
    print("[U] Building U subspace for this forget digit...")
    dec_pruned = prune_and_normalize_decoder(dec_w, prune_threshold=1e-6)
    U = build_U_from_decoder(dec_w, feat_idx)
    # fallback
    if U is None:
        U = np.zeros((dec_w.shape[1], 0))

    # LEACE projector (depends on forget digit)
    P_r = np.eye(U.shape[1]) if U.shape[1] > 0 else np.eye(0)
    mu_r = None
    if edit_params.get("leace", True):
        print("[LEACE] Computing LEACE projector (per-forget-run)...")
        # Note: we use split==dev indices for Y vector positions (dev split independent of set)
        dev_indices = np.where(mask_dev_global)[0]
        # Build Y_dev in order of dev_indices
        Y_dev = np.array([1 if manifest[i]['digit'] == forget_digit else 0 for i in dev_indices], dtype=int).reshape(-1, 1)
        try:
            P_r, mu_r = leace_projector_in_U(Z_all[mask_dev_global], U, Y_dev, ridge=edit_params.get("leace_ridge", 1e-6))
        except Exception as e:
            print(f"  Warning: LEACE computation failed ({e}), falling back to identity projector")
            P_r = np.eye(U.shape[1]) if U.shape[1] > 0 else np.eye(0)
            mu_r = np.zeros((1, U.shape[1]))
    else:
        P_r = np.eye(U.shape[1]) if U.shape[1] > 0 else np.eye(0)
        mu_r = np.zeros((1, U.shape[1]))

    # Editing: SAUCE gating + LEACE + noise injection + invert/normalize
    t_edit_start = time.time()
    Z_edit = Z_all.copy()
    beta = edit_params.get("beta", 0.0)
    if beta > 0.0:
        print("  applying SAUCE gating with beta", beta)
        Z_edit = apply_sauce_gating(sae, Z_edit, feat_idx, beta=beta, device=sae_params.get("device", "cuda"), batch=8192)
    if edit_params.get("leace", True):
        Z_edit = apply_subspace_leace(Z_edit, U, P_r, mu_r)
    noise_sigma = edit_params.get("noise_sigma", 0.0)
    if noise_sigma and noise_sigma > 0:
        Z_edit = inject_noise_in_subspace(Z_edit, U, sigma=noise_sigma)
    E_edit = invert_whitener(Z_edit, precomp_mu := mu, precomp_W := W)
    E_all_norm = l2_normalize(E_all)
    E_edit_norm = l2_normalize(E_edit)
    t_edit_end = time.time()
    editing_time = float(t_edit_end - t_edit_start)

    # Evaluation: EERs
    print("[EVAL] Computing EERs and per-class metrics for this forget digit...")
    test_mask_for_trials = M_ret_test | M_fgt_test
    test_trials, test_labels = build_verification_trials(labels, [forget_digit], test_mask_for_trials)
    scores_test_before = compute_scores(E_all_norm, test_trials)
    scores_test_after = compute_scores(E_edit_norm, test_trials)
    eer_before = eer_from_scores(scores_test_before, test_labels)
    eer_after = eer_from_scores(scores_test_after, test_labels)

    per_class_test = per_class_eers(E_all_norm, E_edit_norm, labels, mask=test_mask_for_trials, rng_seed=seed)
    per_class_ret = {c: vals for c, vals in per_class_test.items() if c != forget_digit}
    per_class_fgt = {c: vals for c, vals in per_class_test.items() if c == forget_digit}
    retain_avg_before, retain_avg_after = avg_eers(per_class_ret)
    forget_avg_before, forget_avg_after = avg_eers(per_class_fgt)

    # Classifier evaluations (classifier trained once per seed outside)
    print("[EVAL] Classifier accuracy (retain vs forget) for this forget digit...")
    all_digits = sorted(list(set(labels)))
    digit2id = {d: i for i, d in enumerate(all_digits)}
    # Build test indices for classifier evaluation
    E_ret_test = E_all_norm[M_ret_test]
    L_ret_test = [digit2id[d] for d in np.array(labels)[M_ret_test]]
    E_fgt_test = E_all_norm[M_fgt_test]
    L_fgt_test = [digit2id[d] for d in np.array(labels)[M_fgt_test]]
    E_ret_test_edit = E_edit_norm[M_ret_test]
    E_fgt_test_edit = E_edit_norm[M_fgt_test]

    acc_ret_before = evaluate_classifier(clf, E_ret_test, L_ret_test, device=sae_params.get("device", "cuda"))
    acc_fgt_before = evaluate_classifier(clf, E_fgt_test, L_fgt_test, device=sae_params.get("device", "cuda"))
    acc_ret_after = evaluate_classifier(clf, E_ret_test_edit, L_ret_test, device=sae_params.get("device", "cuda"))
    acc_fgt_after = evaluate_classifier(clf, E_fgt_test_edit, L_fgt_test, device=sae_params.get("device", "cuda"))

    results = {
        # requested metrics: forget & retain accuracy before/after (classifier)
        "resnet_acc_forget_before": acc_fgt_before,
        "resnet_acc_forget_after": acc_fgt_after,
        "resnet_acc_retain_before": acc_ret_before,
        "resnet_acc_retain_after": acc_ret_after,
        # requested metrics: forget and mean retain EER (before/after)
        "forget_EER_before_avg": forget_avg_before,
        "forget_EER_after_avg": forget_avg_after,
        "retain_EER_before_avg": retain_avg_before,
        "retain_EER_after_avg": retain_avg_after,
        # global EERs (kept for reference if desired)
        "EER_before_raw": eer_before,
        "EER_after_raw": eer_after
    }

    # Save optional artifacts only when requested (single-run or debugging). For --run_all we set save_artifacts=False.
    if save_artifacts and save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        stats_path = os.path.join(save_dir, "verification_eer_summary.json")
        with open(stats_path, "w") as fh:
            json.dump(results, fh, indent=2)
        # minimal interpretability dump if requested
        np.savez_compressed(os.path.join(save_dir, "interpretability_data.npz"),
                            E_all=E_all, E_edit=E_edit, Z_all=Z_all, Z_edit=Z_edit, U=U,
                            labels=np.array(labels), sets=np.array(sets), manifest=np.array(manifest), forget_digits=np.array([forget_digit]))
        torch.save(sae.state_dict(), os.path.join(save_dir, "sae_state_dict.pt"))
        np.savez_compressed(os.path.join(save_dir, "whitener.npz"), mu=mu, W=W)

    return results, editing_time


# ----------------------------
# Multi-run orchestration
# ----------------------------
def run_all_forget_digits(args):
    """
    Orchestrates multi-run optimized execution:
      - Precompute embeddings/whitener once
      - For each seed: train SAE once and classifier once
      - For each digit: run editing/evaluation (no per-run artifacts saved)
      - Write a single consolidated CSV with one row per (seed, forget_digit)
    """
    base_save_dir = args.save_dir
    os.makedirs(base_save_dir, exist_ok=True)
    seeds = args.seeds if args.seeds is not None else [1337, 1338, 1339]
    manifest_seed = args.split_seed if hasattr(args, "split_seed") else args.seed

    # 1) Precompute embeddings & whitener once (manifest built once)
    precomp = precompute_embeddings_and_whitener(
        data_root=args.data_root,
        embeddings_path=args.embeddings_path,
        dev_frac_per_digit=args.dev_frac_per_digit,
        manifest_seed=manifest_seed,
        encoder=args.encoder,
        device=args.device,
        wav2vec2_model=args.wav2vec2_model if hasattr(args, "wav2vec2_model") else "facebook/wav2vec2-base",
        hubert_model=args.hubert_model if hasattr(args, "hubert_model") else "facebook/hubert-base-ls960",
        conformer_model=args.conformer_model if hasattr(args, "conformer_model") else "facebook/wav2vec2-conformer-rel-pos-large"
    )

    digits = precomp["digits"]
    total_runs = len(seeds) * len(digits)
    print(f"[MULTI] Will run {total_runs} runs: {len(seeds)} seeds x {len(digits)} digits")

    csv_path = os.path.join(base_save_dir, "multi_run_results.csv")
    fieldnames = [
        "seed", "forget_digit",
        "resnet_acc_forget_before", "resnet_acc_forget_after",
        "resnet_acc_retain_before", "resnet_acc_retain_after",
        "forget_EER_before_avg", "forget_EER_after_avg",
        "retain_EER_before_avg", "retain_EER_after_avg",
        "sae_training_time_seconds", "editing_time_seconds", "total_sae_edit_time_seconds", "error"
    ]
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        # for each seed: train SAE once + classifier once
        for seed in seeds:
            try:
                sae, sae_train_time, A_all, dec_w = train_sae_for_seed(
                    precomp, seed,
                    sae_multiplier=args.sae_multiplier,
                    sae_variant=args.sae_variant,
                    sae_topk=args.sae_topk,
                    lambda_sparsity=args.lambda_sparsity,
                    sae_epochs=args.sae_epochs,
                    device=args.device
                )

                # Precompute normalized embeddings and train classifier once per seed (train on dev, validate on test)
                E_all = precomp["E_all"]
                labels = precomp["labels"]
                mask_dev = precomp["mask_dev"]
                mask_test = precomp["mask_test"]
                E_all_norm = l2_normalize(E_all)

                # classifier train/val (same for all runs under this seed)
                print(f"[MULTI] Training classifier once for seed={seed} (dev->test)...")
                all_digits = sorted(list(set(labels)))
                digit2id = {d: i for i, d in enumerate(all_digits)}
                X_cls_train = E_all_norm[mask_dev]
                y_cls_train = [digit2id[d] for i, d in enumerate(labels) if mask_dev[i]]
                X_cls_val = E_all_norm[mask_test]
                y_cls_val = [digit2id[d] for i, d in enumerate(labels) if mask_test[i]]
                # reuse existing classifier training function
                clf = train_classifier(X_cls_train, y_cls_train, X_cls_val, y_cls_val,
                                       num_classes=len(all_digits), device=args.device, epochs=args.classifier_epochs if hasattr(args, "classifier_epochs") else 20,
                                       batch_size=512, lr=1e-3)

                # now iterate digits (no per-run saving)
                for digit in digits:
                    print(f"\n[MULTI] seed={seed} forget_digit={digit} ...")
                    try:
                        results, editing_time = single_forget_run(
                            precomp=precomp, sae=sae, dec_w=dec_w, A_all=A_all, clf=clf,
                            seed=seed, forget_digit=digit,
                            sae_params={"device": args.device, "build_U_top": args.build_U_top},
                            edit_params={"beta": args.beta, "leace": not args.no_leace, "leace_ridge": args.leace_ridge, "noise_sigma": args.noise_sigma},
                            compute_ms=False,  # skip MS in multi-run to save time
                            save_artifacts=False,  # do not write per-run artifacts
                            save_dir=None
                        )

                        total_time = float(sae_train_time + editing_time)

                        row = {
                            "seed": seed,
                            "forget_digit": digit,
                            "resnet_acc_forget_before": results.get("resnet_acc_forget_before", ""),
                            "resnet_acc_forget_after": results.get("resnet_acc_forget_after", ""),
                            "resnet_acc_retain_before": results.get("resnet_acc_retain_before", ""),
                            "resnet_acc_retain_after": results.get("resnet_acc_retain_after", ""),
                            "forget_EER_before_avg": results.get("forget_EER_before_avg", ""),
                            "forget_EER_after_avg": results.get("forget_EER_after_avg", ""),
                            "retain_EER_before_avg": results.get("retain_EER_before_avg", ""),
                            "retain_EER_after_avg": results.get("retain_EER_after_avg", ""),
                            "sae_training_time_seconds": float(sae_train_time),
                            "editing_time_seconds": float(editing_time),
                            "total_sae_edit_time_seconds": float(total_time),
                            "error": ""
                        }
                        writer.writerow(row)
                        fh.flush()
                        print(f"[MULTI] wrote row for seed={seed}, digit={digit}")
                    except Exception as e:
                        tb = traceback.format_exc()
                        print(f"[ERROR] run seed={seed}, digit={digit}: {e}\n{tb}")
                        err_row = {k: "" for k in fieldnames}
                        err_row.update({"seed": seed, "forget_digit": digit, "error": str(e)})
                        writer.writerow(err_row)
                        fh.flush()

            except Exception as e:
                tb = traceback.format_exc()
                print(f"[ERROR] during seed-level work for seed={seed}: {e}\n{tb}")
                # If SAE training failed for this seed, write an error row for all its digits
                for digit in digits:
                    err_row = {k: "" for k in fieldnames}
                    err_row.update({"seed": seed, "forget_digit": digit, "error": f"seed-level failure: {str(e)}"})
                    writer.writerow(err_row)
                fh.flush()

    print(f"\n[MULTI] finished. consolidated CSV: {csv_path}")


# ----------------------------
# Single-run entry (no backward compatibility)
# ----------------------------
def run_pipeline(
    data_root: str, save_dir: str = "./runs/exp", device: str = "cuda", seed: int = 1337,
    dev_frac_per_digit: float = 0.75, sae_multiplier: float = 4.0,
    sae_variant: str = "topk", sae_topk: int = 128, lambda_sparsity: float = 0.0, sae_epochs: int = 30,
    build_U_top: int = 200, beta: float = 0.0, leace: bool = True, leace_ridge: float = 1e-6,
    noise_sigma: float = 0.02, forget_digits: Optional[List[str]] = None,
    encoder: str = "ecapa", embeddings_path: str = None,
    wav2vec2_model: str = "facebook/wav2vec2-base",
    hubert_model: str = "facebook/hubert-base-ls960",
    conformer_model: str = "facebook/wav2vec2-conformer-rel-pos-large",
    save_artifacts: bool = True
):
    """
    Single-run pipeline for AudioMNIST digit forgetting.
    """
    # Reuse precompute + seed-specific SAE training + single_forget_run
    precomp = precompute_embeddings_and_whitener(
        data_root=data_root,
        embeddings_path=embeddings_path,
        dev_frac_per_digit=dev_frac_per_digit,
        manifest_seed=seed,
        encoder=encoder,
        device=device,
        wav2vec2_model=wav2vec2_model,
        hubert_model=hubert_model,
        conformer_model=conformer_model
    )

    # For single-run: we expect forget_digits provided (list or single)
    if forget_digits is None:
        raise ValueError("run_pipeline requires forget_digits for single-run usage")
    if isinstance(forget_digits, str):
        forget_digits = [forget_digits]

    # Train SAE (for the provided seed)
    sae, sae_train_time, A_all, dec_w = train_sae_for_seed(
        precomp, seed,
        sae_multiplier=sae_multiplier,
        sae_variant=sae_variant,
        sae_topk=sae_topk,
        lambda_sparsity=lambda_sparsity,
        sae_epochs=sae_epochs,
        device=device
    )

    # Train classifier once for this seed (dev->test)
    E_all = precomp["E_all"]
    labels = precomp["labels"]
    mask_dev = precomp["mask_dev"]
    mask_test = precomp["mask_test"]
    E_all_norm = l2_normalize(E_all)
    all_digits = sorted(list(set(labels)))
    digit2id = {d: i for i, d in enumerate(all_digits)}
    X_cls_train = E_all_norm[mask_dev]
    y_cls_train = [digit2id[d] for i, d in enumerate(labels) if mask_dev[i]]
    X_cls_val = E_all_norm[mask_test]
    y_cls_val = [digit2id[d] for i, d in enumerate(labels) if mask_test[i]]
    clf = train_classifier(X_cls_train, y_cls_train, X_cls_val, y_cls_val, num_classes=len(all_digits), device=device, epochs=20, batch_size=512, lr=1e-3)

    # Run per-forget_digit (if multiple provided, run sequentially and save separate artifact dirs)
    all_results = {}
    for digit in forget_digits:
        run_save_dir = os.path.join(save_dir, f"seed{seed}_forget_{str(digit)}")
        os.makedirs(run_save_dir, exist_ok=True)
        results, editing_time = single_forget_run(
            precomp=precomp, sae=sae, dec_w=dec_w, A_all=A_all, clf=clf,
            seed=seed, forget_digit=digit,
            sae_params={"device": device, "build_U_top": build_U_top},
            edit_params={"beta": beta, "leace": leace, "leace_ridge": leace_ridge, "noise_sigma": noise_sigma},
            compute_ms=True,
            save_artifacts=save_artifacts,
            save_dir=run_save_dir if save_artifacts else None
        )
        # Add timing fields
        results["sae_training_time_seconds"] = float(sae_train_time)
        results["editing_time_seconds"] = float(editing_time)
        results["total_sae_edit_time_seconds"] = float(sae_train_time + editing_time)
        # Save summary JSON if requested
        if save_artifacts:
            stats_path = os.path.join(run_save_dir, "verification_eer_summary.json")
            with open(stats_path, "w") as fh:
                json.dump(results, fh, indent=2)
        all_results[digit] = results

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/data/veer/testing/new/datasets/audiomnist_data")
    parser.add_argument("--save_dir", default="./outputs_audiomnist")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--split_seed", type=int, default=1337, help="Seed used to build the dev/test splits (used once at precompute)")
    parser.add_argument("--dev_frac_per_digit", type=float, default=0.7)
    parser.add_argument("--sae_multiplier", type=float, default=10.0)
    parser.add_argument("--sae_variant", type=str, default="topk", choices=["topk", "standard"], help="SAE variant: topk or standard")
    parser.add_argument("--sae_topk", type=int, default=128, help="Top-K value for topk variant (ignored for standard)")
    parser.add_argument("--lambda_sparsity", type=float, default=0.0, help="L1 sparsity penalty for standard SAE variant")
    parser.add_argument("--sae_epochs", type=int, default=30)
    parser.add_argument("--build_U_top", type=int, default=200)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--no_leace", action="store_true")
    parser.add_argument("--leace_ridge", type=float, default=1e-6)
    parser.add_argument("--noise_sigma", type=float, default=0.02)
    parser.add_argument("--forget_digits", nargs="*", default=["0"])
    parser.add_argument("--encoder", type=str, default="ecapa")
    parser.add_argument("--conformer_model", type=str, default="facebook/wav2vec2-conformer-rel-pos-large")
    parser.add_argument("--embeddings_path", type=str, default=None)
    # Multi-run flags
    parser.add_argument("--run_all", action="store_true", help="Run for all digits (one-forget-per-run) across seeds; optimized to reuse extractor/SAE per-seed")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1337, 1338, 1339], help="Seeds to use for multi-run (space separated)")
    parser.add_argument("--classifier_epochs", type=int, default=20, help="Epochs to train classifier (used in multi-run per-seed classifier training)")

    args = parser.parse_args()

    if args.run_all:
        run_all_forget_digits(args)
    else:
        # Single-run
        run_pipeline(
            data_root=args.data_root, save_dir=args.save_dir, device=args.device, seed=args.seed,
            dev_frac_per_digit=args.dev_frac_per_digit,
            sae_multiplier=args.sae_multiplier, sae_variant=args.sae_variant, sae_topk=args.sae_topk, lambda_sparsity=args.lambda_sparsity, sae_epochs=args.sae_epochs,
            build_U_top=args.build_U_top, beta=args.beta, leace=not args.no_leace, leace_ridge=args.leace_ridge,
            noise_sigma=args.noise_sigma, forget_digits=args.forget_digits,
            encoder=args.encoder, embeddings_path=args.embeddings_path,
            wav2vec2_model=args.wav2vec2_model if hasattr(args, 'wav2vec2_model') else "facebook/wav2vec2-base",
            hubert_model=args.hubert_model if hasattr(args, 'hubert_model') else "facebook/hubert-base-ls960",
            conformer_model=args.conformer_model
        )