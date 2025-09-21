from __future__ import annotations
import os
import time
import json
import random
import csv
import traceback
import glob
from pathlib import Path
from typing import List, Dict, Optional
import argparse

import numpy as np
import torch

from common import (
    seed_all, extract_embeddings, fit_whitener, apply_whitener,
    invert_whitener, l2_normalize, SAE, train_sae, sae_encode_batch,
    prune_and_normalize_decoder, feature_stats, select_features_from_stats, build_U_from_decoder,
    leace_projector_in_U, apply_subspace_leace, apply_sauce_gating, inject_noise_in_subspace,
    compute_scores, eer_from_scores, build_verification_trials,
    train_classifier, evaluate_classifier, per_class_eers, avg_eers,
    # light monosemanticity helpers retained (like voxceleb)
    calculate_monosemanticity_score, compute_monosemanticity_statistics
)


# --------------------
# LibriSpeech manifest helpers (dataset-specific)
# --------------------
def read_librispeech_raw_entries(data_root: str) -> List[Dict]:
    """
    Scan LibriSpeech 'dev-clean' and 'dev-other' for .flac files and return raw entries list:
      {'path': full_path, 'speaker': speaker_id, 'subset': sub}
    """
    raw_entries = []
    for sub in ("dev-clean", "dev-other"):
        pattern = os.path.join(data_root, sub, "*", "*", "*.flac")
        for p in glob.glob(pattern):
            # path example: .../LibriSpeech/dev-clean/84/121123/84-121123-0000.flac
            parts = Path(p).parts
            # speaker id at position -3 (as in LibriSpeech layout)
            if len(parts) >= 3:
                spk = parts[-3]
            else:
                spk = "unknown"
            raw_entries.append({"path": p, "speaker": spk, "subset": sub})
    return raw_entries


def build_manifest(manifest_source: str, embeddings_path: Optional[str] = None,
                   dev_frac_per_spk: float = 0.75, seed: int = 1337) -> List[Dict]:
    """
    Build a manifest with 'split' keys (dev/test) and entries:
      {'path': ..., 'speaker': ..., 'segment_id': optional, 'split': 'dev'|'test'}
    If embeddings_path is provided, build manifest from embeddings labels/ids.
    Otherwise scan the LibriSpeech data_root (manifest_source).
    """
    if embeddings_path is not None:
        data = np.load(embeddings_path, allow_pickle=True)
        labels = data.get('labels', None)
        if labels is None:
            labels = data.get('y', None)
        ids = data.get('segment_ids', None)
        if ids is None:
            ids = data.get('ids', None)

        if labels is None:
            raise RuntimeError("embeddings .npz must contain 'labels' or 'y' when embeddings_path is used")

        labels = list(labels)
        ids = list(ids) if ids is not None else [None] * len(labels)
        raw_manifest = [{'path': f'dummy_{i}', 'speaker': labels[i], 'segment_id': ids[i]} for i in range(len(labels))]
    else:
        raw_manifest = read_librispeech_raw_entries(manifest_source)
        if len(raw_manifest) == 0:
            raise RuntimeError(f"No entries found under {manifest_source}/dev-clean or dev-other")

    # Group by speaker and split per-speaker indices into dev/test
    by_speaker = {}
    for i, m in enumerate(raw_manifest):
        by_speaker.setdefault(m['speaker'], []).append(i)

    rng = random.Random(seed)
    manifest = []
    for speaker, indices in by_speaker.items():
        rng.shuffle(indices)
        n_dev = max(1, int(len(indices) * dev_frac_per_spk))
        if len(indices) > 1 and n_dev >= len(indices):
            n_dev = len(indices) - 1
        dev_indices = indices[:n_dev]
        test_indices = indices[n_dev:]
        for idx in dev_indices:
            entry = raw_manifest[idx].copy()
            entry['split'] = 'dev'
            manifest.append(entry)
        for idx in test_indices:
            entry = raw_manifest[idx].copy()
            entry['split'] = 'test'
            manifest.append(entry)

    if len(manifest) == 0:
        raise RuntimeError("No entries found in manifest after splitting")
    return manifest


# ----------------------------
# Precompute / shared steps (dataset-specific manifest)
# ----------------------------
def precompute_embeddings_and_whitener(data_root: str, embeddings_path: Optional[str], dev_frac_per_spk: float,
                                       manifest_seed: int, encoder: str, device: str,
                                       wav2vec2_model: str, hubert_model: str, conformer_model: str):
    """
    Build manifest, load or extract embeddings aligned to manifest order,
    fit whitener on dev samples. Returns precomputed dict used across runs.
    """
    print("[PRECOMP] Building manifest and splits...")
    entries = build_manifest(data_root, embeddings_path, dev_frac_per_spk, manifest_seed)
    by = {}
    for e in entries:
        by.setdefault(e["speaker"], []).append(e)
    spks = sorted(by.keys())

    dev_manifest = [m for m in entries if m['split'] == 'dev']
    test_manifest = [m for m in entries if m['split'] == 'test']
    manifest_ordered = list(dev_manifest) + list(test_manifest)

    # Load or extract embeddings aligned to manifest_ordered
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
                if sid_s not in id_to_idx:
                    raise RuntimeError(f"segment_id {sid} not found in embeddings file keys")
                E_all[i] = embs[id_to_idx[sid_s]]
            L_all = [m['speaker'] for m in manifest_ordered]
            P_all = [m['path'] for m in manifest_ordered]
        else:
            print("  Warning: embeddings file lacks 'segment_ids' - assuming ordering matches manifest")
            if embs.shape[0] != len(manifest_ordered):
                raise RuntimeError("Embeddings count does not match manifest length and no segment_ids provided for alignment")
            E_all = embs.astype(np.float32)
            L_all = [m['speaker'] for m in manifest_ordered]
            P_all = [m['path'] for m in manifest_ordered]
        print(f"  Loaded {E_all.shape[0]} embeddings, shape {E_all.shape}")
    else:
        print("[PRECOMP] Extracting embeddings (this may take time; done once)...")
        E_all, L_all, P_all = extract_embeddings(manifest_ordered, encoder=encoder, device=device,
                                                 wav2vec2_model=wav2vec2_model, hubert_model=hubert_model,
                                                 conformer_model=conformer_model, label_key='speaker')

    labels = L_all

    # masks
    mask_dev = np.array([m['split'] == 'dev' for m in manifest_ordered], dtype=bool)
    mask_test = np.array([m['split'] == 'test' for m in manifest_ordered], dtype=bool)

    # Fit whitener on dev samples
    print("[PRECOMP] Fitting whitener on DEV samples...")
    E_dev = E_all[mask_dev]
    if E_dev.shape[0] < 2:
        raise RuntimeError("Not enough dev samples to fit whitener")
    mu, W = fit_whitener(E_dev)
    Z_all = apply_whitener(E_all, mu, W)

    precomp = {
        "entries": entries,
        "manifest": manifest_ordered,
        "speakers": spks,
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
# Train SAE (per seed)
# ----------------------------
def train_sae_for_seed(precomp, seed: int, sae_multiplier: float, sae_variant: str, sae_topk: int,
                       lambda_sparsity: float, sae_epochs: int, device: str):
    """
    Train SAE on dev union for the given seed. Returns SAE, training_time, A_all, dec_w.
    """
    seed_all(seed)
    Z_all = precomp["Z_all"]
    mask_dev = precomp["mask_dev"]

    print(f"[SAE] Training SAE for seed {seed} on all dev samples...")
    Z_train = Z_all[mask_dev]
    perm = np.random.permutation(Z_train.shape[0])
    cut = int(0.8 * len(perm)) if len(perm) > 1 else int(0.5 * len(perm))
    if cut <= 0:
        Z_sae_train, Z_sae_val = Z_train, Z_train
    else:
        Z_sae_train, Z_sae_val = Z_train[perm[:cut]], Z_train[perm[cut:]]

    sae = SAE(input_dim=Z_all.shape[1], hidden_mult=sae_multiplier, k=sae_topk, variant=sae_variant)
    t_sae_start = time.time()
    sae = train_sae(sae, Z_sae_train, Z_sae_val, epochs=sae_epochs, batch_size=64, lr=1e-3, device=device, lambda_sparsity=lambda_sparsity)
    sae_training_time = float(time.time() - t_sae_start)

    # encode activations for all samples
    print(f"[SAE] Encoding activations for all samples (seed {seed})...")
    A_all = sae_encode_batch(sae, Z_all, device=device, batch=8192)
    dec_w = sae.dec.weight.detach().cpu().numpy()

    return sae, sae_training_time, A_all, dec_w


# ----------------------------
# Per-speaker run (uses precomputed and trained SAE)
# ----------------------------
def single_forget_run(precomp, sae, dec_w, A_all, clf, seed: int, forget_spk: str,
                      sae_params: Dict, edit_params: Dict, compute_ms: bool = True,
                      save_artifacts: bool = True, save_dir: Optional[str] = None):
    """
    Run per-forget-speaker editing and evaluation. Returns (results_dict, editing_time_seconds).
    The function is intentionally lean (no heavy validation suite).
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
    sets = [f"{'forget' if m['speaker'] == forget_spk else 'retain'}:{m['split']}" for m in manifest]
    M_ret_dev = np.array([s == 'retain:dev' for s in sets], dtype=bool)
    M_fgt_dev = np.array([s == 'forget:dev' for s in sets], dtype=bool)
    M_ret_test = np.array([s == 'retain:test' for s in sets], dtype=bool)
    M_fgt_test = np.array([s == 'forget:test' for s in sets], dtype=bool)

    # Features / activations for mining
    A_ret = A_all[M_ret_dev]
    A_fgt = A_all[M_fgt_dev]

    # Optional monosemanticity (light, mirrors voxceleb)
    if compute_ms:
        try:
            E_all_norm = l2_normalize(E_all)
            ms_original = calculate_monosemanticity_score(E_all_norm, E_all_norm, top_m=20, min_activations=5)
            _ = compute_monosemanticity_statistics(ms_original)
            ms_sae = calculate_monosemanticity_score(A_all, E_all_norm, top_m=20, min_activations=5)
            _ = compute_monosemanticity_statistics(ms_sae)
        except Exception:
            # best-effort only; don't fail the run on ms computation
            pass

    # Feature selection
    stats = feature_stats(A_fgt, A_ret)
    feat_idx = select_features_from_stats(stats, top_k=sae_params.get("build_U_top", 200), fdr=0.05)

    # Build U subspace from decoder according to selected features
    dec_pruned = prune_and_normalize_decoder(dec_w, prune_threshold=1e-6)
    U = build_U_from_decoder(dec_w, feat_idx)
    if U is None:
        U = np.zeros((dec_w.shape[1], 0))

    # LEACE projector per-forget-run
    P_r = np.eye(U.shape[1]) if U.shape[1] > 0 else np.eye(0)
    mu_r = np.zeros((1, U.shape[1]))
    if edit_params.get("leace", True) and U.shape[1] > 0:
        # Build Y_dev over dev split (1 for forget_spk, 0 otherwise)
        dev_indices = np.where(mask_dev_global)[0]
        Y_dev = np.array([1 if manifest[i]['speaker'] == forget_spk else 0 for i in dev_indices], dtype=int).reshape(-1, 1)
        try:
            P_r, mu_r = leace_projector_in_U(Z_all[mask_dev_global], U, Y_dev, ridge=edit_params.get("leace_ridge", 1e-6))
        except Exception:
            # fallback to identity projector if LEACE fails
            P_r = np.eye(U.shape[1]) if U.shape[1] > 0 else np.eye(0)
            mu_r = np.zeros((1, U.shape[1]))

    # Editing (SAUCE gating, subspace LEACE, noise)
    t_edit_start = time.time()
    Z_edit = Z_all.copy()
    beta = edit_params.get("beta", 0.0)
    if beta > 0.0:
        Z_edit = apply_sauce_gating(sae, Z_edit, feat_idx, beta=beta, device=sae_params.get("device", "cuda"), batch=8192)
    if edit_params.get("leace", True):
        Z_edit = apply_subspace_leace(Z_edit, U, P_r, mu_r)
    noise_sigma = edit_params.get("noise_sigma", 0.0)
    if noise_sigma and noise_sigma > 0.0:
        Z_edit = inject_noise_in_subspace(Z_edit, U, sigma=noise_sigma)
    E_edit = invert_whitener(Z_edit, mu, W)
    E_all_norm = l2_normalize(E_all)
    E_edit_norm = l2_normalize(E_edit)
    t_edit_end = time.time()
    editing_time = float(t_edit_end - t_edit_start)

    # Evaluation - verification EERs (forget vs rest)
    test_mask_for_trials = M_ret_test | M_fgt_test
    test_trials, test_labels = build_verification_trials(labels, [forget_spk], test_mask_for_trials)
    scores_before = compute_scores(E_all_norm, test_trials)
    scores_after = compute_scores(E_edit_norm, test_trials)
    eer_before = eer_from_scores(scores_before, test_labels)
    eer_after = eer_from_scores(scores_after, test_labels)

    per_class = per_class_eers(E_all_norm, E_edit_norm, labels, mask=test_mask_for_trials, rng_seed=seed)
    per_class_ret = {c: vals for c, vals in per_class.items() if c != forget_spk}
    per_class_fgt = {c: vals for c, vals in per_class.items() if c == forget_spk}
    retain_avg_before, retain_avg_after = avg_eers(per_class_ret)
    forget_avg_before, forget_avg_after = avg_eers(per_class_fgt)

    # Classifier metrics (classifier trained once per seed outside)
    all_spks = sorted(list(set(labels)))
    spk2id = {s: i for i, s in enumerate(all_spks)}
    E_ret_test = E_all_norm[M_ret_test]
    L_ret_test = [spk2id[d] for d in np.array(labels)[M_ret_test]]
    E_fgt_test = E_all_norm[M_fgt_test]
    L_fgt_test = [spk2id[d] for d in np.array(labels)[M_fgt_test]]
    E_ret_test_edit = E_edit_norm[M_ret_test]
    E_fgt_test_edit = E_edit_norm[M_fgt_test]

    acc_ret_before = evaluate_classifier(clf, E_ret_test, L_ret_test, device=sae_params.get("device", "cuda"))
    acc_fgt_before = evaluate_classifier(clf, E_fgt_test, L_fgt_test, device=sae_params.get("device", "cuda"))
    acc_ret_after = evaluate_classifier(clf, E_ret_test_edit, L_ret_test, device=sae_params.get("device", "cuda"))
    acc_fgt_after = evaluate_classifier(clf, E_fgt_test_edit, L_fgt_test, device=sae_params.get("device", "cuda"))

    results = {
        "resnet_acc_forget_before": acc_fgt_before,
        "resnet_acc_forget_after": acc_fgt_after,
        "resnet_acc_retain_before": acc_ret_before,
        "resnet_acc_retain_after": acc_ret_after,
        "forget_EER_before_avg": forget_avg_before,
        "forget_EER_after_avg": forget_avg_after,
        "retain_EER_before_avg": retain_avg_before,
        "retain_EER_after_avg": retain_avg_after,
        "EER_before_raw": eer_before,
        "EER_after_raw": eer_after,
        "editing_time_seconds": float(editing_time)
    }

    # Optionally save minimal artifacts (SAE state, whitener, interpretability dump)
    if save_artifacts and save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        stats_path = os.path.join(save_dir, "verification_eer_summary.json")
        with open(stats_path, "w") as fh:
            json.dump(results, fh, indent=2)
        # Minimal dumps (no heavy validation)
        np.savez_compressed(os.path.join(save_dir, "interpretability_data.npz"),
                            E_all=E_all, E_edit=E_edit, Z_all=Z_all, Z_edit=Z_edit, U=U,
                            labels=np.array(labels), sets=np.array(sets), forget_spk=np.array([forget_spk]))
        torch.save(sae.state_dict(), os.path.join(save_dir, "sae_state_dict.pt"))
        np.savez_compressed(os.path.join(save_dir, "whitener.npz"), mu=mu, W=W)

    return results, editing_time


# ----------------------------
# Multi-run orchestration (voxceleb-like)
# ----------------------------
def run_all_forget_speakers(args):
    """
    Optimized orchestration for running one-forget-per-run across many speakers and seeds:
      - Precompute embeddings/whitener once
      - For each seed: train SAE once, train classifier once (dev->test)
      - For each speaker: call single_forget_run (no per-run artifacts)
      - Write consolidated CSV with one row per (seed, forget_speaker)
    """
    base_save_dir = args.save_dir
    os.makedirs(base_save_dir, exist_ok=True)
    seeds = args.seeds if args.seeds is not None else [1337, 1338, 1339]
    manifest_seed = args.split_seed if hasattr(args, "split_seed") else args.seed

    # Precompute (only once)
    precomp = precompute_embeddings_and_whitener(
        data_root=args.data_root,
        embeddings_path=args.embeddings_path,
        dev_frac_per_spk=args.dev_frac_per_spk,
        manifest_seed=manifest_seed,
        encoder=args.encoder,
        device=args.device,
        wav2vec2_model=getattr(args, "wav2vec2_model", "facebook/wav2vec2-base"),
        hubert_model=getattr(args, "hubert_model", "facebook/hubert-base-ls960"),
        conformer_model=getattr(args, "conformer_model", "facebook/wav2vec2-conformer-rel-pos-large")
    )

    speakers = precomp["speakers"]
    total_runs = len(seeds) * len(speakers)
    print(f"[MULTI] Will run {total_runs} runs: {len(seeds)} seeds x {len(speakers)} speakers")

    csv_path = os.path.join(base_save_dir, "multi_run_results.csv")
    fieldnames = [
        "seed", "forget_spk",
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

                # Train classifier once for this seed (dev -> test)
                E_all = precomp["E_all"]
                labels = precomp["labels"]
                mask_dev = precomp["mask_dev"]
                mask_test = precomp["mask_test"]
                E_all_norm = l2_normalize(E_all)

                print(f"[MULTI] Training classifier once for seed={seed} (dev->test)...")
                all_spks = sorted(list(set(labels)))
                spk2id = {s: i for i, s in enumerate(all_spks)}
                X_cls_train = E_all_norm[mask_dev]
                y_cls_train = [spk2id[s] for i, s in enumerate(labels) if mask_dev[i]]
                X_cls_val = E_all_norm[mask_test]
                y_cls_val = [spk2id[s] for i, s in enumerate(labels) if mask_test[i]]
                clf = train_classifier(X_cls_train, y_cls_train, X_cls_val, y_cls_val,
                                       num_classes=len(all_spks), device=args.device,
                                       epochs=getattr(args, "classifier_epochs", 20),
                                       batch_size=512, lr=1e-3)

                # Iterate speakers (no per-run saving)
                for spk in speakers:
                    print(f"\n[MULTI] seed={seed} forget_spk={spk} ...")
                    try:
                        results, editing_time = single_forget_run(
                            precomp=precomp, sae=sae, dec_w=dec_w, A_all=A_all, clf=clf,
                            seed=seed, forget_spk=spk,
                            sae_params={"device": args.device, "build_U_top": args.build_U_top},
                            edit_params={"beta": args.beta, "leace": not args.no_leace, "leace_ridge": args.leace_ridge, "noise_sigma": args.noise_sigma},
                            compute_ms=False,
                            save_artifacts=False,
                            save_dir=None
                        )

                        total_time = float(sae_train_time + editing_time)

                        row = {
                            "seed": seed,
                            "forget_spk": spk,
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
                        print(f"[MULTI] wrote row for seed={seed}, spk={spk}")
                    except Exception as e:
                        tb = traceback.format_exc()
                        print(f"[ERROR] run seed={seed}, spk={spk}: {e}\n{tb}")
                        err_row = {k: "" for k in fieldnames}
                        err_row.update({"seed": seed, "forget_spk": spk, "error": str(e)})
                        writer.writerow(err_row)
                        fh.flush()

            except Exception as e:
                tb = traceback.format_exc()
                print(f"[ERROR] during seed-level work for seed={seed}: {e}\n{tb}")
                # If SAE training failed for this seed, write an error row for all its speakers
                for spk in speakers:
                    err_row = {k: "" for k in fieldnames}
                    err_row.update({"seed": seed, "forget_spk": spk, "error": f"seed-level failure: {str(e)}"})
                    writer.writerow(err_row)
                fh.flush()

    print(f"\n[MULTI] finished. consolidated CSV: {csv_path}")


# ----------------------------
# Backwards-compatible single-run wrapper (keeps original semantics)
# ----------------------------
def run_pipeline(
    data_root: str, save_dir: str = "./runs/exp", device: str = "cuda", seed: int = 1337,
    retain_frac: float = 0.975, dev_frac_per_spk: float = 0.75, sae_multiplier: float = 4.0,
    sae_variant: str = "topk", sae_topk: int = 128, lambda_sparsity: float = 0.0, sae_epochs: int = 30,
    build_U_top: int = 200, beta: float = 0.0, leace: bool = True, leace_ridge: float = 1e-6,
    noise_sigma: float = 0.02, per_speaker_union: bool = False, forget_spks: Optional[List[str]] = None,
    encoder: str = "ecapa", embeddings_path: str = None,
    wav2vec2_model: str = "facebook/wav2vec2-base",
    hubert_model: str = "facebook/hubert-base-ls960",
    conformer_model: str = "facebook/wav2vec2-conformer-rel-pos-large",
    save_artifacts: bool = True
):
    """
    Backwards-compatible single-run flow that uses the same precompute/train/forget-run functions.
    """
    precomp = precompute_embeddings_and_whitener(
        data_root=data_root,
        embeddings_path=embeddings_path,
        dev_frac_per_spk=dev_frac_per_spk,
        manifest_seed=seed,
        encoder=encoder,
        device=device,
        wav2vec2_model=wav2vec2_model,
        hubert_model=hubert_model,
        conformer_model=conformer_model
    )

    if forget_spks is None:
        raise ValueError("run_pipeline requires forget_spks for single-run usage")
    if isinstance(forget_spks, str):
        forget_spks = [forget_spks]

    # Train SAE for the provided seed
    sae, sae_train_time, A_all, dec_w = train_sae_for_seed(
        precomp, seed,
        sae_multiplier=sae_multiplier,
        sae_variant=sae_variant,
        sae_topk=sae_topk,
        lambda_sparsity=lambda_sparsity,
        sae_epochs=sae_epochs,
        device=device
    )

    # Train classifier once (dev -> test)
    E_all = precomp["E_all"]
    labels = precomp["labels"]
    mask_dev = precomp["mask_dev"]
    mask_test = precomp["mask_test"]
    E_all_norm = l2_normalize(E_all)
    all_spks = sorted(list(set(labels)))
    spk2id = {s: i for i, s in enumerate(all_spks)}
    X_cls_train = E_all_norm[mask_dev]
    y_cls_train = [spk2id[s] for i, s in enumerate(labels) if mask_dev[i]]
    X_cls_val = E_all_norm[mask_test]
    y_cls_val = [spk2id[s] for i, s in enumerate(labels) if mask_test[i]]
    clf = train_classifier(X_cls_train, y_cls_train, X_cls_val, y_cls_val, num_classes=len(all_spks), device=device, epochs=20, batch_size=512, lr=1e-3)

    # Run per-forget_spk sequentially
    all_results = {}
    for spk in forget_spks:
        run_save_dir = os.path.join(save_dir, f"seed{seed}_forget_{str(spk)}")
        os.makedirs(run_save_dir, exist_ok=True)
        results, editing_time = single_forget_run(
            precomp=precomp, sae=sae, dec_w=dec_w, A_all=A_all, clf=clf,
            seed=seed, forget_spk=spk,
            sae_params={"device": device, "build_U_top": build_U_top},
            edit_params={"beta": beta, "leace": leace, "leace_ridge": leace_ridge, "noise_sigma": noise_sigma},
            compute_ms=True,
            save_artifacts=save_artifacts,
            save_dir=run_save_dir if save_artifacts else None
        )
        results["sae_training_time_seconds"] = float(sae_train_time)
        results["editing_time_seconds"] = float(editing_time)
        results["total_sae_edit_time_seconds"] = float(sae_train_time + editing_time)
        if save_artifacts:
            stats_path = os.path.join(run_save_dir, "verification_eer_summary.json")
            with open(stats_path, "w") as fh:
                json.dump(results, fh, indent=2)
        all_results[spk] = results

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/data/veer/testing/new/datasets/librispeech_data_dev/LibriSpeech", help="LibriSpeech root containing dev-clean/dev-other")
    parser.add_argument("--save_dir", type=str, default="./outputs_librispeech")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--split_seed", type=int, default=1337, help="Seed for dev/test split")
    parser.add_argument("--dev_frac_per_spk", type=float, default=0.8)
    parser.add_argument("--sae_multiplier", type=float, default=10.0)
    parser.add_argument("--sae_variant", type=str, default="topk", choices=["topk", "standard"])
    parser.add_argument("--sae_topk", type=int, default=128)
    parser.add_argument("--lambda_sparsity", type=float, default=0.0)
    parser.add_argument("--sae_epochs", type=int, default=100)
    parser.add_argument("--build_U_top", type=int, default=200)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--no_leace", action="store_true")
    parser.add_argument("--leace_ridge", type=float, default=1e-6)
    parser.add_argument("--noise_sigma", type=float, default=0.02)
    parser.add_argument("--forget_spks", nargs="*", default=["174"], help="Speakers to forget")
    parser.add_argument("--encoder", type=str, default="ecapa")
    parser.add_argument("--conformer_model", type=str, default="facebook/wav2vec2-conformer-rel-pos-large")
    parser.add_argument("--wav2vec2_model", type=str, default="facebook/wav2vec2-base")
    parser.add_argument("--hubert_model", type=str, default="facebook/hubert-base-ls960")
    parser.add_argument("--embeddings_path", type=str, default=None)
    parser.add_argument("--run_all", action="store_true", help="Run across all speakers and seeds")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1337, 1338, 1339])
    parser.add_argument("--classifier_epochs", type=int, default=20)
    args = parser.parse_args()

    if args.run_all:
        run_all_forget_speakers(args)
    else:
        run_pipeline(
            data_root=args.data_root, save_dir=args.save_dir, device=args.device, seed=args.seed,
            dev_frac_per_spk=args.dev_frac_per_spk,
            sae_multiplier=args.sae_multiplier, sae_variant=args.sae_variant, sae_topk=args.sae_topk, lambda_sparsity=args.lambda_sparsity, sae_epochs=args.sae_epochs,
            build_U_top=args.build_U_top, beta=args.beta, leace=not args.no_leace, leace_ridge=args.leace_ridge,
            noise_sigma=args.noise_sigma, forget_spks=args.forget_spks,
            encoder=args.encoder, embeddings_path=args.embeddings_path,
            wav2vec2_model=args.wav2vec2_model, hubert_model=args.hubert_model, conformer_model=args.conformer_model
        )
