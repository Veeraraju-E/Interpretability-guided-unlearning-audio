# speech_commands.py (updated)
from __future__ import annotations
import os
import time
import json
import random
import csv
import traceback
from typing import List, Dict, Optional
import argparse
import numpy as np
import torch
from pathlib import Path

from common import (
    seed_all, human_time, extract_embeddings, fit_whitener, apply_whitener,
    invert_whitener, l2_normalize, SAE, train_sae, sae_encode_batch,
    prune_and_normalize_decoder, feature_stats, select_features_from_stats, build_U_from_decoder,
    leace_projector_in_U, apply_subspace_leace, apply_sauce_gating, inject_noise_in_subspace,
    compute_scores, eer_from_scores, build_verification_trials,
    train_classifier, evaluate_classifier, per_class_eers, avg_eers, safe_linear_probe,
    # monosemanticity helpers
    calculate_monosemanticity_score, compute_monosemanticity_statistics
)


# -------------------- Manifest --------------------
def build_manifest(data_root: str, dev_frac_per_cmd: float = 0.75, seed: int = 1337,
                   allowed_exts: List[str] = [".wav"]) -> List[Dict]:
    """
    Build manifest entries for Speech Commands dataset folder.
    Expects directory structure:
        <data_root>/<command_label>/*.wav

    Returns list of dicts:
        {"path": path, "speaker": spk, "command": command_label, "split": "dev"/"test", "segment_id": id}
    NOTE: we include a 'segment_id' field (full path string) so precomputed-embeddings alignment works.
    """
    data_root = os.path.expanduser(data_root)
    if not os.path.isdir(data_root):
        raise RuntimeError(f"Speech Commands root not found or not a directory: {data_root}")

    raw_entries = []
    data_root_path = Path(data_root)

    # Get command labels (subdirectories)
    command_dirs = [d for d in data_root_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if not command_dirs:
        raise RuntimeError(f"No command directories found in {data_root}")

    for command_dir in command_dirs:
        command_label = command_dir.name

        # gather audio files with allowed extensions
        audio_files = []
        for ext in allowed_exts:
            audio_files.extend(list(command_dir.glob(f"*{ext}")))
            audio_files.extend(list(command_dir.glob(f"*{ext.upper()}")))

        for audio_file in audio_files:
            fname = audio_file.stem  # filename without extension
            # Extract speaker id if possible. Typical pattern: <speakerid>_nohash_<utterance>
            parts = fname.split("_")
            if len(parts) >= 1 and len(parts[0]) > 0:
                spk = parts[0]
            else:
                spk = "unknown"
            raw_entries.append({
                "path": str(audio_file),
                "speaker": spk,
                "command": command_label,
                # segment_id as full path (helps alignment with embedding dumps that use paths)
                "segment_id": str(audio_file)
            })

    if len(raw_entries) == 0:
        raise RuntimeError(f"No Speech Commands audio files found under {data_root}")

    # group by command and create dev/test splits per command (seeded)
    by_command = {}
    for entry in raw_entries:
        by_command.setdefault(entry['command'], []).append(entry)

    rng = random.Random(seed)
    manifest = []
    for command, entries in by_command.items():
        rng.shuffle(entries)
        n_dev = max(1, int(len(entries) * dev_frac_per_cmd))
        if len(entries) > 1 and n_dev >= len(entries):
            n_dev = len(entries) - 1

        dev_entries = entries[:n_dev]
        test_entries = entries[n_dev:]

        for e in dev_entries:
            e_copy = e.copy()
            e_copy['split'] = 'dev'
            manifest.append(e_copy)
        for e in test_entries:
            e_copy = e.copy()
            e_copy['split'] = 'test'
            manifest.append(e_copy)

    return manifest


# ----------------------------
# Precompute / shared steps (mirrors voxceleb.py semantics)
# ----------------------------
def precompute_embeddings_and_whitener(manifest_path: str, embeddings_path: Optional[str], dev_frac_per_cmd: float,
                                       manifest_seed: int, encoder: str, device: str,
                                       wav2vec2_model: str, hubert_model: str, conformer_model: str):
    """
    Build manifest (dev/test splits), extract/load embeddings aligned to manifest order,
    fit a whitener on *dev* samples (using all dev samples), return a dict of precomputed objects.

    NOTE: 'manifest_path' is the SpeechCommands root directory (data_root).
    """
    print("[PRECOMP] Building manifest and splits (once)...")
    entries = build_manifest(manifest_path, dev_frac_per_cmd, manifest_seed)
    by = {}
    for e in entries:
        by.setdefault(e["command"], []).append(e)
    spks = sorted(by.keys())  # these are command labels for speech-commands

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
                if sid_s not in id_to_idx:
                    # try basename fallback
                    sid_basename = os.path.basename(sid_s)
                    if sid_basename in id_to_idx:
                        E_all[i] = embs[id_to_idx[sid_basename]]
                    else:
                        raise RuntimeError(f"segment_id {sid_s} not found in embeddings file keys")
                else:
                    E_all[i] = embs[id_to_idx[sid_s]]
            L_all = [m['command'] for m in manifest_ordered]
            P_all = [m['path'] for m in manifest_ordered]
        else:
            print("  Warning: embeddings file lacks 'segment_ids' - assuming ordering already matches manifest")
            if embs.shape[0] != len(manifest_ordered):
                raise RuntimeError("Embeddings count does not match manifest length and no segment_ids provided for alignment")
            E_all = embs.astype(np.float32)
            L_all = [m['command'] for m in manifest_ordered]
            P_all = [m['path'] for m in manifest_ordered]
        print(f"  Loaded {E_all.shape[0]} embeddings, shape {E_all.shape}")
    else:
        print("[PRECOMP] Extracting embeddings (this may take time; done once)...")
        E_all, L_all, P_all = extract_embeddings(manifest_ordered, encoder=encoder, device=device,
                                                 wav2vec2_model=wav2vec2_model, hubert_model=hubert_model,
                                                 conformer_model=conformer_model, label_key='command')

    labels = L_all
    sets = [f"?:{m['split']}" for m in manifest_ordered]  # placeholder sets; specific forget assignment is per-run

    # dev/test masks by split only (independent of forget class)
    mask_dev = np.array([m['split'] == 'dev' for m in manifest_ordered], dtype=bool)
    mask_test = np.array([m['split'] == 'test' for m in manifest_ordered], dtype=bool)

    # Fit whitener on dev samples (all dev samples)
    print("[PRECOMP] Fitting whitener on DEV (all dev samples)...")
    E_dev = E_all[mask_dev]
    if E_dev.shape[0] < 2:
        raise RuntimeError("Not enough dev samples to fit whitener")
    mu, W = fit_whitener(E_dev)
    Z_all = apply_whitener(E_all, mu, W)

    precomp = {
        "entries": entries,
        "manifest": manifest_ordered,
        "speakers": spks,   # NOTE: these are 'commands' in SpeechCommands context
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
# Train SAE (once per seed) - same as voxceleb.py
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
    if cut <= 0:
        Z_sae_train, Z_sae_val = Z_train, Z_train
    else:
        Z_sae_train, Z_sae_val = Z_train[perm[:cut]], Z_train[perm[cut:]]

    sae = SAE(input_dim=Z_all.shape[1], hidden_mult=sae_multiplier, k=sae_topk, variant=sae_variant)
    t_sae_start = time.time()
    sae = train_sae(sae, Z_sae_train, Z_sae_val, epochs=sae_epochs, batch_size=64, lr=1e-3, device=device, lambda_sparsity=lambda_sparsity)
    t_sae_end = time.time()
    sae_training_time = float(t_sae_end - t_sae_start)

    # encode all activations once for this seed (reused across classes)
    print(f"[SAE] Encoding SAE activations for all samples (seed {seed})...")
    A_all = sae_encode_batch(sae, Z_all, device=device, batch=8192)

    # get decoder weights for U building later
    dec_w = sae.dec.weight.detach().cpu().numpy()

    return sae, sae_training_time, A_all, dec_w


# ----------------------------
# Per-class run (uses precomputed and trained SAE) - mirrors single_forget_run in voxceleb.py
# ----------------------------
def single_forget_run(precomp, sae, dec_w, A_all, clf, seed: int, forget_spk: str, sae_params: Dict, edit_params: Dict, compute_ms: bool = True, save_artifacts: bool = True, save_dir: Optional[str] = None):
    """
    Run the per-forget-class steps (feature stats, select features, build U, LEACE,
    SAUCE gating, editing, evaluation). Returns `results` (dict) and editing_time_seconds.
    Note: `forget_spk` is the command to forget in the SpeechCommands context (kept name for parity with voxceleb).
    """
    manifest = precomp["manifest"]
    labels = precomp["labels"]
    E_all = precomp["E_all"]
    Z_all = precomp["Z_all"]
    mu = precomp["mu"]
    W = precomp["W"]
    mask_dev_global = precomp["mask_dev"]
    mask_test_global = precomp["mask_test"]

    # Build per-run sets (forget vs retain) using 'command' field in manifest entries
    sets = [f"{'forget' if m['command'] == forget_spk else 'retain'}:{m['split']}" for m in manifest]

    M_ret_dev = np.array([s == 'retain:dev' for s in sets], dtype=bool)
    M_fgt_dev = np.array([s == 'forget:dev' for s in sets], dtype=bool)
    M_ret_test = np.array([s == 'retain:test' for s in sets], dtype=bool)
    M_fgt_test = np.array([s == 'forget:test' for s in sets], dtype=bool)

    # Feature mining (depends on forget class)
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

    print("[FEATURE] Computing feature stats and selecting features for this forget class...")
    stats = feature_stats(A_fgt, A_ret)
    feat_idx_global = select_features_from_stats(stats, top_k=sae_params.get("build_U_top", 200), fdr=0.05)
    feat_idx = feat_idx_global

    # Build U subspace from decoder using selected features
    print("[U] Building U subspace for this forget class...")
    dec_pruned = prune_and_normalize_decoder(dec_w, prune_threshold=1e-6)
    U = build_U_from_decoder(dec_w, feat_idx)
    if U is None:
        U = np.zeros((dec_w.shape[1], 0))

    # LEACE projector (depends on forget class)
    P_r = np.eye(U.shape[1]) if U.shape[1] > 0 else np.eye(0)
    mu_r = None
    if edit_params.get("leace", True):
        print("[LEACE] Computing LEACE projector (per-run)...")
        dev_indices = np.where(mask_dev_global)[0]
        Y_dev = np.array([1 if manifest[i]['command'] == forget_spk else 0 for i in dev_indices], dtype=int).reshape(-1, 1)
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

    # Evaluation: verification EERs
    print("[EVAL] Computing EERs and per-class metrics for this forget class...")
    test_mask_for_trials = M_ret_test | M_fgt_test
    test_trials, test_labels = build_verification_trials(labels, [forget_spk], test_mask_for_trials)
    scores_test_before = compute_scores(E_all_norm, test_trials)
    scores_test_after = compute_scores(E_edit_norm, test_trials)
    eer_before = eer_from_scores(scores_test_before, test_labels)
    eer_after = eer_from_scores(scores_test_after, test_labels)

    per_class_test = per_class_eers(E_all_norm, E_edit_norm, labels, mask=test_mask_for_trials, rng_seed=seed)
    per_class_ret = {c: vals for c, vals in per_class_test.items() if c != forget_spk}
    per_class_fgt = {c: vals for c, vals in per_class_test.items() if c == forget_spk}
    retain_avg_before, retain_avg_after = avg_eers(per_class_ret)
    forget_avg_before, forget_avg_after = avg_eers(per_class_fgt)

    # Classifier evaluations (classifier trained once per seed outside)
    print("[EVAL] Classifier accuracy (retain vs forget) for this forget class...")
    all_spks = sorted(list(set(labels)))
    spk2id = {s: i for i, s in enumerate(all_spks)}
    # Build test indices for classifier evaluation
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
        "EER_after_raw": eer_after
    }

    # Save optional artifacts only when requested (single-run or debugging). For --run_all we set save_artifacts=False.
    if save_artifacts and save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        stats_path = os.path.join(save_dir, "verification_eer_summary.json")
        with open(stats_path, "w") as fh:
            json.dump(results, fh, indent=2)
        # interpretability dump
        np.savez_compressed(os.path.join(save_dir, "interpretability_data.npz"),
                            E_all=E_all, E_edit=E_edit, Z_all=Z_all, Z_edit=Z_edit, U=U,
                            labels=np.array(labels), sets=np.array(sets), manifest=np.array(manifest), forget_spks=np.array([forget_spk]))
        torch.save(sae.state_dict(), os.path.join(save_dir, "sae_state_dict.pt"))
        np.savez_compressed(os.path.join(save_dir, "whitener.npz"), mu=mu, W=W)

    return results, editing_time


# ----------------------------
# Multi-run orchestration (mirrors voxceleb.run_all_forget_speakers)
# ----------------------------
def run_all_forget_speakers(args):
    """
    Orchestrates multi-run optimized execution:
      - Precompute embeddings/whitener once
      - For each seed: train SAE once and classifier once
      - For each command: run editing/evaluation (no per-run artifacts saved)
      - Write a single consolidated CSV with one row per (seed, forget_command)
    """
    base_save_dir = args.save_dir
    os.makedirs(base_save_dir, exist_ok=True)
    seeds = args.seeds if args.seeds is not None else [1337, 1338, 1339]
    manifest_seed = args.split_seed if hasattr(args, "split_seed") else args.seed

    # 1) Precompute embeddings & whitener once (manifest built once)
    precomp = precompute_embeddings_and_whitener(
        manifest_path=args.data_root,
        embeddings_path=args.embeddings_path,
        dev_frac_per_cmd=args.dev_frac_per_cmd,
        manifest_seed=manifest_seed,
        encoder=args.encoder,
        device=args.device,
        wav2vec2_model=args.wav2vec2_model if hasattr(args, "wav2vec2_model") else "facebook/wav2vec2-base",
        hubert_model=args.hubert_model if hasattr(args, "hubert_model") else "facebook/hubert-base-ls960",
        conformer_model=args.conformer_model if hasattr(args, "conformer_model") else "facebook/wav2vec2-conformer-rel-pos-large"
    )

    commands = precomp["speakers"]  # these are command labels
    total_runs = len(seeds) * len(commands)
    print(f"[MULTI] Will run {total_runs} runs: {len(seeds)} seeds x {len(commands)} commands")

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

                # Precompute normalized embeddings and train classifier once per seed (dev->test)
                E_all = precomp["E_all"]
                labels = precomp["labels"]
                mask_dev = precomp["mask_dev"]
                mask_test = precomp["mask_test"]
                E_all_norm = l2_normalize(E_all)

                # classifier train/val (same for all runs under this seed)
                print(f"[MULTI] Training classifier once for seed={seed} (dev->test)...")
                all_spks = sorted(list(set(labels)))
                spk2id = {s: i for i, s in enumerate(all_spks)}
                X_cls_train = E_all_norm[mask_dev]
                y_cls_train = [spk2id[s] for i, s in enumerate(labels) if mask_dev[i]]
                X_cls_val = E_all_norm[mask_test]
                y_cls_val = [spk2id[s] for i, s in enumerate(labels) if mask_test[i]]
                clf = train_classifier(X_cls_train, y_cls_train, X_cls_val, y_cls_val,
                                       num_classes=len(all_spks), device=args.device, epochs=args.classifier_epochs if hasattr(args, "classifier_epochs") else 20,
                                       batch_size=512, lr=1e-3)

                # now iterate commands (no per-run saving)
                for cmd in commands:
                    print(f"\n[MULTI] seed={seed} forget_cmd={cmd} ...")
                    try:
                        results, editing_time = single_forget_run(
                            precomp=precomp, sae=sae, dec_w=dec_w, A_all=A_all, clf=clf,
                            seed=seed, forget_spk=cmd,
                            sae_params={"device": args.device, "build_U_top": args.build_U_top},
                            edit_params={"beta": args.beta, "leace": not args.no_leace, "leace_ridge": args.leace_ridge, "noise_sigma": args.noise_sigma},
                            compute_ms=False,  # skip MS in multi-run to save time
                            save_artifacts=False,  # do not write per-run artifacts
                            save_dir=None
                        )

                        total_time = float(sae_train_time + editing_time)

                        row = {
                            "seed": seed,
                            "forget_spk": cmd,
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
                        print(f"[MULTI] wrote row for seed={seed}, cmd={cmd}")
                    except Exception as e:
                        tb = traceback.format_exc()
                        print(f"[ERROR] run seed={seed}, cmd={cmd}: {e}\n{tb}")
                        err_row = {k: "" for k in fieldnames}
                        err_row.update({"seed": seed, "forget_spk": cmd, "error": str(e)})
                        writer.writerow(err_row)
                        fh.flush()

            except Exception as e:
                tb = traceback.format_exc()
                print(f"[ERROR] during seed-level work for seed={seed}: {e}\n{tb}")
                # If SAE training failed for this seed, write an error row for all its commands
                for cmd in commands:
                    err_row = {k: "" for k in fieldnames}
                    err_row.update({"seed": seed, "forget_spk": cmd, "error": f"seed-level failure: {str(e)}"})
                    writer.writerow(err_row)
                fh.flush()

    print(f"\n[MULTI] finished. consolidated CSV: {csv_path}")


# ----------------------------
# Backwards-compatible single-run (keeps original semantics but reuses new helpers)
# ----------------------------
def run_pipeline(
    data_root: str, save_dir: str = "./runs/exp", device: str = "cuda", seed: int = 1337,
    retain_frac: float = 0.975, dev_frac_per_cmd: float = 0.75, sae_multiplier: float = 4.0,
    sae_variant: str = "topk", sae_topk: int = 128, lambda_sparsity: float = 0.0, sae_epochs: int = 30,
    build_U_top: int = 200, beta: float = 0.0, leace: bool = True, leace_ridge: float = 1e-6,
    noise_sigma: float = 0.02, per_speaker_union: bool = False, forget_commands: Optional[List[str]] = None,
    encoder: str = "ecapa", embeddings_path: str = None,
    wav2vec2_model: str = "facebook/wav2vec2-base",
    hubert_model: str = "facebook/hubert-base-ls960",
    conformer_model: str = "facebook/wav2vec2-conformer-rel-pos-large",
    save_artifacts: bool = True
):
    """
    Single-run pipeline (backwards compatible). Behaves similarly to old speech_commands.run_pipeline
    but reuses precompute/train_sae/single_forget_run structure for parity with voxceleb.py.
    """
    # Precompute embeddings & whitener
    precomp = precompute_embeddings_and_whitener(
        manifest_path=data_root,
        embeddings_path=embeddings_path,
        dev_frac_per_cmd=dev_frac_per_cmd,
        manifest_seed=seed,
        encoder=encoder,
        device=device,
        wav2vec2_model=wav2vec2_model,
        hubert_model=hubert_model,
        conformer_model=conformer_model
    )

    # For single-run: we expect forget_commands provided (list or single)
    if forget_commands is None:
        raise ValueError("run_pipeline requires forget_commands for single-run usage")
    if isinstance(forget_commands, str):
        forget_commands = [forget_commands]

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
    all_spks = sorted(list(set(labels)))
    spk2id = {s: i for i, s in enumerate(all_spks)}
    X_cls_train = E_all_norm[mask_dev]
    y_cls_train = [spk2id[s] for i, s in enumerate(labels) if mask_dev[i]]
    X_cls_val = E_all_norm[mask_test]
    y_cls_val = [spk2id[s] for i, s in enumerate(labels) if mask_test[i]]
    clf = train_classifier(X_cls_train, y_cls_train, X_cls_val, y_cls_val, num_classes=len(all_spks), device=device, epochs=20, batch_size=512, lr=1e-3)

    # Run per-forget_command (if multiple provided, run sequentially and save separate artifact dirs)
    all_results = {}
    for cmd in forget_commands:
        run_save_dir = os.path.join(save_dir, f"seed{seed}_forget_{str(cmd)}")
        os.makedirs(run_save_dir, exist_ok=True)
        results, editing_time = single_forget_run(
            precomp=precomp, sae=sae, dec_w=dec_w, A_all=A_all, clf=clf,
            seed=seed, forget_spk=cmd,
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
        all_results[cmd] = results

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/data/veer/testing/new/datasets/speech_commands", help="SpeechCommands root containing command directories")
    parser.add_argument("--save_dir", type=str, default="./out_speech_commands")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--split_seed", type=int, default=1337, help="Seed used to build the dev/test splits (used once at precompute)")
    parser.add_argument("--retain_frac", type=float, default=0.7)
    parser.add_argument("--dev_frac_per_cmd", type=float, default=0.7)
    parser.add_argument("--sae_multiplier", type=float, default=10.0)
    parser.add_argument("--sae_variant", type=str, default="topk", choices=["topk", "standard"], help="SAE variant: topk or standard")
    parser.add_argument("--sae_topk", type=int, default=128, help="Top-K value for topk variant (ignored for standard)")
    parser.add_argument("--lambda_sparsity", type=float, default=0.0, help="L1 sparsity penalty for standard SAE variant")
    parser.add_argument("--sae_epochs", type=int, default=100)
    parser.add_argument("--build_U_top", type=int, default=200)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--no_leace", action="store_true")
    parser.add_argument("--leace_ridge", type=float, default=1e-6)
    parser.add_argument("--noise_sigma", type=float, default=0.02)
    parser.add_argument("--forget_commands", nargs="*", default=["yes"], help="Commands to forget (e.g. 'yes' 'no' 'up')")
    parser.add_argument("--encoder", type=str, default="ecapa", help="Encoder to use - 'ecapa', 'wav2vec2', 'hubert', 'conformer'")
    parser.add_argument("--conformer_model", type=str, default="facebook/wav2vec2-conformer-rel-pos-large")
    parser.add_argument("--embeddings_path", type=str, default=None)
    # Multi-run flags (same style as voxceleb)
    parser.add_argument("--run_all", action="store_true", help="Run for all commands (one-forget-per-run) across seeds; optimized to reuse extractor/SAE per-seed")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1337, 1338, 1339], help="Seeds to use for multi-run (space separated)")
    parser.add_argument("--classifier_epochs", type=int, default=20, help="Epochs to train classifier (used in multi-run per-seed classifier training)")

    args = parser.parse_args()

    if args.run_all:
        # run optimized multi-run over all commands in dataset
        run_all_forget_speakers(args)
    else:
        # Single-run (backwards-compatible)
        run_pipeline(
            data_root=args.data_root, save_dir=args.save_dir, device=args.device, seed=args.seed,
            retain_frac=args.retain_frac, dev_frac_per_cmd=args.dev_frac_per_cmd,
            sae_multiplier=args.sae_multiplier, sae_variant=args.sae_variant, sae_topk=args.sae_topk, lambda_sparsity=args.lambda_sparsity, sae_epochs=args.sae_epochs,
            build_U_top=args.build_U_top, beta=args.beta, leace=not args.no_leace, leace_ridge=args.leace_ridge,
            noise_sigma=args.noise_sigma, forget_commands=args.forget_commands, encoder=args.encoder, embeddings_path=args.embeddings_path,
            wav2vec2_model=getattr(args, 'wav2vec2_model', "facebook/wav2vec2-base"),
            hubert_model=getattr(args, 'hubert_model', "facebook/hubert-base-ls960"),
            conformer_model=args.conformer_model
        )
