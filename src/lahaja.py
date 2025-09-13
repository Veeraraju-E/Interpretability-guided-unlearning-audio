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
from datasets import load_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm

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

def _extract_audio_info_from_item(item: Dict, index: int = 0) -> Optional[Dict]:
    """
    Enhanced helper to extract audio information from a huggingface dataset item.
    Returns a dict with 'array', 'sampling_rate', and 'path' (if available).
    For streaming datasets, the audio data is directly available.
    """

    audio_info = {}

    # Case 1: The item contains the audio data within a dictionary under a standard key.
    # This handles structures like {'audio': {'path': ..., 'array': ..., 'sampling_rate': ...}}
    # or {'audio_filepath': {'path': ..., 'array': ..., 'sampling_rate': ...}}
    potential_audio_dict_keys = ['audio', 'audio_filepath']
    for key in potential_audio_dict_keys:
        if key in item and isinstance(item[key], dict):
            audio_dict = item[key]
            if 'array' in audio_dict and 'sampling_rate' in audio_dict:
                audio_info['array'] = audio_dict['array']
                audio_info['sampling_rate'] = audio_dict['sampling_rate']
                audio_info['path'] = audio_dict.get('path', f"audio_{index}")
                return audio_info

    # Case 2: Top-level keys for array and sampling_rate exist.
    if 'array' in item and 'sampling_rate' in item:
        audio_info['array'] = item['array']
        audio_info['sampling_rate'] = item['sampling_rate']
        audio_info['path'] = item.get('path', f"audio_{index}")
        return audio_info

    # Case 3: Only a path is available, which might be a string or inside a dict.
    path_keys = ('path', 'file', 'audio_filepath', 'audio_path', 'audio')
    for key in path_keys:
        if key in item:
            val = item[key]
            path_to_use = None
            if isinstance(val, str):
                path_to_use = val
            elif isinstance(val, dict):
                # Look for a 'path' or 'file' key inside the dict
                if 'path' in val and isinstance(val['path'], str):
                    path_to_use = val['path']
                elif 'file' in val and isinstance(val['file'], str):
                    path_to_use = val['file']

            if path_to_use:
                audio_info['path'] = path_to_use
                return audio_info

    return None


def load_lahaja_dataset(split: str = "test", streaming: bool = False) -> List[Dict]:
    """
    Load Lahaja dataset with improved audio handling.
    If streaming=True, processes the dataset in streaming mode which often works better
    for datasets with audio files.
    """
    try:
        # First try with streaming=True as it often works better for audio datasets
        print(f"Loading Lahaja dataset split='{split}' with streaming={streaming}...")
        ds = load_dataset("ai4bharat/Lahaja", split=split, streaming=streaming)

        manifest = []
        for i, item in enumerate(ds):
            # Extract audio information
            audio_info = _extract_audio_info_from_item(item, i)
            if audio_info is None:
                raise RuntimeError(
                    f"Could not extract audio information for dataset entry index {i}. "
                    f"Available keys: {list(item.keys())}. "
                    f"Please inspect the dataset structure manually."
                )

            # Build manifest entry
            entry = {
                'path': audio_info.get('path', f"lahaja_audio_{i}"),
                'speaker': item.get('sp_id') or item.get('spk_id') or item.get('speaker') or item.get('speaker_id'),
                'segment_id': f"lahaja_{i}",
                'text': item.get('text') or item.get('transcription'),
                'lang': item.get('lang') or item.get('language'),
                'gender': item.get('gender'),
                'age_group': item.get('age_group') or item.get('age'),
                'native_state': item.get('native_state') or item.get('state'),
                'duration': item.get('duration')
            }

            # Add audio data if available (for direct processing)
            if 'array' in audio_info:
                entry['audio_array'] = audio_info['array']
                entry['sampling_rate'] = audio_info['sampling_rate']

            manifest.append(entry)

        print(f"Successfully loaded {len(manifest)} entries from Lahaja dataset")

        # Print sample entry for verification
        if manifest:
            print("Sample manifest entry:")
            sample = manifest[0].copy()
            # Don't print the audio array as it's too long
            if 'audio_array' in sample:
                sample['audio_array'] = f"<array of length {len(sample['audio_array'])}>"
            print(json.dumps(sample, indent=2))

        return manifest

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying alternative approach...")

        # Fallback: try without streaming
        if streaming:
            return load_lahaja_dataset(split=split, streaming=False)
        else:
            raise RuntimeError(f"Failed to load Lahaja dataset: {e}")


def _ensure_manifest_paths_are_strings(manifest: List[Dict]) -> None:
    """
    Mutates manifest in-place ensuring each entry['path'] is a string path.
    If it's a dict, attempt to extract a string path from it (common HF audio dict).
    Also handles cases where audio data is embedded directly.
    """
    for i, entry in enumerate(manifest):
        p = entry.get('path')
        if isinstance(p, str):
            continue

        if isinstance(p, dict):
            # prefer typical keys
            for k in ('path', 'file', 'audio_filepath'):
                if k in p and isinstance(p[k], str):
                    entry['path'] = p[k]
                    break
            else:
                # fallback: first string value
                for v in p.values():
                    if isinstance(v, str):
                        entry['path'] = v
                        break
                else:
                    # If no path found but we have audio data, create a synthetic path
                    if 'audio_array' in entry:
                        entry['path'] = f"synthetic_audio_{i}"
                    else:
                        raise RuntimeError(f"Manifest entry at index {i} has non-string 'path' dict and no fallback string found: {p}")
        elif p is None:
            # If path is None but we have audio data, create a synthetic path
            if 'audio_array' in entry:
                entry['path'] = f"synthetic_audio_{i}"
            else:
                raise RuntimeError(f"Manifest entry at index {i} missing 'path' field and no audio data")
        else:
            # convert other types to str if reasonable (e.g., PathLike)
            try:
                entry['path'] = str(p)
            except Exception:
                if 'audio_array' in entry:
                    entry['path'] = f"synthetic_audio_{i}"
                else:
                    raise RuntimeError(f"Manifest entry at index {i} has unconvertible 'path': {p}")


def build_manifest(split: str = "test", embeddings_path: str = None, dev_frac_per_spk: float = 0.75, seed: int = 1337) -> List[Dict]:
    if embeddings_path is not None:
        data = np.load(embeddings_path, allow_pickle=True)
        labels = data['labels'].tolist() if 'labels' in data else (data['y'].tolist() if 'y' in data else None)
        ids = data['segment_ids'].tolist() if 'segment_ids' in data else (data['ids'].tolist() if 'ids' in data else None)
        if labels is None:
            raise RuntimeError("embeddings .npz must contain 'labels' or 'y' array when embeddings_path is used")
        raw_manifest = [{'path': f'dummy_{i}', 'speaker': labels[i], 'segment_id': ids[i] if ids is not None else None} for i in range(len(labels))]
    else:
        # Try streaming first, then fallback to regular loading
        try:
            raw_manifest = load_lahaja_dataset(split, streaming=True)
        except:
            print("Streaming failed, trying regular loading...")
            raw_manifest = load_lahaja_dataset(split, streaming=False)

    # Filter out entries without speaker information
    valid_manifest = [m for m in raw_manifest if m.get('speaker') is not None]
    if len(valid_manifest) == 0:
        raise RuntimeError("No entries with speaker information found in dataset")

    print(f"Found {len(valid_manifest)} entries with speaker information out of {len(raw_manifest)} total entries")

    by_speaker = {}
    for i, m in enumerate(valid_manifest):
        speaker = m['speaker']
        by_speaker.setdefault(speaker, []).append(i)

    print(f"Found {len(by_speaker)} unique speakers: {list(by_speaker.keys())[:10]}..." +
          ("" if len(by_speaker) <= 10 else f" (and {len(by_speaker) - 10} more)"))

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
            entry = valid_manifest[idx].copy()
            entry['split'] = 'dev'
            manifest.append(entry)
        for idx in test_indices:
            entry = valid_manifest[idx].copy()
            entry['split'] = 'test'
            manifest.append(entry)

    if len(manifest) == 0:
        raise RuntimeError("No entries found in manifest")

    print(f"Built manifest with {len(manifest)} entries total")
    return manifest


# ----------------------------
# Precompute / shared steps
# ----------------------------
def precompute_embeddings_and_whitener(dataset_split: str, embeddings_path: Optional[str], dev_frac_per_spk: float,
                                       manifest_seed: int, encoder: str, device: str,
                                       wav2vec2_model: str, hubert_model: str, conformer_model: str):
    """
    Build manifest (dev/test splits), extract/load embeddings aligned to manifest order,
    fit a whitener on *dev* samples (using all dev samples), return a dict of precomputed objects.
    This is run exactly once for --run_all to avoid repeating extractor work.
    """
    print("[PRECOMP] Building manifest and splits (once)...")
    entries = build_manifest(dataset_split, embeddings_path, dev_frac_per_spk, manifest_seed)
    
    # Ensure paths are strings
    _ensure_manifest_paths_are_strings(entries)
    
    by = {}
    for e in entries:
        by.setdefault(e["speaker"], []).append(e)
    spks = sorted(by.keys())

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
                    raise RuntimeError(f"segment_id {sid} not found in embeddings file keys")
                E_all[i] = embs[id_to_idx[sid_s]]
            L_all = [m['speaker'] for m in manifest_ordered]
            P_all = [m['path'] for m in manifest_ordered]
        else:
            print("  Warning: embeddings file lacks 'segment_ids' - assuming ordering already matches manifest")
            if embs.shape[0] != len(manifest_ordered):
                raise RuntimeError("Embeddings count does not match manifest length and no segment_ids provided for alignment")
            E_all = embs.astype(np.float32)
            L_all = [m['speaker'] for m in manifest_ordered]
            P_all = [m['path'] for m in manifest_ordered]
        print(f"  Loaded {E_all.shape[0]} embeddings, shape {E_all.shape}")
    else:
        # Extract embeddings using Lahaja-specific audio processing
        print("[PRECOMP] Extracting embeddings from in-memory audio arrays (this may take time; done once)...")
        
        wav2vec2_model_name = "theainerd/Wav2Vec2-large-xlsr-hindi"
        processor = Wav2Vec2Processor.from_pretrained(wav2vec2_model_name)
        model = Wav2Vec2Model.from_pretrained(wav2vec2_model_name).to(device)

        E_all_list, L_all, P_all = [], [], []
        target_sr = 16000

        with torch.no_grad():
            for entry in tqdm(manifest_ordered, desc=f"Extracting {wav2vec2_model_name}"):
                if 'audio_array' not in entry:
                    print(f"Warning: Skipping entry {entry.get('segment_id')} as it lacks 'audio_array'.")
                    continue

                audio_array = entry['audio_array']
                sampling_rate = entry['sampling_rate']

                if sampling_rate != target_sr:
                    # This is a placeholder for resampling if needed.
                    # For simplicity, we assume SR is correct. If not, you'd add torchaudio.transforms here.
                    raise RuntimeError(f"Unexpected sample rate {sampling_rate} for {entry['path']}. Expected {target_sr}.")

                # Process the raw audio array
                inputs = processor(audio_array, sampling_rate=target_sr, return_tensors="pt", padding=True)
                input_values = inputs.input_values.to(device)

                # Get the embeddings
                outputs = model(input_values)
                # Use the mean of the last hidden state as the utterance-level embedding
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

                E_all_list.append(embedding)
                L_all.append(entry['speaker'])
                P_all.append(entry['path'])

        E_all = np.array(E_all_list)

    labels = L_all
    sets = [f"?:{m['split']}" for m in manifest_ordered]  # placeholder sets; specific forget assignment is per-run

    # dev/test masks by split only (independent of forget speaker)
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

    # encode all activations once for this seed (reused across speakers)
    print(f"[SAE] Encoding SAE activations for all samples (seed {seed})...")
    A_all = sae_encode_batch(sae, Z_all, device=device, batch=8192)

    # get decoder weights for U building later
    dec_w = sae.dec.weight.detach().cpu().numpy()

    return sae, sae_training_time, A_all, dec_w


# ----------------------------
# Per-speaker run (uses precomputed and trained SAE)
# ----------------------------
def single_forget_run(precomp, sae, dec_w, A_all, clf, seed: int, forget_spk: str, sae_params: Dict, edit_params: Dict, compute_ms: bool = True, save_artifacts: bool = True, save_dir: Optional[str] = None):
    """
    Run the per-forget speaker steps (feature stats, select features, build U, LEACE,
    SAUCE gating, editing, evaluation). Returns `results` (dict), sae_training_time (maybe None),
    editing_time_seconds.
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
    sets = [f"{'forget' if m['speaker'] == forget_spk else 'retain'}:{m['split']}" for m in manifest]

    M_ret_dev = np.array([s == 'retain:dev' for s in sets], dtype=bool)
    M_fgt_dev = np.array([s == 'forget:dev' for s in sets], dtype=bool)
    M_ret_test = np.array([s == 'retain:test' for s in sets], dtype=bool)
    M_fgt_test = np.array([s == 'forget:test' for s in sets], dtype=bool)

    # Feature mining (depends on forget speaker)
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

    print("[FEATURE] Computing feature stats and selecting features for this forget speaker...")
    stats = feature_stats(A_fgt, A_ret)
    feat_idx_global = select_features_from_stats(stats, top_k=sae_params.get("build_U_top", 200), fdr=0.05)
    feat_idx = feat_idx_global

    # Build U subspace from decoder using selected features
    print("[U] Building U subspace for this forget speaker...")
    dec_pruned = prune_and_normalize_decoder(dec_w, prune_threshold=1e-6)
    U = build_U_from_decoder(dec_w, feat_idx)
    # fallback
    if U is None:
        U = np.zeros((dec_w.shape[1], 0))

    # LEACE projector (depends on forget speaker)
    P_r = np.eye(U.shape[1]) if U.shape[1] > 0 else np.eye(0)
    mu_r = None
    if edit_params.get("leace", True):
        print("[LEACE] Computing LEACE projector (per-forget-run)...")
        # Note: we use split==dev indices for Y vector positions (dev split independent of set)
        dev_indices = np.where(mask_dev_global)[0]
        # Build Y_dev in order of dev_indices
        Y_dev = np.array([1 if manifest[i]['speaker'] == forget_spk else 0 for i in dev_indices], dtype=int).reshape(-1, 1)
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
    print("[EVAL] Computing EERs and per-class metrics for this forget speaker...")
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
    print("[EVAL] Classifier accuracy (retain vs forget) for this forget speaker...")
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
                            labels=np.array(labels), sets=np.array(sets), manifest=np.array(manifest), forget_spks=np.array([forget_spk]))
        torch.save(sae.state_dict(), os.path.join(save_dir, "sae_state_dict.pt"))
        np.savez_compressed(os.path.join(save_dir, "whitener.npz"), mu=mu, W=W)

    return results, editing_time


# ----------------------------
# Multi-run orchestration
# ----------------------------
def run_all_forget_speakers(args):
    """
    Orchestrates multi-run optimized execution:
      - Precompute embeddings/whitener once
      - For each seed: train SAE once and classifier once
      - For each speaker: run editing/evaluation (no per-run artifacts saved)
      - Write a single consolidated CSV with one row per (seed, forget_speaker)
    """
    base_save_dir = args.save_dir
    os.makedirs(base_save_dir, exist_ok=True)
    seeds = args.seeds if args.seeds is not None else [1337, 1338, 1339]
    manifest_seed = args.split_seed if hasattr(args, "split_seed") else args.seed

    # 1) Precompute embeddings & whitener once (manifest built once)
    precomp = precompute_embeddings_and_whitener(
        dataset_split=args.dataset_split,
        embeddings_path=args.embeddings_path,
        dev_frac_per_spk=args.dev_frac_per_spk,
        manifest_seed=manifest_seed,
        encoder=args.encoder if hasattr(args, "encoder") else "wav2vec2",
        device=args.device,
        wav2vec2_model=args.wav2vec2_model if hasattr(args, "wav2vec2_model") else "theainerd/Wav2Vec2-large-xlsr-hindi",
        hubert_model=args.hubert_model if hasattr(args, "hubert_model") else "facebook/hubert-base-ls960",
        conformer_model=args.conformer_model if hasattr(args, "conformer_model") else "facebook/wav2vec2-conformer-rel-pos-large"
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
                all_spks = sorted(list(set(labels)))
                spk2id = {s: i for i, s in enumerate(all_spks)}
                X_cls_train = E_all_norm[mask_dev]
                y_cls_train = [spk2id[s] for i, s in enumerate(labels) if mask_dev[i]]
                X_cls_val = E_all_norm[mask_test]
                y_cls_val = [spk2id[s] for i, s in enumerate(labels) if mask_test[i]]
                # reuse existing classifier training function
                clf = train_classifier(X_cls_train, y_cls_train, X_cls_val, y_cls_val,
                                       num_classes=len(all_spks), device=args.device, epochs=args.classifier_epochs if hasattr(args, "classifier_epochs") else 20,
                                       batch_size=512, lr=1e-3)

                # now iterate speakers (no per-run saving)
                for spk in speakers:
                    print(f"\n[MULTI] seed={seed} forget_spk={spk} ...")
                    try:
                        results, editing_time = single_forget_run(
                            precomp=precomp, sae=sae, dec_w=dec_w, A_all=A_all, clf=clf,
                            seed=seed, forget_spk=spk,
                            sae_params={"device": args.device, "build_U_top": args.build_U_top},
                            edit_params={"beta": args.beta, "leace": not args.no_leace, "leace_ridge": args.leace_ridge, "noise_sigma": args.noise_sigma},
                            compute_ms=False,  # skip MS in multi-run to save time
                            save_artifacts=False,  # do not write per-run artifacts
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
# Backwards-compatible single-run (keeps original semantics)
# ----------------------------
def run_pipeline(
    dataset_split: str = "test", save_dir: str = "./runs/exp", device: str = "cuda", seed: int = 1337,
    retain_frac: float = 0.975, dev_frac_per_spk: float = 0.75, sae_multiplier: float = 4.0,
    sae_variant: str = "topk", sae_topk: int = 128, lambda_sparsity: float = 0.0, sae_epochs: int = 30,
    build_U_top: int = 200, beta: float = 0.0, leace: bool = True, leace_ridge: float = 1e-6,
    noise_sigma: float = 0.02, per_speaker_union: bool = False, forget_spks: Optional[List[str]] = None,
    encoder: str = "wav2vec2", embeddings_path: str = None,
    wav2vec2_model: str = "theainerd/Wav2Vec2-large-xlsr-hindi",
    hubert_model: str = "facebook/hubert-base-ls960",
    conformer_model: str = "facebook/wav2vec2-conformer-rel-pos-large",
    save_artifacts: bool = True
):
    """
    Kept for backwards compatibility and single-run usage. Behaves like original pipeline,
    but can be called with save_artifacts=False to skip writing files.
    """
    # We'll reuse precompute + seed-specific SAE training + single_forget_run
    precomp = precompute_embeddings_and_whitener(
        dataset_split=dataset_split,
        embeddings_path=embeddings_path,
        dev_frac_per_spk=dev_frac_per_spk,
        manifest_seed=seed,
        encoder=encoder,
        device=device,
        wav2vec2_model=wav2vec2_model,
        hubert_model=hubert_model,
        conformer_model=conformer_model
    )

    # For single-run: we expect forget_spks provided (list or single)
    if forget_spks is None:
        raise ValueError("run_pipeline requires forget_spks for single-run usage")
    if isinstance(forget_spks, str):
        forget_spks = [forget_spks]

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

    # Run per-forget_spk (if multiple provided, run sequentially and save separate artifact dirs)
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
        # Add timing fields
        results["sae_training_time_seconds"] = float(sae_train_time)
        results["editing_time_seconds"] = float(editing_time)
        results["total_sae_edit_time_seconds"] = float(sae_train_time + editing_time)
        # Save summary JSON if requested
        if save_artifacts:
            stats_path = os.path.join(run_save_dir, "verification_eer_summary.json")
            with open(stats_path, "w") as fh:
                json.dump(results, fh, indent=2)
        all_results[spk] = results

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_split", type=str, default="test", choices=["test", "validation", "train"])
    parser.add_argument("--save_dir", type=str, default="./out_lahaja")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--split_seed", type=int, default=1337, help="Seed used to build the dev/test splits (used once at precompute)")
    parser.add_argument("--retain_frac", type=float, default=0.7)
    parser.add_argument("--dev_frac_per_spk", type=float, default=0.7)
    parser.add_argument("--sae_multiplier", type=float, default=10.0)
    parser.add_argument("--sae_variant", type=str, default="topk", choices=["topk", "standard"], help="SAE variant: topk or standard")
    parser.add_argument("--sae_topk", type=int, default=128, help="Top-K value for topk variant (ignored for standard)")
    parser.add_argument("--lambda_sparsity", type=float, default=0.0, help="L1 sparsity penalty for standard SAE variant")
    parser.add_argument("--sae_epochs", type=int, default=150)
    parser.add_argument("--build_U_top", type=int, default=200)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--no_leace", action="store_true")
    parser.add_argument("--leace_ridge", type=float, default=1e-6)
    parser.add_argument("--noise_sigma", type=float, default=0.02)
    parser.add_argument("--forget_spks", nargs="*", default=None)
    parser.add_argument("--encoder", type=str, default="wav2vec2")
    parser.add_argument("--wav2vec2_model", type=str, default="theainerd/Wav2Vec2-large-xlsr-hindi")
    parser.add_argument("--hubert_model", type=str, default="facebook/hubert-base-ls960")
    parser.add_argument("--conformer_model", type=str, default="facebook/wav2vec2-conformer-rel-pos-large")
    parser.add_argument("--embeddings_path", type=str, default=None)
    # Multi-run flags
    parser.add_argument("--run_all", action="store_true", help="Run for all speakers (one-forget-per-run) across seeds; optimized to reuse extractor/SAE per-seed")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1337, 1338, 1339], help="Seeds to use for multi-run (space separated)")
    parser.add_argument("--classifier_epochs", type=int, default=20, help="Epochs to train classifier (used in multi-run per-seed classifier training)")

    args = parser.parse_args()

    if args.run_all:
        run_all_forget_speakers(args)
    else:
        # Single-run (backwards-compatible)
        run_pipeline(
            dataset_split=args.dataset_split, save_dir=args.save_dir, device=args.device, seed=args.seed,
            retain_frac=args.retain_frac, dev_frac_per_spk=args.dev_frac_per_spk,
            sae_multiplier=args.sae_multiplier, sae_variant=args.sae_variant, sae_topk=args.sae_topk, lambda_sparsity=args.lambda_sparsity, sae_epochs=args.sae_epochs,
            build_U_top=args.build_U_top, beta=args.beta, leace=not args.no_leace, leace_ridge=args.leace_ridge,
            noise_sigma=args.noise_sigma, forget_spks=args.forget_spks,
            encoder=args.encoder, embeddings_path=args.embeddings_path,
            wav2vec2_model=args.wav2vec2_model,
            hubert_model=args.hubert_model,
            conformer_model=args.conformer_model
        )