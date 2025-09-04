from __future__ import annotations
import os, time, json, random
from pathlib import Path
from typing import List, Dict, Optional
import argparse
import numpy as np
import torch

from common import (
    seed_all, human_time, extract_embeddings, fit_whitener, apply_whitener,
    invert_whitener, l2_normalize, TopKSAE, train_sae, sae_encode_batch,
    prune_and_normalize_decoder, feature_stats, select_features_from_stats, build_U_from_decoder,
    leace_projector_in_U, apply_subspace_leace, apply_sauce_gating, inject_noise_in_subspace,
    compute_scores, eer_from_scores, build_verification_trials,
    train_classifier, evaluate_classifier, per_class_eers, avg_eers, safe_linear_probe
)
from validation_checks import run_validation_checks, create_all_plots


# -------------------- Manifest --------------------
def build_manifest(data_root: str, dev_frac_per_cmd: float = 0.75, seed: int = 1337, 
                  allowed_exts: List[str] = [".wav"]) -> List[Dict]:
    """
    Build manifest entries for Speech Commands dataset folder.
    Expects a flat directory structure with command directories.
    Returns list of dicts: {"path": p, "speaker": spk, "command": command_label, "split": "dev"/"test"}
    """
    data_root = os.path.expanduser(data_root)
    if not os.path.isdir(data_root):
        raise RuntimeError(f"Speech Commands root not found or not a directory: {data_root}")

    # First collect all raw entries
    raw_entries = []
    data_root_path = Path(data_root)
    
    # Get command labels
    command_dirs = [d for d in data_root_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if not command_dirs:
        raise RuntimeError(f"No command directories found in {data_root}")
    
    print(f"Found command directories: {[d.name for d in command_dirs]}")
    
    for command_dir in command_dirs:
        command_label = command_dir.name
        
        for ext in allowed_exts:
            audio_files = list(command_dir.glob(f"*{ext}")) + list(command_dir.glob(f"*{ext.upper()}"))
            
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
                    "command": command_label
                })
    
    if len(raw_entries) == 0:
        raise RuntimeError(f"No Speech Commands audio files found under {data_root} with extensions {allowed_exts}")

    print(f"Found {len(raw_entries)} audio files across {len(command_dirs)} command categories.")

    # Group by command and create splits
    by_command = {}
    for entry in raw_entries:
        command = entry['command']
        by_command.setdefault(command, []).append(entry)

    rng = random.Random(seed)
    manifest = []
    for command, entries in by_command.items():
        rng.shuffle(entries)
        n_dev = max(1, int(len(entries) * dev_frac_per_cmd))
        # ensure that if command has >1 utts, we leave at least 1 for test
        if len(entries) > 1 and n_dev >= len(entries): 
            n_dev = len(entries) - 1
        
        dev_entries = entries[:n_dev]
        test_entries = entries[n_dev:]
        
        for entry in dev_entries:
            entry_copy = entry.copy()
            entry_copy['split'] = 'dev'
            manifest.append(entry_copy)
        for entry in test_entries:
            entry_copy = entry.copy()
            entry_copy['split'] = 'test'
            manifest.append(entry_copy)
    
    return manifest


# -------------------- Orchestration --------------------
def run_pipeline(
    data_root: str, save_dir: str = "./runs/exp", device: str = "cuda", seed: int = 1337,
    retain_frac: float = 0.975, dev_frac_per_cmd: float = 0.75, sae_hidden_mult: float = 4.0,
    sae_topk: int = 128, sae_epochs: int = 30,
    build_U_top: int = 200, beta: float = 0.0, leace: bool = True, leace_ridge: float = 1e-6, 
    noise_sigma: float = 0.02, per_speaker_union: bool = False, forget_commands: Optional[List[str]] = None,
    encoder: str = "ecapa", wav2vec2_model: str = "facebook/wav2vec2-base", 
    hubert_model: str = "facebook/hubert-base-ls960",
    conformer_model: str = "facebook/wav2vec2-conformer-rel-pos-large"
):
    """
    Main pipeline orchestration - aligned with voxceleb.py framework.
    """
    seed_all(seed)
    os.makedirs(save_dir, exist_ok=True)
    t0_all = time.time()

    # Default forget commands if none specified
    if forget_commands is None:
        forget_commands = ["yes"]
    elif isinstance(forget_commands, str):
        forget_commands = [forget_commands]
    else:
        forget_commands = list(forget_commands)

    # 1) manifest + splits
    print("[1/12] Building manifest and splits...")
    entries = build_manifest(data_root, dev_frac_per_cmd, seed)

    # group utterances by command (use full entries ordering)
    by = {}
    for e in entries:
        by.setdefault(e["command"], []).append(e)
    commands = sorted(by.keys())
    # test_counts = {c: sum(1 for x in by[c] if x.get('split') == 'test') for c in commands}

    for cmd in forget_commands:
        if cmd not in commands:
            raise ValueError(f"Requested forget command '{cmd}' not found in dataset commands: {commands}")

    print(f"  FORGETTING COMMANDS ({len(forget_commands)}): {forget_commands}")

    retain_commands = [c for c in commands if c not in forget_commands]

    # add 'set' labels to manifest
    manifest = []
    for entry in entries:
        entry_copy = entry.copy()
        entry_copy['set'] = 'forget' if entry['command'] in forget_commands else 'retain'
        manifest.append(entry_copy)

    dev_manifest = [m for m in manifest if m['split'] == 'dev']
    test_manifest = [m for m in manifest if m['split'] == 'test']
    dev_retain = [m for m in dev_manifest if m['set'] == 'retain']
    dev_forget = [m for m in dev_manifest if m['set'] == 'forget']
    test_retain = [m for m in test_manifest if m['set'] == 'retain']
    test_forget = [m for m in test_manifest if m['set'] == 'forget']
    print(f"  retain_commands {len(retain_commands)} forget_commands {len(forget_commands)}")
    print(f"  dev_retain {len(dev_retain)} dev_forget {len(dev_forget)} | test_retain {len(test_retain)} test_forget {len(test_forget)}")

    # 2) Embedding extraction
    print("[2/12] Extracting embeddings...")
    all_manifest = dev_manifest + test_manifest
    E_all, L_all, P_all = extract_embeddings(all_manifest, encoder=encoder, device=device, wav2vec2_model=wav2vec2_model, hubert_model=hubert_model, conformer_model=conformer_model, label_key='command')
    manifest = list(dev_manifest) + list(test_manifest)
    labels = L_all

    # Create set labels
    sets = [f"{m['set']}:{m['split']}" for m in manifest]
    
    # all masks
    M_ret_dev = np.array([s == 'retain:dev' for s in sets], dtype=bool)
    M_fgt_dev = np.array([s == 'forget:dev' for s in sets], dtype=bool)
    M_ret_test = np.array([s == 'retain:test' for s in sets], dtype=bool)
    M_fgt_test = np.array([s == 'forget:test' for s in sets], dtype=bool)

    # 3) Whitening on retain-dev
    print("[3/12] Fitting whitener on Train-retain (retain-dev)...")
    E_ret_train = E_all[M_ret_dev]
    if E_ret_train.shape[0] < 2: 
        raise RuntimeError("Not enough retain-dev samples to fit whitener")
    mu, W = fit_whitener(E_ret_train)
    Z_all = apply_whitener(E_all, mu, W)

    # 4) SAE training on dev (retain + forget)
    print("[4/12] Training SAE on Z_train (dev union: retain+forget)...")
    Z_train = Z_all[M_ret_dev | M_fgt_dev]
    perm = np.random.permutation(Z_train.shape[0])
    cut = int(0.8 * len(perm))
    Z_sae_train, Z_sae_val = Z_train[perm[:cut]], Z_train[perm[cut:]]
    sae = TopKSAE(input_dim=Z_all.shape[1], hidden_mult=sae_hidden_mult, k=sae_topk)
    sae = train_sae(sae, Z_sae_train, Z_sae_val, epochs=sae_epochs, batch_size=4096, lr=1e-3, device=device)

    # 5) SAE encodes for feature mining
    print("[5/12] Encoding SAE activations for train-retain and train-forget...")
    Z_ret_dev = Z_all[M_ret_dev]
    Z_fgt_dev = Z_all[M_fgt_dev]
    A_ret = sae_encode_batch(sae, Z_ret_dev, device=device, batch=8192)
    A_fgt = sae_encode_batch(sae, Z_fgt_dev, device=device, batch=8192)

    # 6) feature stats & selection
    print("[6/12] Computing feature statistics and selecting features...")
    stats = feature_stats(A_fgt, A_ret)
    feat_idx_global = select_features_from_stats(stats, top_k=build_U_top, fdr=0.05)
    feat_idx = feat_idx_global
    if per_speaker_union:
        print("  computing per-forget-speaker feature unions... (not implemented, using global)")
        feat_idx = feat_idx_global

    # 7) prune & normalize decoder & build U
    print("[7/12] Building U subspace...")
    dec_w = sae.dec.weight.detach().cpu().numpy()   # [d, m]
    dec_pruned = prune_and_normalize_decoder(dec_w, prune_threshold=1e-6)  # returns [d, r]
    U = build_U_from_decoder(dec_w, feat_idx)
    print(f"  U shape: {U.shape}")

    # 8) compute LEACE in U
    P_r = np.eye(U.shape[1]) if U.shape[1] > 0 else np.eye(0)
    mu_r = None
    if leace:
        print("[8/12] Computing LEACE projector in U...")
        mask_dev = np.array([s.endswith(':dev') for s in sets], dtype=bool)
        dev_indices = np.where(mask_dev)[0]
        Y_dev = np.array([1 if manifest[i]['command'] in forget_commands else 0 for i in dev_indices], dtype=int).reshape(-1, 1)
        # safety check: shapes must align
        if Y_dev.shape[0] != Z_all[mask_dev].shape[0]: 
            raise RuntimeError("Mismatch between Y_dev and Z_all[mask_dev] lengths")
        try:
            P_r, mu_r = leace_projector_in_U(Z_all[mask_dev], U, Y_dev, ridge=leace_ridge)
        except Exception as e:
            print(f"  Warning: LEACE computation failed ({e}), falling back to identity projector")
            P_r = np.eye(U.shape[1]) if U.shape[1] > 0 else np.eye(0)
            mu_r = np.zeros((1, U.shape[1]))
    else:
        print("[8/12] Skipping LEACE (identity)")

    # 9) Apply SAUCE gating (optional) then LEACE and optional noise
    print("[9/12] Applying SAUCE gating (beta) and LEACE edit...")
    Z_edit = Z_all.copy()
    if beta > 0.0:
        print("  applying SAUCE gating with beta", beta)
        Z_edit = apply_sauce_gating(sae, Z_all, feat_idx, beta=beta, device=device, batch=8192)
    if leace:
        Z_edit = apply_subspace_leace(Z_edit, U, P_r, mu_r)
    if noise_sigma and noise_sigma > 0:
        Z_edit = inject_noise_in_subspace(Z_edit, U, sigma=noise_sigma)

    # map back to embedding space and l2 normalize
    E_edit = invert_whitener(Z_edit, mu, W)
    E_all_norm, E_edit_norm = l2_normalize(E_all), l2_normalize(E_edit)

    results = {}

    # 9b) Extended Validation Checks
    print("[9b] Running extended validation checks...")
    mask_dev = np.array([s.endswith(':dev') for s in sets], dtype=bool)
    ZU_dev = Z_all[mask_dev] @ U if U.shape[1] > 0 else Z_all[mask_dev][:, :0]
    Y_dev = np.array([1 if manifest[i]['command'] in forget_commands else 0 for i in np.where(mask_dev)[0]], dtype=int).reshape(-1, 1)
    run_validation_checks(
        save_dir=os.path.join(save_dir, "validation"),
        Z_all=Z_all, Z_edit=Z_edit,
        E_all_norm=E_all_norm, E_edit_norm=E_edit_norm,
        ZU_mean=ZU_dev.mean(axis=0, keepdims=True) if U.shape[1] > 0 else (Z_all[mask_dev] @ U).mean(0, keepdims=True),
        U=U, P_r=P_r, mu_r=mu_r, dec_w=dec_w, sae=sae,
        feat_idx=feat_idx, sets=sets, labels=labels,
        mask_dev=mask_dev, mask_test=(M_ret_test | M_fgt_test),
        y_dev_bin=Y_dev, leace_ridge=leace_ridge,
        mu=mu, W=W, device=device
    )

    # 9c) Create all diagnostic plots
    print("[9c] Creating all diagnostic plots...")
    Z_before = Z_all[mask_dev]
    Z_after = Z_edit[mask_dev]
    ZU_before = Z_before @ U if U.shape[1] > 0 else Z_before[:, :0]
    ZU_after = Z_after @ U if U.shape[1] > 0 else Z_after[:, :0]
    create_all_plots(
        Z_before, Z_after, ZU_before, ZU_after,
        labels=[labels[i] for i in np.where(mask_dev)[0]],
        forget_classes=forget_commands,
        decoder_weights=dec_w, feat_idx=feat_idx,
        A_retain=A_ret, A_forget=A_fgt,
        save_dir=os.path.join(save_dir, "validation"), random_state=seed
    )

    # ================= Verification Steps =================
    # 10) Per-class EERs
    print("[10/12] Per-class verification EERs...")
    test_trials, test_labels = build_verification_trials(labels, forget_commands, M_ret_test | M_fgt_test)
    scores_test_before, scores_test_after = compute_scores(E_all_norm, test_trials), compute_scores(E_edit_norm, test_trials)
    eer_before, eer_after = eer_from_scores(scores_test_before, test_labels), eer_from_scores(scores_test_after, test_labels) 
    results["EER_before_raw"], results["EER_after_raw"] = eer_before, eer_after
    print(f"  Global EER: before {eer_before:.4f}, after {eer_after:.4f}")

    per_class_test = per_class_eers(E_all_norm, E_edit_norm, labels, mask=M_ret_test | M_fgt_test, rng_seed=seed)
    per_class_ret = {c: vals for c, vals in per_class_test.items() if c in retain_commands}
    per_class_fgt = {c: vals for c, vals in per_class_test.items() if c in forget_commands}

    retain_avg_before, retain_avg_after = avg_eers(per_class_ret)
    forget_avg_before, forget_avg_after = avg_eers(per_class_fgt)
    results.update({
        "retain_EER_before_avg": retain_avg_before,
        "retain_EER_after_avg": retain_avg_after,
        "forget_EER_before_avg": forget_avg_before,
        "forget_EER_after_avg": forget_avg_after,
    })

    # 11) Linear Probing
    print("[11/12] Linear probing evaluation...")
    split_mask = {"dev": M_ret_dev | M_fgt_dev, "test": M_ret_test | M_fgt_test}
    acc_before, auc_before = safe_linear_probe(Z_all, labels, U, split_mask, forget_classes=forget_commands)
    acc_after, auc_after = safe_linear_probe(Z_edit, labels, U, split_mask, forget_classes=forget_commands)
    results.update({
        "linear_probe_acc_before": acc_before,
        "linear_probe_acc_after": acc_after,
        "linear_probe_auc_before": auc_before,
        "linear_probe_auc_after": auc_after,
    })

    print(f'  Linear Probing accuracies and AUC, before: ({acc_before:.4f}, {auc_before:.4f}), after: ({acc_after:.4f}, {auc_after:.4f})')

    # Print summary
    print("\n--- Average EER summary ---")
    print(f"  Retain avg EER: before {retain_avg_before:.4f}, after {retain_avg_after:.4f}")
    print(f"  Forget avg EER: before {forget_avg_before:.4f}, after {forget_avg_after:.4f}")

    stats_path = os.path.join(save_dir, "verification_eer_summary.json")
    with open(stats_path, "w") as fh: json.dump(results, fh, indent=2)

    # 12) Classifier
    print("[12/12] Command classification accuracy (ResNet) ...")
    all_commands = sorted(list(set(labels)))
    cmd2id = {c: i for i, c in enumerate(all_commands)}

    X_cls_train = E_all_norm[M_ret_dev | M_fgt_dev]
    y_cls_train = [cmd2id[c] for i, c in enumerate(labels) if (M_ret_dev | M_fgt_dev)[i]]
    X_cls_val = E_all_norm[M_ret_test | M_fgt_test]
    y_cls_val = [cmd2id[c] for i, c in enumerate(labels) if (M_ret_test | M_fgt_test)[i]]

    clf = train_classifier(X_cls_train, y_cls_train, X_cls_val, y_cls_val, num_classes=len(all_commands), 
                          device=device, epochs=20, batch_size=512, lr=1e-3)

    # Evaluate retain/forget sets before and after
    E_ret_test = E_all_norm[M_ret_test]
    L_ret_test = [cmd2id[c] for c in np.array(labels)[M_ret_test]]
    E_fgt_test = E_all_norm[M_fgt_test]
    L_fgt_test = [cmd2id[c] for c in np.array(labels)[M_fgt_test]]
    E_ret_test_edit = E_edit_norm[M_ret_test]
    E_fgt_test_edit = E_edit_norm[M_fgt_test]

    acc_ret_before = evaluate_classifier(clf, E_ret_test, L_ret_test, device=device)
    acc_fgt_before = evaluate_classifier(clf, E_fgt_test, L_fgt_test, device=device)
    acc_ret_after = evaluate_classifier(clf, E_ret_test_edit, L_ret_test, device=device)
    acc_fgt_after = evaluate_classifier(clf, E_fgt_test_edit, L_fgt_test, device=device)

    print(f"  Classifier Retain: before {acc_ret_before*100:.2f}% after {acc_ret_after*100:.2f}%")
    print(f"  Classifier Forget: before {acc_fgt_before*100:.2f}% after {acc_fgt_after*100:.2f}%")

    results.update({
        "resnet_acc_retain_before": acc_ret_before,
        "resnet_acc_retain_after": acc_ret_after,
        "resnet_acc_forget_before": acc_fgt_before,
        "resnet_acc_forget_after": acc_fgt_after,
    })

    # Final save with all results
    stats_path = os.path.join(save_dir, "verification_eer_summary.json")
    with open(stats_path, "w") as fh: json.dump(results, fh, indent=2)

    print(f"\nPipeline finished in {human_time(time.time() - t0_all)}. Results saved to {stats_path}")


# -------------------- CLI --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="../datasets/speech_commands", help="SpeechCommands root containing command directories")
    parser.add_argument("--save_dir", default="./outputs_speechcommands", help="store artifacts")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--retain_frac", type=float, default=0.7)
    parser.add_argument("--dev_frac_per_cmd", type=float, default=0.7)
    parser.add_argument("--sae_hidden_mult", type=float, default=10.0)
    parser.add_argument("--sae_topk", type=int, default=128)
    parser.add_argument("--sae_epochs", type=int, default=100)
    parser.add_argument("--build_U_top", type=int, default=200)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--no_leace", action="store_true")
    parser.add_argument("--leace_ridge", type=float, default=1e-6)
    parser.add_argument("--noise_sigma", type=float, default=0.02)
    parser.add_argument("--forget_commands", nargs="*", default=["yes"], help="Commands to forget (e.g. 'yes' 'no' 'up')")
    parser.add_argument("--encoder", type=str, default="ecapa", help="Encoder to use - 'ecapa', 'wav2vec2', 'hubert', 'conformer'")
    parser.add_argument("--conformer_model", type=str, default="facebook/wav2vec2-conformer-rel-pos-large")
    args = parser.parse_args()

    run_pipeline(
        data_root=args.data_root, save_dir=args.save_dir, device=args.device, seed=args.seed,
        retain_frac=args.retain_frac, dev_frac_per_cmd=args.dev_frac_per_cmd,
        sae_hidden_mult=args.sae_hidden_mult, sae_topk=args.sae_topk, sae_epochs=args.sae_epochs,
        build_U_top=args.build_U_top, beta=args.beta, leace=not args.no_leace, leace_ridge=args.leace_ridge, 
        noise_sigma=args.noise_sigma, forget_commands=args.forget_commands, encoder=args.encoder,
        wav2vec2_model=getattr(args, 'wav2vec2_model', "facebook/wav2vec2-base"),
        hubert_model=getattr(args, 'hubert_model', "facebook/hubert-base-ls960"),
        conformer_model=args.conformer_model
    )