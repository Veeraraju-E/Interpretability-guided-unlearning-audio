from __future__ import annotations
import os, time, random
from pathlib import Path
from typing import List, Dict
import numpy as np

from common import (
    seed_all, extract_embeddings, fit_whitener, apply_whitener,
    invert_whitener, TopKSAE, train_sae, sae_encode_batch,
    feature_stats, select_features_from_stats, build_U_from_decoder,
    leace_projector_in_U, apply_subspace_leace, apply_sauce_gating,
    inject_noise_in_subspace, eer_from_scores, compute_scores,
    train_classifier, evaluate_classifier, linear_probe_U, 
    build_verification_trials, per_class_eers, compute_intra_class_cosine_similarities, avg_eers
)

from itertools import combinations

# -------------------- Manifest builder for Speech Commands --------------------
def build_manifest_from_speechcommands(data_root: str, allowed_exts: List[str]=[".wav"]) -> List[Dict]:
    """
    Build manifest entries for Speech Commands dataset folder.
    Expects a flat directory structure
    Returns list of dicts: {"path": p, "speaker": spk, "command": command_label}
    """
    data_root = os.path.expanduser(data_root)
    if not os.path.isdir(data_root):
        raise RuntimeError(f"Speech Commands root not found or not a directory: {data_root}")

    entries = []
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
                if len(parts) >= 1 and len(parts[0]) > 0: spk = parts[0]
                else: spk = "unknown"                
                entries.append({
                    "path": str(audio_file), 
                    "speaker": spk, 
                    "command": command_label
                })
    
    if len(entries) == 0:
        print(f"Warning: No Speech Commands audio files found under {data_root} with extensions {allowed_exts}.")
        raise RuntimeError(f"No Speech Commands audio files found under {data_root}")

    print(f"Found {len(entries)} audio files across {len(command_dirs)} command categories.")
    return entries

# -------------------- Command classifier (ResNet-like) --------------------
def run_pipeline(data_root: str, save_dir: str = "./runs/exp", device: str = "cuda", seed: int = 1337,
                 dev_frac: float = 0.75,
                 sae_hidden_mult: float = 4.0, sae_k: int = 128, sae_epochs: int = 30,
                 build_U_top: int = 200, beta: float = 0.0, leace: bool = True,
                 leace_ridge: float = 1e-6, noise_sigma: float = 0.02,
                 forget_commands: List[str] = ["yes"], encoder: str = "ecapa",
                 wav2vec2_model: str = "facebook/wav2vec2-base",
                 hubert_model: str = "facebook/hubert-base-ls960"):
    """
    Full pipeline that unlearns the specified forget_commands (list of strings, e.g., ["yes", "no"]).
    Returns a dict of summary metrics and saves artifacts to save_dir.
    """
    seed_all(seed)
    os.makedirs(save_dir, exist_ok=True)
    t0_all = time.time()

    print(f"\n===== Running pipeline (encoder={encoder}) =====")
    print(f"Forgetting commands: {forget_commands}")

    # 1) manifest + splits
    print("[1/12] Building manifest and splits...")
    entries = build_manifest_from_speechcommands(data_root)

    # group by command label
    by_cmd = {}
    for e in entries:
        by_cmd.setdefault(e["command"], []).append(e)

    commands = sorted(by_cmd.keys())
    for cmd in forget_commands:
        if cmd not in commands:
            raise ValueError(f"Requested forget command '{cmd}' not found in dataset commands: {commands}. Please check the downloaded data structure or the forget_commands list.")

    manifest = []
    rng = random.Random(seed)
    for c, utts in by_cmd.items():
        rng.shuffle(utts)
        n_dev = max(1, int(len(utts) * dev_frac))
        dev_utts = utts[:n_dev]
        test_utts = utts[n_dev:]
        setlbl = "forget" if c in forget_commands else "retain"
        for u in dev_utts: manifest.append({**u, "set": setlbl, "split": "dev"})
        for u in test_utts: manifest.append({**u, "set": setlbl, "split": "test"})

    dev_manifest = [m for m in manifest if m["split"]=="dev"]
    test_manifest = [m for m in manifest if m["split"]=="test"]
    dev_retain = [m for m in dev_manifest if m["set"]=="retain"]
    dev_forget = [m for m in dev_manifest if m["set"]=="forget"]
    test_retain = [m for m in test_manifest if m["set"]=="retain"]
    test_forget = [m for m in test_manifest if m["set"]=="forget"]

    print(f"  dev_retain {len(dev_retain)} dev_forget {len(dev_forget)} | test_retain {len(test_retain)} test_forget {len(test_forget)}")
    print(f"  Forgetting commands: {forget_commands}")
    print(f"  Retaining commands: {[c for c in commands if c not in forget_commands]}")

    # 2) extract embeddings based on encoder
    E_all, L_all, _ = extract_embeddings(manifest, encoder=encoder, device=device, wav2vec2_model=wav2vec2_model, hubert_model=hubert_model, label_key='command')
    E_all = np.asarray(E_all)
    labels_commands = list(L_all)  # list of command strings per utterance
    sets = [m["split"] for m in manifest]  # 'dev' or 'test'
    is_dev = np.array([s == 'dev' for s in sets], dtype=bool)
    is_test = np.array([s == 'test' for s in sets], dtype=bool)

    # create binary target: 1 if command in forget_commands else 0
    y_binary = np.array([1 if c in forget_commands else 0 for c in labels_commands], dtype=int)

    # masks
    M_forget_dev = is_dev & (y_binary == 1)
    M_forget_test = is_test & (y_binary == 1)
    M_retain_dev = is_dev & (y_binary == 0)
    M_retain_test = is_test & (y_binary == 0)

    print(f"  forget-dev: {M_forget_dev.sum()}, forget-test: {M_forget_test.sum()}, retain-dev: {M_retain_dev.sum()}, retain-test: {M_retain_test.sum()}")

    # 3) Whitener on retain-dev
    print("[3/12] Fitting whitener on retain-dev...")
    E_ret_train = E_all[M_retain_dev]
    if E_ret_train.shape[0] < 2:
        raise RuntimeError("Not enough retain-dev samples to fit whitener")
    mu, W = fit_whitener(E_ret_train)
    Z_all = apply_whitener(E_all, mu, W)

    # 4) SAE training on Z_dev
    print("[4/12] Training SAE on Z_dev...")
    Z_dev = Z_all[is_dev]
    if Z_dev.shape[0] < 2:
        raise RuntimeError("Not enough dev samples to train SAE")

    perm = np.random.permutation(Z_dev.shape[0])
    cut = max(1, int(0.8 * len(perm)))
    Z_sae_train = Z_dev[perm[:cut]]
    Z_sae_val = Z_dev[perm[cut:]] if cut < len(perm) else Z_dev[perm[:1]]
    sae = TopKSAE(input_dim=Z_all.shape[1], hidden_mult=sae_hidden_mult, k=sae_k)
    sae = train_sae(sae, Z_sae_train, Z_sae_val, epochs=sae_epochs, batch_size=min(4096, max(8, Z_sae_train.shape[0])), lr=1e-3, device=device)

    # 5) SAE activations
    print("[5/12] Encoding SAE activations...")
    Z_ret_dev = Z_all[M_retain_dev]
    Z_fgt_dev = Z_all[M_forget_dev]
    A_ret = sae_encode_batch(sae, Z_ret_dev, device=device, batch=8192)
    A_fgt = sae_encode_batch(sae, Z_fgt_dev, device=device, batch=8192) if Z_fgt_dev.shape[0] > 0 else np.zeros((0, sae.m))

    # 6) feature stats & selection
    print("[6/12] Computing feature statistics and selecting discriminative features...")
    stats = feature_stats(A_fgt, A_ret)
    feat_idx_global = select_features_from_stats(stats, top_k=build_U_top, fdr=0.05)
    feat_idx = feat_idx_global

    # 7) build U from decoder atoms
    print("[7/12] Building subspace U from decoder atoms...")
    dec_w = sae.dec.weight.detach().cpu().numpy()
    U = build_U_from_decoder(dec_w, feat_idx)
    print(f"  U shape: {U.shape}")

    # 8) compute LEACE projector in U
    P_r = np.eye(U.shape[1])
    mu_r = np.zeros((1, U.shape[1])) if U.shape[1] > 0 else None
    if leace and U.shape[1] > 0:
        print("[8/12] Computing LEACE projector in U...")
        mask_dev = is_dev
        Y_dev = y_binary[mask_dev].reshape(-1, 1)
        if mask_dev.sum() == 0:
            raise RuntimeError("No dev samples found for LEACE computation.")
        try:
            P_r, mu_r = leace_projector_in_U(Z_all[mask_dev], U, Y_dev, ridge=leace_ridge)
            np.savez(os.path.join(save_dir, "leace_Pr.npz"), U=U, P_r=P_r)
        except Exception as e:
            print("  LEACE computation failed, falling back to identity. Error:", e)
            P_r = np.eye(U.shape[1])
            mu_r = np.zeros((1, U.shape[1])) if U.shape[1] > 0 else None
    else:
        print("[8/12] Skipping LEACE (identity)")

    # 9) Apply SAUCE gating, LEACE, and noise
    print("[9/12] Applying SAUCE gating and LEACE edit...")
    Z_edit = Z_all.copy()
    if beta and beta > 0.0 and len(feat_idx) > 0:
        Z_edit = apply_sauce_gating(sae, Z_all, feat_idx, beta=beta, device=device, batch=8192)
    if leace and U.shape[1] > 0:
        Z_edit = apply_subspace_leace(Z_edit, U, P_r, mu_r)
    if noise_sigma and noise_sigma > 0 and U.shape[1] > 0:
        Z_edit = inject_noise_in_subspace(Z_edit, U, sigma=noise_sigma)

    E_edit = invert_whitener(Z_edit, mu, W)
    E_all_norm, E_edit_norm = E_all / (np.linalg.norm(E_all, axis=1, keepdims=True) + 1e-12), E_edit / (np.linalg.norm(E_edit, axis=1, keepdims=True) + 1e-12)

    # 10) Evaluations
    print("[10/12] Command classification accuracy before/after edits...")
    all_commands = sorted(set(labels_commands))
    cmd2id = {c:i for i,c in enumerate(all_commands)}

    # Train classifier on dev data
    X_cls_train = E_all_norm[is_dev]
    y_cls_train = [cmd2id[c] for c in np.array(labels_commands)[is_dev]]
    X_cls_val = E_all_norm[is_test]
    y_cls_val = [cmd2id[c] for c in np.array(labels_commands)[is_test]]

    if len(y_cls_train) == 0 or len(y_cls_val) == 0:
         print("Warning: Not enough samples for classifier training or validation. Skipping classification evaluation.")
         acc_ret_before = float('nan')
         acc_fgt_before = float('nan')
         acc_ret_after = float('nan')
         acc_fgt_after = float('nan')
         clf = None
    else:
        clf = train_classifier(X_cls_train, y_cls_train, X_cls_val, y_cls_val, num_classes=len(all_commands), device=device, epochs=20, batch_size=512, lr=1e-3)

        # Build test sets
        E_ret_test = E_all_norm[M_retain_test]
        L_ret_test = [cmd2id[c] for c in np.array(labels_commands)[M_retain_test]]
        E_fgt_test = E_all_norm[M_forget_test]
        L_fgt_test = [cmd2id[c] for c in np.array(labels_commands)[M_forget_test]]
        E_ret_test_edit = E_edit_norm[M_retain_test]
        E_fgt_test_edit = E_edit_norm[M_forget_test]

        # Evaluate before edits
        acc_ret_before = evaluate_classifier(clf, E_ret_test, L_ret_test, device=device) if len(L_ret_test) > 0 else float('nan')
        acc_fgt_before = evaluate_classifier(clf, E_fgt_test, L_fgt_test, device=device) if len(L_fgt_test) > 0 else float('nan')

        # Evaluate after edits
        acc_ret_after = evaluate_classifier(clf, E_ret_test_edit, L_ret_test, device=device) if len(L_ret_test) > 0 else float('nan')
        acc_fgt_after = evaluate_classifier(clf, E_fgt_test_edit, L_fgt_test, device=device) if len(L_fgt_test) > 0 else float('nan')

    print(f"  Classifier Retain: before {acc_ret_before*100:.2f}% after {acc_ret_after*100:.2f}%")
    print(f"  Classifier Forget: before {acc_fgt_before*100:.2f}% after {acc_fgt_after*100:.2f}%")

    print("[11/12] Verification evaluation (EER, minDCF, Linear Probe).")
    print(f"  Verification trials: positive pairs within forget commands {forget_commands}, negative pairs across forget/retain")
    results = {}
    
    dev_trials, dev_labels = build_verification_trials(labels_commands, forget_commands, is_dev)
    test_trials, test_labels = build_verification_trials(labels_commands, forget_commands, is_test)

    if len(test_trials) == 0:
        print("Warning: Not enough test trials for verification evaluation. Skipping.")
        results["EER_before_raw"] = float('nan')
        results["EER_after_raw"] = float('nan')
    else:
        # Compute scores before edit
        scores_dev_before = compute_scores(E_all_norm, dev_trials)
        scores_test_before = compute_scores(E_all_norm, test_trials)

        # Compute scores after edit
        scores_dev_after = compute_scores(E_edit_norm, dev_trials)
        scores_test_after = compute_scores(E_edit_norm, test_trials)

        # Compute EER
        if len(np.unique(test_labels)) < 2:
            print("Warning: Not enough positive and negative samples in test trials for EER calculation. EER will be NaN.")
            eer_before = (float('nan'), float('nan'))
            eer_after = (float('nan'), float('nan'))
        else:
            eer_before = eer_from_scores(scores_test_before, test_labels)
            eer_after = eer_from_scores(scores_test_after, test_labels)

        results["EER_before_raw"] = eer_before
        results["EER_after_raw"] = eer_after

    # Linear probe in reduced coordinates U
    split_mask = {"dev": is_dev, "test": is_test}
    if is_dev.sum() == 0 or is_test.sum() == 0:
        print("Warning: Not enough dev or test samples for linear probe. Skipping.")
        acc_before, auc_before = float('nan'), float('nan')
        acc_after, auc_after = float('nan'), float('nan')
    else:
        acc_before, auc_before = linear_probe_U(Z_all, labels_commands, U, split_mask, forget_classes=forget_commands)
        acc_after, auc_after = linear_probe_U(Z_edit, labels_commands, U, split_mask, forget_classes=forget_commands)
        print(f'Linear probe accuracy before: {acc_before, auc_before}, after : {acc_after, auc_after}')

    # Per-class EERs on TEST partition
    print("[12/12] Computing per-class EERs and cosine similarities...")
    per_class_test = per_class_eers(E_all_norm, E_edit_norm, labels_commands, mask=(M_retain_test | M_forget_test), rng_seed=seed)

    # Intra-class cosine similarities on TEST partition
    intra_sim_before = compute_intra_class_cosine_similarities(E_all_norm, labels_commands, mask=(M_retain_test | M_forget_test), rng_seed=seed)
    intra_sim_after = compute_intra_class_cosine_similarities(E_edit_norm, labels_commands, mask=(M_retain_test | M_forget_test), rng_seed=seed)

    # Partition per-class dict into retain vs forget classes
    all_commands_sorted = sorted(set(labels_commands))
    forget_classes = forget_commands
    retain_classes = [c for c in all_commands_sorted if c not in forget_commands]

    per_class_ret = {c: vals for c, vals in per_class_test.items() if c in retain_classes}
    per_class_fgt = {c: vals for c, vals in per_class_test.items() if c in forget_classes}

    retain_avg_before, retain_avg_after = avg_eers(per_class_ret)
    forget_avg_before, forget_avg_after = avg_eers(per_class_fgt)

    results["retain_EER_before_avg"] = retain_avg_before
    results["retain_EER_after_avg"] = retain_avg_after
    results["forget_EER_before_avg"] = forget_avg_before
    results["forget_EER_after_avg"] = forget_avg_after

    # Print summary
    print("\n--- Verification Results ---")
    for k, v in results.items():
        print(f"{k}: {v}")

    print("\nPer-class EERs:")
    for c, (eb, ea) in sorted(per_class_test.items()):
        if c in forget_commands:
            print("  [FORGET COMMAND]")
        print(f"  class {c}: EER_before={eb:.4f}, EER_after={ea:.4f}")

    print("\nPer-class Intra-class Cosine Similarities:")
    for c in sorted(intra_sim_before.keys()):
        sim_before = intra_sim_before.get(c, float('nan'))
        sim_after = intra_sim_after.get(c, float('nan'))
        if c in forget_commands:
            print("  [FORGET COMMAND]")
        print(f"  class {c}: similarity_before={sim_before:.4f}, similarity_after={sim_after:.4f}")

    print("\n--- Average EER summary ---")
    print(f"  Retain avg EER: before {retain_avg_before:.4f}, after {retain_avg_after:.4f}")
    print(f"  Forget avg EER: before {forget_avg_before:.4f}, after {forget_avg_after:.4f}")
    print(f"  Forgot commands: {forget_commands}")
    print(f"  Retained commands: {[c for c in commands if c not in forget_commands]}")
    print("Done. Total time:", time.strftime('%H:%M:%S', time.gmtime(time.time() - t0_all)))

# -------------------- CLI --------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="../datasets/speech_commands", help="SpeechCommands root containing recordings/")
    p.add_argument("--save_dir", default="./outputs_speechcommands", help="where to store artifacts")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--dev_frac", type=float, default=0.9)
    p.add_argument("--sae_hidden_mult", type=float, default=10.0)
    p.add_argument("--sae_k", type=int, default=128)
    p.add_argument("--sae_epochs", type=int, default=100)
    p.add_argument("--build_U_top", type=int, default=200)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--leace", action="store_true", help="Compute LEACE projector in U (default: False if not provided)")
    p.add_argument("--leace_ridge", type=float, default=1e-6)
    p.add_argument("--noise_sigma", type=float, default=0.02)
    p.add_argument("--forget_commands", nargs='+', default=["yes"], help="Commands to forget, e.g. 'yes' 'no' 'up'")
    p.add_argument("--encoder", type=str, default="ecapa", help="Encoder to use - 'ecapa', 'wav2vec2', 'hubert'")
    args = p.parse_args()

    # Run
    run_pipeline(
        data_root=args.data_root, save_dir=args.save_dir, device=args.device, seed=args.seed,
        dev_frac=args.dev_frac,
        sae_hidden_mult=args.sae_hidden_mult, sae_k=args.sae_k, sae_epochs=args.sae_epochs,
        build_U_top=args.build_U_top, beta=args.beta, leace=args.leace, leace_ridge=args.leace_ridge, noise_sigma=args.noise_sigma,
        encoder=args.encoder, forget_commands=args.forget_commands
    )
