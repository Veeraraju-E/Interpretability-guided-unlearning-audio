"""
Waveform FGSM/PGD/APGD-CE (Croce & Hein ICML 2020; AutoAttack main component).

Adapted from torchattacks.APGD for mono sequences (batch, time), clip domain [-1, 1].
"""

from typing import Dict, List, Literal, Tuple
import numpy as np
import torch
import torch.nn as nn

from src.core.adversarial.hf_waveform_model import (
    HFWaveformToLogits,
    manifest_entries_have_audio,
    resolve_hf_model_name,
    prepare_mono_waveform_for_model,
)

AttackMethod = Literal["fgsm", "pgd", "autoattack"]
AttackNorm = Literal["linf", "l2"]


def _clip_wav(audio: torch.Tensor) -> torch.Tensor:
    return audio.clamp(-1.0, 1.0)


def attack_fgsm(
    model: nn.Module,
    wav: torch.Tensor,
    y: torch.Tensor,
    *,
    epsilon: float,
    norm: AttackNorm,
) -> torch.Tensor:
    """wav: (N, T); y: (N,)"""
    ce = nn.CrossEntropyLoss()
    wav = wav.detach().clone()
    wav.requires_grad_(True)
    logits = model(wav)
    loss = ce(logits, y)
    g = torch.autograd.grad(loss, wav)[0]
    x = wav.detach()
    if norm == "linf":
        adv = x + epsilon * torch.sign(g)
        adv = torch.min(torch.max(adv, x - epsilon), x + epsilon)
    elif norm == "l2":
        g_flat = g.view(g.size(0), -1)
        g_norm = g_flat.norm(dim=1).clamp_min(1e-12).view(-1, 1)
        adv = x + epsilon * g / g_norm
        delta = adv - x
        d_flat = delta.view(delta.size(0), -1)
        rn = d_flat.norm(dim=1).clamp_min(1e-12).view(-1, 1)
        adv = x + delta * torch.min(torch.ones_like(rn), epsilon / rn)
    else:
        raise ValueError(f"Unsupported norm '{norm}'. Use linf or l2.")
    return _clip_wav(adv.detach())


def attack_pgd(
    model: nn.Module,
    wav: torch.Tensor,
    y: torch.Tensor,
    *,
    epsilon: float,
    alpha: float,
    steps: int,
    norm: AttackNorm,
    random_start: bool = True,
) -> torch.Tensor:
    """wav: (N, T); y: (N,)"""
    if steps < 1:
        raise ValueError("PGD steps must be >= 1.")
    ce = nn.CrossEntropyLoss()
    x = wav.detach()
    adv = x.clone()

    if random_start:
        if norm == "linf":
            adv = adv + torch.empty_like(adv).uniform_(-epsilon, epsilon)
            adv = torch.min(torch.max(adv, x - epsilon), x + epsilon)
        else:
            noise = torch.randn_like(adv)
            n_flat = noise.view(noise.size(0), -1).norm(dim=1).clamp_min(1e-12).view(-1, 1)
            rnd = epsilon * noise / n_flat
            adv = x + rnd
            delta = adv - x
            d_flat = delta.view(delta.size(0), -1)
            rn = d_flat.norm(dim=1).clamp_min(1e-12).view(-1, 1)
            adv = x + delta * torch.min(torch.ones_like(rn), epsilon / rn)

    adv = _clip_wav(adv.detach()).clone()

    for _ in range(steps):
        adv.requires_grad_(True)
        logits = model(adv)
        loss = ce(logits, y)
        g = torch.autograd.grad(loss, adv)[0]
        if norm == "linf":
            adv = adv.detach() + alpha * torch.sign(g)
            adv = torch.min(torch.max(adv.detach(), x - epsilon), x + epsilon)
        else:
            g_flat = g.view(g.size(0), -1)
            g_n = g_flat.norm(dim=1).clamp_min(1e-12).view(-1, 1)
            adv = adv.detach() + alpha * g / g_n
            delta = adv - x
            d_flat = delta.view(delta.size(0), -1)
            rn = d_flat.norm(dim=1).clamp_min(1e-12).view(-1, 1)
            adv = x + delta * torch.min(torch.ones_like(rn), epsilon / rn)

        adv = _clip_wav(adv.detach())
    return adv.detach()



def _scalar_oscillation(loss_hist: np.ndarray, j: int, k_cur: int, k3: float) -> bool:
    t = 0.0
    for cnt in range(k_cur):
        t += float(loss_hist[j - cnt] > loss_hist[j - cnt - 1])
    return bool(t <= (k_cur * k3))


def apgd_ce_one_sample(
    model: nn.Module,
    x_flat: torch.Tensor,
    y: torch.Tensor,
    *,
    epsilon: float,
    steps: int,
    norm: AttackNorm,
    rho: float,
    device: torch.device,
) -> torch.Tensor:
    """
    APGD-CE (batch=1) after torchattacks.APGD.attack_single_run.
    Linf: adaptive step scheduling + oscillation heuristic.
    L2: restarted PGD with the same epsilon/steps schedule (canonical APGD L2 projector differs in border cases;
        we use deterministic PGD on the L2 ball as the faithful practical substitute for raw waveforms).
    """
    if x_flat.shape[0] != 1:
        raise ValueError("apgd_ce_one_sample expects batch size 1.")
    if norm == "l2":
        alpha = max(float(epsilon) / max(float(steps), 1.0), 1e-10)
        return attack_pgd(
            model,
            x_flat,
            y,
            epsilon=float(epsilon),
            alpha=float(alpha),
            steps=int(steps),
            norm="l2",
            random_start=True,
        )
    steps_2 = max(int(0.22 * steps), 1)
    steps_min = max(int(0.06 * steps), 1)
    size_decr = max(int(0.03 * steps), 1)

    x = x_flat.detach()

    t = 2 * torch.rand_like(x) - 1
    x_adv = x + epsilon * t / (t.abs().amax() + 1e-12)
    x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
    x_adv = _clip_wav(x_adv)

    criterion_indiv = nn.CrossEntropyLoss(reduction="none")

    loss_hist: List[float] = []

    x_adv.requires_grad_(True)
    logits = model(x_adv)
    loss_indiv = criterion_indiv(logits, y)
    loss = loss_indiv.sum()
    grad = torch.autograd.grad(loss, x_adv)[0].detach()

    loss_best = loss_indiv.detach().clone()
    grad_best = grad.clone()
    x_best_adv = x_adv.detach().clone()
    if logits.argmax(dim=1).item() != y.item():
        x_best_adv = x_adv.detach().clone()

    loss_hist.append(float(loss_best.item()))

    step_size = torch.tensor(2.0 * epsilon, dtype=x.dtype, device=x.device)

    x_adv_old = x_adv.clone().detach().requires_grad_(False)
    x_best = x_adv.detach().clone()
    counter3 = 0
    kk = steps_2
    reduced_last_check = False
    loss_best_last_check = loss_best.detach().clone()

    for ii in range(steps):
        with torch.no_grad():
            x_adv_n = x_adv.detach()
            grad2 = x_adv_n - x_adv_old
            x_adv_old = x_adv_n.clone()
            a = 0.75 if ii > 0 else 1.0

            x_adv_1 = x_adv_n + step_size * torch.sign(grad)
            x_adv_1 = torch.minimum(torch.maximum(x_adv_1, x - epsilon), x + epsilon)
            x_mid = x_adv_n + (x_adv_1 - x_adv_n) * a + grad2 * (1.0 - a)
            x_adv_1 = torch.minimum(torch.maximum(x_mid, x - epsilon), x + epsilon)
            x_adv = _clip_wav(x_adv_1)

        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss_indiv = criterion_indiv(logits, y)
        loss_sum = loss_indiv.sum()
        grad = torch.autograd.grad(loss_sum, x_adv)[0].detach()

        li = loss_indiv.detach()

        fooled = logits.argmax(dim=1).item() != y.item()
        if fooled:
            x_best_adv = x_adv.detach().clone()

        loss_hist.append(float(li.item()))
        with torch.no_grad():
            if li.item() > loss_best.item():
                loss_best = li.detach().clone()
                x_best = x_adv.detach().clone()
                grad_best = grad.clone()

            counter3 += 1
            if counter3 >= kk:
                arr = np.array(loss_hist, dtype=np.float64)
                jj = arr.size - 1
                oscill = _scalar_oscillation(arr, jj, kk, rho) if jj >= kk else False
                lb_prev = float(loss_best_last_check.detach().cpu().item())
                lc_cur = float(loss_best.detach().cpu().item())
                no_improve = (not reduced_last_check) and (lb_prev >= lc_cur)
                combined = oscill or no_improve
                reduced_last_check = combined
                loss_best_last_check = loss_best.detach().clone()

                if combined:
                    step_size *= 0.5
                    x_adv = x_best.clone().detach()
                    grad = grad_best.clone()

                counter3 = 0
                kk = max(kk - size_decr, steps_min)

    return _clip_wav(x_best_adv.detach())


def run_attack_single(
    model: nn.Module,
    wav_1_t: torch.Tensor,
    ti: torch.Tensor,
    *,
    method: AttackMethod,
    epsilon: float,
    alpha: float,
    steps: int,
    norm: AttackNorm,
    n_restarts: int,
    device: torch.device,
) -> torch.Tensor:
    """wav_1_t: (1,T), ti: class index tensor."""
    y = ti.view(1).long()

    if method == "fgsm":
        return attack_fgsm(model, wav_1_t, y, epsilon=epsilon, norm=norm)
    if method == "pgd":
        return attack_pgd(
            model, wav_1_t, y,
            epsilon=epsilon, alpha=alpha, steps=steps, norm=norm, random_start=True,
        )
    if method == "autoattack":
        best = wav_1_t.clone()
        best_loss = None
        for r in range(max(1, n_restarts)):
            if r > 0:
                torch.manual_seed((torch.randint(1 << 30, (1,)).item() + r) % (1 << 31))
            adv_try = apgd_ce_one_sample(
                model, wav_1_t, y,
                epsilon=epsilon, steps=steps, norm=norm, rho=0.75, device=device,
            )
            with torch.no_grad():
                ce = nn.CrossEntropyLoss()
                L = float(ce(model(adv_try), y).item())
            if best_loss is None or L > best_loss:
                best_loss = L
                best = adv_try.clone()
        return best

    raise ValueError(f"Unknown attack method '{method}'.")


def build_attack_include_mask(manifest: List[Dict], forget_labels_str: List[str], subset: str, mask_dev: np.ndarray) -> np.ndarray:
    subset = subset.strip().lower()
    if subset == "all":
        return np.ones(len(manifest), dtype=bool)
    if subset == "forget_dev":
        fl = {str(f) for f in forget_labels_str}
        idx = []
        for i, m in enumerate(manifest):
            lbl = str(m["label"])
            dev = bool(mask_dev[i])
            idx.append(dev and (lbl in fl))
        return np.array(idx, dtype=bool)
    raise ValueError("attack.subset must be 'all' or 'forget_dev'.")


def waveform_attack_manifest(
    manifest_ordered: List[Dict],
    mask_dev: np.ndarray,
    cls2id: Dict[str, int],
    clf: nn.Module,
    encoder_cfg: Dict,
    *,
    attack_cfg: Dict,
    device: str,
    forget_labels: List,
) -> List[Dict]:
    from tqdm import tqdm

    method = attack_cfg["method"].strip().lower()
    if method in ("none", "off", "false"):
        raise RuntimeError("waveform_attack_manifest called while attack method is none.")
    if method == "autoattack":
        method_t: AttackMethod = "autoattack"
    elif method == "fgsm":
        method_t = "fgsm"
    elif method == "pgd":
        method_t = "pgd"
    else:
        raise ValueError(f"Unknown attack.method '{attack_cfg['method']}'.")

    norm_raw = str(attack_cfg.get("norm", "linf")).strip().lower()
    if norm_raw not in ("linf", "l2"):
        raise ValueError("attack.norm must be 'linf' or 'l2'.")
    norm_t: AttackNorm = norm_raw  # type: ignore

    epsilon = float(attack_cfg["eps"])
    steps = int(attack_cfg["steps"])
    alpha = float(attack_cfg.get("alpha", epsilon / max(steps, 1)))
    n_restarts = int(attack_cfg.get("restarts", 1))
    subset = attack_cfg.get("subset", "all")

    tgt_device = torch.device(device)
    model_name = resolve_hf_model_name(encoder_cfg)
    wrapped = HFWaveformToLogits(model_name, clf, device=str(device))
    tgt_sr = wrapped.target_sr

    include_mask = build_attack_include_mask(
        manifest_ordered, [str(f) for f in forget_labels], subset, mask_dev,
    )

    out: List[Dict] = []
    for i, entry in tqdm(list(enumerate(manifest_ordered)), desc=f"attack_{method}", total=len(manifest_ordered)):
        e = dict(entry)
        wav = prepare_mono_waveform_tensor(e, device=tgt_device, target_sr=tgt_sr)
        if wav.dim() != 2 or wav.size(0) != 1:
            raise RuntimeError(f"Expected mono waveform shape (1, T), got {wav.shape}")

        lbl = e.get("label", "NA")
        kid = cls2id[str(lbl)]
        ti = torch.tensor([kid], device=tgt_device, dtype=torch.long)
        wav_input = wav.to(tgt_device).detach()

        if include_mask[i]:
            adv_w = run_attack_single(
                wrapped,
                wav_input.clone(),
                ti,
                method=method_t,
                epsilon=epsilon,
                alpha=alpha,
                steps=steps,
                norm=norm_t,
                n_restarts=n_restarts,
                device=tgt_device,
            )
            arr = adv_w.squeeze(0).detach().cpu().numpy().astype(np.float32)
        else:
            arr = wav.squeeze(0).detach().cpu().numpy().astype(np.float32)

        e["audio_array"] = arr
        e["sampling_rate"] = tgt_sr
        out.append(e)
    return out


def build_attacked_manifest(pre_clean, cfg: Dict, clf: nn.Module, *, device: str) -> List[Dict]:
    atk = cfg.get("attack") or {}
    if not atk.get("enabled"):
        raise RuntimeError("build_attacked_manifest expects attack.enabled True.")

    encoder_cfg = cfg["encoder"]
    emb_p = encoder_cfg.get("embeddings_path")
    if emb_p is not None and str(emb_p).strip().lower() not in ("none", "", "null"):
        raise RuntimeError(
            "Waveform attacks require loading raw waveforms via the dataset adapter (encoder.embeddings_path must be unset). "
            "Precomputed embeddings-only runs cannot differentiate through the waveform."
        )

    if not manifest_entries_have_audio(pre_clean.manifest):
        raise RuntimeError(
            "Waveform attacks require raw audio per manifest row ('audio_array' + 'sampling_rate') or a readable file path."
        )

    atk = atk.copy()
    cls2id = _classes_to_cls2id(pre_clean.labels)

    return waveform_attack_manifest(
        list(pre_clean.manifest),
        pre_clean.mask_dev,
        cls2id,
        clf,
        encoder_cfg,
        attack_cfg=atk,
        device=device,
        forget_labels=list(cfg["dataset"]["forget_labels"]),
    )


def _classes_to_cls2id(labels: List[str]) -> Dict[str, int]:
    classes = sorted(list(set(str(x) for x in labels)))
    return {str(c): i for i, c in enumerate(classes)}


def rebuild_precomp_after_attack(attacked_manifest: List[Dict], encoder_cfg: Dict, device: str):
    """Recompute embeddings and whitening."""
    import numpy as np
    from src.core.encoders import extract_embeddings
    from src.datasets.adapter_base import Precomp
    from src.core.utils.whitening import fit_whitener, apply_whitener

    wav2vec2_model = encoder_cfg.get("wav2vec2_model", "facebook/wav2vec2-base")
    hubert_model = encoder_cfg.get("hubert_model", "facebook/hubert-base-ls960")
    conformer_model = encoder_cfg.get("conformer_model", "facebook/wav2vec2-conformer-rel-pos-large")

    E_all, labels, _paths = extract_embeddings(
        attacked_manifest,
        encoder=encoder_cfg["name"],
        device=device,
        wav2vec2_model=wav2vec2_model,
        hubert_model=hubert_model,
        conformer_model=conformer_model,
        label_key="label",
    )
    manifest_labels = [str(m["label"]) for m in attacked_manifest]
    if list(labels) != manifest_labels:
        raise RuntimeError("Internal fault: embeddings label order mismatches attacked manifest.")

    split_dev = np.array([m["split"] == "dev" for m in attacked_manifest], dtype=bool)
    split_test = np.array([m["split"] == "test" for m in attacked_manifest], dtype=bool)
    classes_sorted = sorted(list(set(labels)))

    mu, W = fit_whitener(E_all[split_dev])
    Z_all = apply_whitener(E_all, mu, W)

    return Precomp(
        manifest=list(attacked_manifest),
        labels=labels,
        classes=classes_sorted,
        E_all=E_all,
        Z_all=Z_all,
        mu=mu,
        W=W,
        mask_dev=split_dev,
        mask_test=split_test,
    )
