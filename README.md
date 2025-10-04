# Interpretability-guided-unlearning-audio (Modular)

This repo performs class/speaker unlearning in audio with SAE-driven feature mining, LEACE edits, and comprehensive evaluations and interpretability.

## Quickstart

1) Choose a dataset config under `configs/`:
- `configs/voxceleb.yaml`
- `configs/speech_commands.yaml`
- `configs/audiomnist.yaml`
- `configs/librispeech.yaml`
- `configs/lahaja.yaml`

2) Run single unlearning (config-driven):

```bash
python src/cli/run.py --config configs/voxceleb.yaml
```

Set `orchestration.mode` to `multi` for all classes or `continual` for sequential forgetting.

3) Interpretability over saved artifacts:

```bash
python src/cli/interpret.py --config configs/voxceleb.yaml
```

4) Grid search (basic, modular wrapper):

```bash
python src/cli/grid.py --config configs/voxceleb.yaml
```

## Structure
- `src/core/*`: whitening, encoders, SAE, feature mining, edits, evals
- `src/datasets/*`: dataset adapters (unified manifest + precompute)
- `src/orchestration/*`: single, multi, continual unlearning flows
- `src/cli/*`: config-driven entrypoints for training, analysis, and grid search
