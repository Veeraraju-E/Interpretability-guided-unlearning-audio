# Interpretability-guided-unlearning-audio

A comprehensive framework for interpretability-guided unlearning in audio models using SAE-driven feature mining, LEACE edits, and advanced evaluation techniques.

## Installation

```bash
# Clone the repository
git clone github.com/Veeraraju-E/Interpretability-guided-unlearning-audio.git
cd Interpretability-guided-unlearning-audio

# Install dependencies
pip install -r requirements.txt
```

## Quickstart

### 1. Choose a Dataset Configuration

The framework supports multiple datasets with pre-configured settings:

- `configs/voxceleb.yaml` - VoxCeleb speaker verification
- `configs/audiomnist.yaml` - AudioMNIST digit classification  
- `configs/speech_commands.yaml` - Speech Commands recognition
- `configs/librispeech.yaml` - LibriSpeech speaker identification
- `configs/lahaja.yaml` - Lahaja speech recognition
- `configs/mls_spanish.yaml` - MLS Spanish speech recognition

### 2. CLI Commands

#### Run Unlearning Experiments

```bash
# Run unlearning experiments (mode determined by config)
python -m src.cli.main run --config path/to/config.yaml

# Show configuration summary before running
python -m src.cli.main run --config path/to/config.yaml --show-config
```

The `run` command supports three modes (set via `orchestration.mode` in config):
- `single`: Unlearn specific classes (default)
- `multi`: Unlearn all classes sequentially  
- `continual`: Sequential forgetting with cumulative effects

#### Interpretability Analysis

```bash
# Run interpretability analysis using config (auto-finds run directories)
python -m src.cli.main interpret --config path/to/config.yaml

# Run interpretability analysis on specific run directory
python -m src.cli.main interpret --run-dir path/to/seed42_forget_speaker123

# Specify device
python -m src.cli.main interpret --config path/to/config.yaml --device cpu
```

#### Grid Search

```bash
# Run grid search over SAE parameters
python -m src.cli.main grid --config path/to/config.yaml

# Show config before running
python -m src.cli.main grid --config path/to/config.yaml --show-config
```

#### Configuration Utilities

```bash
# List available configuration files
python -m src.cli.main list-configs

# Validate a configuration file
python -m src.cli.main validate-config --config path/to/config.yaml
```

## Architecture

```
src/
├── core/
│   ├── edits.py             # LEACE and noise injection
│   ├── encoders.py          # Audio encoders (ECAPA, wav2vec2, etc.)
│   ├── sae.py               # Sparse Autoencoder
│   ├── interpretation.py   # Interpretation protocol (Steps 1-6)
│   ├── acoustic_features.py # Acoustic feature extraction
│   ├── evals/               # Evaluation metrics, probes, and attribute directions
│   └── utils/               # Whitening, normalization, and stats
├── datasets/                # Dataset adapters and preprocessing
├── orchestration/           # Unlearning workflow orchestration
│   ├── single_run.py        # Single-class unlearning
│   ├── multi_run.py         # Multi-class unlearning
│   └── continual.py         # Continual unlearning
└── cli/                     # Command-line interfaces
    ├── main.py              # Main CLI (run, interpret, grid, list-configs, validate-config)
    ├── grid_search.py       # Grid search implementation
    └── aggregate_stats.py    # Statistics aggregation utilities
```

<!-- ## Citation

If you use our method, please consider citing:

```bibtex
@article{sae-guided-unlearning-audio,
  title={sae-guided Unlearning for Audio Models},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
``` -->

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the Apache License.