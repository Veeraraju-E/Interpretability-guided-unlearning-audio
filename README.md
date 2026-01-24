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

### 2. Run Unlearning Experiments

#### CLI

```bash
# Single-class unlearning
python -m src.cli.main run --config path/to/config

# Multi-class unlearning (all classes)
python -m src.cli.main run --config path/to/config --show-config

# Continual unlearning (sequential forgetting)
python -m src.cli.main run --config path/to/config

# Grid search over SAE parameters
python -m src.cli.main grid --config path/to/hyperparam/config

# List available configurations
python -m src.cli.main list-configs
```

#### Additional Analysis Scripts

The framework also includes standalone analysis scripts that can be run directly:

```bash
# Hyperparameter analysis
python -m src.orchestration.hyperparameter_analysis --config path/to/config [--mode analyze|plot|all]

# Build U top analysis
python -m src.orchestration.build_u_top_analysis --config path/to/config

# SAE training loss analysis
python -m src.orchestration.sae_training_loss_analysis --config path/to/config [--mode train|plot|all]
```

### 3. Configuration Modes

Set the `orchestration.mode` in your config file:

- `single`: Unlearn specific classes (default)
- `multi`: Unlearn all classes sequentially  
- `continual`: Sequential forgetting with cumulative effects

## Architecture

```
src/
├── core/
│   ├── edits.py                   # LEACE
│   ├── encoders.py                # Audio encoders
│   ├── sae.py                     # Sparse Autoencoder
│   ├── evals/                     # Evaluation metrics and probes
│   └── utils/                     # Whitening, normalization, and stats
├── datasets/                      # Dataset adapters and preprocessing
├── orchestration/                 # Unlearning workflow orchestration
│   ├── single_run.py              # Single-class unlearning
│   ├── multi_run.py               # Multi-class unlearning
│   ├── continual.py               # Continual unlearning
│   └── hyperparameter_analysis.py # Hyperparameter analysis
└── cli/                     # Command-line interfaces
    ├── main.py              # Main CLI
    └── grid_search.py       # Grid search CLI
```

<!-- ## Citation

If you use our method, please consider citing:

```bibtex
@article{sae-guided-unlearning-audio,
  title={sae-guided Unlearning for Audio Models},
  author={Your Name},
  journal={Your Journal},
  year={2026}
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