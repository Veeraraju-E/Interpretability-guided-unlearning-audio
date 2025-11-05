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

### 2. Run Unlearning Experiments

#### CLI

```bash
# Single-class unlearning
./unlearn-audio run --config configs/voxceleb.yaml

# Multi-class unlearning (all classes)
./unlearn-audio run --config configs/voxceleb.yaml --show-config

# Continual unlearning (sequential forgetting)
./unlearn-audio run --config configs/voxceleb.yaml

# Grid search over SAE parameters
./unlearn-audio grid --config configs/voxceleb.yaml

# Interpretability analysis
./unlearn-audio interpret --config configs/voxceleb.yaml

# List available configurations
./unlearn-audio list-configs

# Validate configuration
./unlearn-audio validate-config --config configs/voxceleb.yaml
```


### 3. Configuration Modes

Set the `orchestration.mode` in your config file:

- `single`: Unlearn specific classes (default)
- `multi`: Unlearn all classes sequentially  
- `continual`: Sequential forgetting with cumulative effects

## Architecture

```
src/
├── core/                    # Core algorithms and utilities (consolidated)
│   ├── edits.py             # LEACE and noise injection (merged from edits/)
│   ├── encoders.py          # Audio encoders (merged from encoders/)
│   ├── sae.py               # Sparse Autoencoder (merged from sae/)
│   ├── analysis/            # Feature discovery and subspace analysis
│   ├── evals/               # Evaluation metrics and probes
│   └── utils/               # Whitening, normalization, and statistics
├── datasets/                # Dataset adapters and preprocessing
├── orchestration/           # Unlearning workflow orchestration
│   ├── single_run.py        # Single-class unlearning
│   ├── multi_run.py         # Multi-class unlearning
│   ├── build_u_top_analysis.py # Build U top analysis
│   ├── hyperparameter_analysis.py # Hyperparameter analysis
│   └── sae_training_loss_analysis.py # SAE training analysis
└── cli/                     # Command-line interfaces
    ├── main.py              # Main CLI
    └── grid_search.py       # Grid search implementation
```

<!-- ## Citation

If you use this framework in your research, please cite:

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

This project is licensed under the MIT License - see the LICENSE file for details.