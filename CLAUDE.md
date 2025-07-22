# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TAPAS is a research codebase for "The Art of Imitation: Learning Long-Horizon Manipulation Tasks from Few Demonstrations". It implements Task-Parametrized Gaussian Mixture Models (TP-GMM) for robot manipulation learning from demonstrations.

## Core Architecture

### Policy System
- **Primary Policy**: `GMMPolicy` using TP-GMM for task-parametrized learning
- **Alternative Policies**: `LSTMPolicy`, `DiffusionPolicy`, `MotionPlannerPolicy`
- **Unified Interface**: All policies inherit from `Policy` base class
- **Key Files**: `tapas_gmm/policy/gmm.py`, `tapas_gmm/policy/models/tpgmm.py`

### Encoder Architecture
- **Vision Encoders**: Keypoints, CNN, ViT, Beta-VAE, MONet, Transporter
- **Dense Correspondence**: ViT-based dense correspondence learning for keypoint detection
- **Modular Design**: Encoders are policy-agnostic and can be mixed/matched
- **Key Files**: `tapas_gmm/encoder/`, `tapas_gmm/dense_correspondence/`

### Environment Interface
- **Supported Environments**: RLBench, ManiSkill2, Franka Emika robot
- **Unified Interface**: `BaseEnvironment` class with common observation handling
- **Multi-Camera Support**: Handles multiple camera views with proper transforms
- **Key Files**: `tapas_gmm/env/`, environment-specific configs in `conf/env/`

### Data Pipeline
- **Observation System**: TensorDict-based multi-modal observations (RGB, depth, proprioception)
- **Fragment-Based Training**: Efficient training on trajectory fragments
- **Pre-computed Encodings**: Supports offline encoding for faster training
- **Key Files**: `tapas_gmm/dataset/`, `tapas_gmm/utils/observation.py`

## Common Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Set up RLBench environment
source rlbench_mode.sh

# Install riepybdlib for Riemannian geometry
git clone git@github.com:vonHartz/riepybdlib.git
cd riepybdlib && pip install -e .
```

### Data Collection
```bash
# Collect demonstrations
tapas-collect --config conf/collect_data/rlbench_expert.py -t TaskName

# Collect with multiprocessing
tapas-collect --config conf/collect_data/maniskill_mp.py -t TaskName
```

### Training and Evaluation
```bash
# Encode trajectories with keypoints
tapas-kp-encode --config conf/kp_encode_trajectories/vit/base.py -t TaskName -f demos

# Train behavior cloning policy
tapas-bc --config conf/behavior_cloning/keypoints/default.py -t TaskName

# Evaluate policy
tapas-eval --config conf/evaluate/gmm/test_auto_tgrip_rlbench.py -t TaskName -f demos
```

### Pretrain Encoders
```bash
# Pretrain keypoint encoder
tapas-pretrain --config conf/pretrain/keypoints/default.py -t TaskName

# Pretrain CNN encoder  
tapas-pretrain --config conf/pretrain/cnn/rlbench.py -t TaskName
```

## Configuration System

### Structure
- **Hierarchical Configs**: Dataclass-based configs in `conf/` directory
- **Machine-Specific**: Use `conf/_machine.py` for machine-specific paths
- **Command Line Overrides**: Use `--overwrite` flag to override any config key

### Key Config Categories
- `conf/collect_data/`: Data collection configurations
- `conf/encoder/`: Encoder architecture configurations
- `conf/policy/`: Policy configurations
- `conf/env/`: Environment-specific configurations
- `conf/evaluate/`: Evaluation configurations

### Example Config Override
```bash
tapas-eval --config conf/evaluate/gmm/test.py -t TaskName -f demos \
  --overwrite wandb_mode=disabled policy.suffix=tx data_naming_config.data_root=/custom/path
```

## Development Guidelines

### Adding New Policies
1. Inherit from `Policy` base class in `tapas_gmm/policy/policy.py`
2. Implement required methods: `predict_action()`, `load_checkpoint()`, `save_checkpoint()`
3. Add policy config dataclass inheriting from `PolicyConfig`
4. Register policy in `tapas_gmm/policy/__init__.py`

### Adding New Encoders
1. Inherit from appropriate base class (e.g., `ObservationEncoder`)
2. Implement forward pass and encoding dimension methods
3. Add encoder config to `conf/encoder/` directory
4. Register encoder in `tapas_gmm/encoder/__init__.py`

### Adding New Environments
1. Inherit from `BaseEnvironment` class
2. Implement required methods: `reset()`, `step()`, `get_observation()`
3. Add environment config in `conf/env/` directory
4. Register environment in `tapas_gmm/env/__init__.py`

## Key Geometric Concepts

### Riemannian Manifolds
- **SO(3) Rotations**: Proper quaternion handling with `riepybdlib`
- **SE(3) Poses**: Homogeneous transforms for pose representation
- **Manifold Operations**: Logarithmic maps, exponential maps, geodesics

### Observation Transforms
- **Frame Transforms**: Proper coordinate frame handling between cameras/robot
- **Keypoint Projection**: 3D keypoints to 2D image coordinates
- **Dense Correspondence**: Pixel-to-pixel matching across viewpoints

## Testing and Validation

### Running Tests
```bash
# No specific test framework - validate through example workflows
# Test data collection
tapas-collect --config conf/collect_data/rlbench_expert.py -t StackCups

# Test encoding
tapas-kp-encode --config conf/kp_encode_trajectories/vit/base.py -t StackCups -f demos

# Test evaluation
tapas-eval --config conf/evaluate/gmm/test.py -t StackCups -f demos
```

### Validation Notebooks
- `notebooks/rlbench/`: Task-specific validation notebooks
- `notebooks/franka/`: Real robot validation notebooks
- Use notebooks to visualize learned policies and debug issues

## Common Issues and Solutions

### CUDA/GPU Issues
- Ensure proper GPU selection with `tapas_gmm.utils.select_gpu`
- Check CUDA paths in environment setup
- Use `nvitop` for GPU monitoring

### Environment Setup
- RLBench requires specific Coppelia Sim setup - use `rlbench_mode.sh`
- ManiSkill2 requires specific GPU drivers for rendering
- Franka requires specific hardware setup

### Configuration Errors
- Verify `conf/_machine.py` has correct paths for your machine
- Check that all required dependencies are installed for chosen environment
- Use `--overwrite` to debug configuration issues

## Research Extensions

### New Task Domains
- Add new environments by extending `BaseEnvironment`
- Create task-specific configs in `conf/env/`
- Implement task-specific reward functions and success criteria

### Novel Representation Learning
- Experiment with new encoders in `tapas_gmm/encoder/`
- Implement new dense correspondence methods
- Add new keypoint detection strategies

### Policy Improvements
- Extend TP-GMM with new parametrization methods
- Implement new policy architectures
- Add multi-task learning capabilities

## File Structure Overview

```
tapas_gmm/
├── policy/           # Policy implementations (GMM, LSTM, Diffusion)
├── encoder/          # Vision encoders and representation learning
├── env/             # Environment interfaces and wrappers
├── dataset/         # Data loading and processing
├── utils/           # Utilities for geometry, observation, etc.
├── viz/             # Visualization tools
└── dense_correspondence/  # Dense correspondence learning

conf/
├── collect_data/    # Data collection configurations
├── encoder/         # Encoder configurations
├── policy/          # Policy configurations
├── env/            # Environment configurations
└── evaluate/       # Evaluation configurations
```

## Additional Resources

- **Project Website**: http://tapas-gmm.cs.uni-freiburg.de/
- **Paper**: IEEE RA-L 2024 "The Art of Imitation: Learning Long-Horizon Manipulation Tasks From Few Demonstrations"
- **Dependencies**: RLBench, ManiSkill2, riepybdlib, MPlib for motion planning