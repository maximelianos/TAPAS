# Robotic Tool Use

* Object pictures are collected in [SAPIEN](https://sapien.ucsd.edu/browse) simulator
* 2D tool keypoints are detected with [FUNCTO](https://sites.google.com/view/functo)
* The set of keypoints is filtered to obtain handles and functional points using segmentation with [Grounded DINO+SAM](https://huggingface.co/docs/transformers/main/en/model_doc/grounding-dino)

## Dependencies

A Nvidia GPU is required for FUNCTO and Grounded SAM.

## Installation

Install Miniconda.

Add alias to ~/.bashrc:
```
alias c='conda activate' 
```

Create conda environment.

```
conda create -n tapas python=3.10 jupyter notebook
c tapas
```

Install SAPIEN.

```
pip install sapien==2.2.2
```

See functo folder to install FUNCTO.

Optional:
- See `maniskill_src` to install Maniskill simulator with custom environments
- See `tapas_README.md` to install [TAPAS](https://tapas-gmm.cs.uni-freiburg.de/)

## Run

See functo folder to generate SAPIEN images, run keypoint detection and Grounded SAM.
