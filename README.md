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

Install FUNCTO (see readme in `attempt_five` directory).

We cannot use endpoints of another university (the university which created functo). Instead, we implemented local classes to use Owl and SAM.

Install necessary libraries:

```
pip install torch pillow numpy torchvision matplotlib tensordict pandas loguru omegaconf scipy tqdm jsonpickle numba wandb open3d opencv-python shapely
```

Clone TAPAS additionally into `attempt_five`.

replace attempt_five/TAPAS/tapas_gmm/encoder/keypoints.py with keypoints.py which is provided in https://github.com/maximelianos/TAPAS
(specifically, encode(self, batch: SceneObservation) function differs between two files.

Install grounding DINO.

```
cd attempt_five
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -r requirements.txt
pip install -e .
```

Install SAM.

```
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
pip install -e .
```

**Optional**:
- See `maniskill_src` to install Maniskill simulator with custom environments
- See `tapas_README.md` to install [TAPAS](https://tapas-gmm.cs.uni-freiburg.de/)

## Run

Go to `attempt_five` directory.

To run functo:
* Move the image which you desire to obtain keypoints to test_data/observe_test_rgb and name it as 00000.jpg.
* In the attempt_five/run_functo_pipeline file, change "test_target_label": "bucket" to whatever the object you want to use.
* In the attempt_five/run_functo_pipeline file, in row 120, change text_prompts=["bucket handle", "bucket"] accordingly.
* For example, if you want to get spout of a kettle and handle of a kettle, please use "kettle spout", "kettle handle"
* Adjust box names in line 125 and 128 accordingly. Like "kettle spout", "kettle handle".
* Change directory to my_models and use python test_tapas_keypoints.py
