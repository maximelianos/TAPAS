# ViT Keypoint Extraction and Trajectory Prediction in TAPAS

## Overview

This report analyzes the Vision Transformer (ViT) based keypoint extraction system in TAPAS, focusing on the types of keypoints extracted and their usage in trajectory prediction. The system uses dense correspondence learning with ViT features to identify and track semantically meaningful keypoints across different viewpoints and time steps.

## Keypoint Types and Classifications

### 1. Keypoint Categories

The TAPAS system extracts different types of keypoints based on their semantic meaning and functional role:

#### **Center Keypoints**
- **Definition**: Geometric centers or centroids of objects
- **Selection Method**: `ReferenceSelectionTypes.MASK_CENTER` (line 18 in `conf/encoder/vit_keypoints/candidates_center.py`)
- **Usage**: Provide stable reference points for object tracking and pose estimation
- **Implementation**: Computed via `get_mask_center()` function (`tapas_gmm/encoder/keypoints.py:1018`)

#### **Functional Keypoints** 
- **Definition**: Task-relevant locations such as grasp points, contact surfaces, or interaction areas
- **Selection Method**: `ReferenceSelectionTypes.MASK_AVG` (line 18 in `conf/encoder/vit_keypoints/candidates_avg.py`)
- **Usage**: Enable task-specific manipulation behaviors by identifying action-relevant features
- **Implementation**: Computed via `get_masked_avg_descriptor()` function (`tapas_gmm/encoder/keypoints.py:1044`)

#### **Hold Keypoints**
- **Definition**: Locations for maintaining object poses during manipulation
- **Selection Method**: `ReferenceSelectionTypes.MANUAL` (line 47 in `conf/encoder/vit_keypoints/nofilter.py`)
- **Usage**: Support stable grasping and object manipulation throughout task execution
- **Implementation**: Manual selection through `KeypointSelector` interface (`tapas_gmm/encoder/keypoints.py:1082`)

### 2. Keypoint Extraction Pipeline

#### **Dense Correspondence Learning**
The keypoint extraction process begins with dense correspondence learning using ViT features:

```python
# From tapas_gmm/encoder/keypoints.py:420-424
image_a_pred = self.compute_descriptor(img_a)
image_a_pred = self.process_network_output(image_a_pred, 1)
image_b_pred = self.compute_descriptor(img_b)
image_b_pred = self.process_network_output(image_b_pred, 1)
```

#### **Keypoint Types Enumeration**
The system supports multiple keypoint learning strategies (`tapas_gmm/encoder/models/keypoints/keypoints.py:10-22`):

- **SD (Sample Descriptors)**: Random sampling from masked reference images
- **SDS (Sample Descriptors with Selection)**: High-certainty, well-distributed keypoints
- **WDS (Weighted Descriptor Selection)**: Learnable weighted combinations
- **ODS (Optimized Descriptor Set)**: End-to-end optimization with policy
- **E2E (End-to-End)**: Joint training of correspondence network and policy

## Keypoint Localization Methods

### 1. Spatial Expectation vs Mode-Based Extraction

#### **Spatial Expectation** (`tapas_gmm/encoder/keypoints.py:1264-1276`)
```python
def get_spatial_expectation(self, softmax_activations):
    expected_x = torch.sum(softmax_activations * self.pos_x, dim=(2, 3))
    expected_y = torch.sum(softmax_activations * self.pos_y, dim=(2, 3))
    stacked_2d_features = torch.cat((expected_x, expected_y), 1)
    return stacked_2d_features
```

- **Method**: Computes weighted average position using probability distributions
- **Advantages**: Sub-pixel accuracy, differentiable, robust to noise
- **Usage**: Preferred for training neural networks

#### **Mode-Based Extraction** (`tapas_gmm/encoder/keypoints.py:1278-1301`)
```python
def get_mode(self, softmax_activations):
    s = softmax_activations.shape
    sm_flat = softmax_activations.view(s[0], s[1], -1)
    modes_flat = torch.argmax(sm_flat, dim=2)
    # Convert to 2D coordinates and normalize to [-1, 1]
    return stacked_2d_features
```

- **Method**: Finds pixel with maximum activation (argmax)
- **Advantages**: Discrete coordinates, computationally simple
- **Usage**: Used for final inference when discrete pixel coordinates are required

### 2. Descriptor Matching and Correspondence

#### **Reference Descriptor Selection**
Reference descriptors are established during initialization (`tapas_gmm/encoder/keypoints.py:794-963`):

```python
def select_reference_descriptors(self, dataset, traj_idx=0, img_idx=0, 
                                object_labels=None, cam="wrist"):
    ref_obs = dataset.sample_data_point_with_object_labels(
        cam=cam, img_idx=img_idx, traj_idx=traj_idx
    )
    # Extract descriptors and establish reference keypoints
```

#### **Softmax Correspondence Maps**
Keypoint localization uses softmax over descriptor distances (`tapas_gmm/encoder/keypoints.py:1199-1221`):

```python
def softmax_of_reference_descriptors(cls, descriptor_images, ref_descriptor, 
                                   taper=1, cosine=False):
    neg_squared_norm_diffs = cls.compute_reference_descriptor_distances(
        descriptor_images, ref_descriptor, cosine=cosine
    )
    softmax_activations = softmax(neg_squared_norm_diffs_flat * taper)
    return softmax_activations
```

## Trajectory Prediction Using Keypoints

### 1. Task-Parametrized Gaussian Mixture Models (TP-GMM)

#### **Keypoint-Based Task Parameters**
Keypoints serve as task parameters for TP-GMM (`tapas_gmm/utils/keypoints.py:30-44`):

```python
def tp_from_keypoints(kp: torch.Tensor, indeces: Sequence[int] | None) -> list[torch.Tensor]:
    unflattened_kp = unflatten_keypoints(kp)
    if indeces is None:
        selected_kp = unflattened_kp
    else:
        selected_kp = unflattened_kp[..., indeces, :]
    poses = poses_from_keypoints(selected_kp)
    return [p for p in poses.swapdims(0, 1)]
```

#### **Trajectory Generation Process**
1. **Keypoint Extraction**: Extract current keypoints from camera observations
2. **Task Parameter Computation**: Convert keypoints to task parameters for TP-GMM
3. **Trajectory Prediction**: Use TP-GMM to predict action sequence
4. **Motion Planning**: Apply TOPPRA for smooth trajectory execution

### 2. Multi-Modal Integration

#### **Observation Encoding**
Keypoints are integrated with other modalities (`tapas_gmm/encoder/keypoints.py:496-564`):

```python
def encode(self, batch: SceneObservation) -> tuple[torch.Tensor, dict]:
    camera_obs = batch.camera_obs
    rgb = tuple(o.rgb for o in camera_obs)
    depth = tuple(o.depth for o in camera_obs)
    extr = tuple(o.extr for o in camera_obs)
    intr = tuple(o.intr for o in camera_obs)
    
    # Compute keypoints with multi-camera support
    kp, info = self._encode(rgb, depth, extr, intr, prior, ...)
    return kp, info
```

#### **3D Projection and World Coordinates**
Keypoints are projected to 3D world coordinates for trajectory planning (`tapas_gmm/encoder/keypoints.py:683-709`):

```python
if projection == ProjectionTypes.UVD:
    kp = append_depth_to_uv(kp, depth, self.image_width - 1, self.image_height - 1)
elif projection in [ProjectionTypes.LOCAL_SOFT, ProjectionTypes.GLOBAL_SOFT]:
    kp = model_based_vision.soft_pixels_to_3D_world(
        kp, post, depth, extrinsics, intrinsics, 
        self.image_width - 1, self.image_height - 1
    )
```

## Filtering and Temporal Consistency

### 1. Particle Filter for Keypoint Tracking

#### **Configuration** (`conf/encoder/vit_keypoints/pf.py:23-31`)
```python
filter_config = ParticleFilterConfig(
    descriptor_distance_for_outside_pixels=(1,),
    filter_noise_scale=0.01,
    use_gripper_motion=True,
    gripper_motion_prob=0.25,
    sample_from_each_obs=True,
    return_spread=False,
    clip_projection=False,
)
```

#### **Temporal Consistency**
Particle filters maintain keypoint consistency across time steps (`tapas_gmm/filter/particle_filter.py:66-100`):

- **State Prediction**: Predict keypoint locations based on motion model
- **Observation Update**: Update predictions using current visual observations
- **Resampling**: Maintain particle diversity for robust tracking

### 2. Discrete Filter for Motion Modeling

#### **Motion Model Application** (`tapas_gmm/filter/discrete_filter.py:79-100`)
```python
def get_motion_model(self, depth_a, intr_a, extr_a, depth_b, intr_b, extr_b):
    # Compute pixel motion between frames
    return self._get_motion_model(depth_a, intr_a, extr_a, depth_b, intr_b, extr_b)
```

## Performance Characteristics

### 1. Computational Efficiency

#### **Descriptor Computation**
- **ViT Feature Extraction**: Dense features computed at reduced resolution (32x32 for 256x256 images)
- **Correspondence Search**: Efficient softmax computation over spatial dimensions
- **Multi-Camera Support**: Parallel processing across camera views

#### **Memory Optimization**
- **Pre-computed Encodings**: Support for disk-based descriptor storage
- **Batch Processing**: Efficient processing of multiple trajectory segments

### 2. Robustness Features

#### **Noise Handling**
- **Gaussian Noise Addition**: Optional noise injection for robustness (`tapas_gmm/encoder/keypoints.py:561-562`)
- **Threshold Filtering**: Distance-based keypoint filtering (`tapas_gmm/encoder/keypoints.py:621-626`)
- **Overshadowing**: Multi-camera consistency enforcement (`tapas_gmm/encoder/keypoints.py:609-617`)

#### **Failure Recovery**
- **Particle Filter Reset**: Automatic reinitialization for tracking failures
- **Reference Descriptor Updates**: Adaptive reference updating during long sequences

## Configuration and Usage

### 1. Key Configuration Parameters

#### **Encoder Configuration** (`conf/encoder/vit_keypoints/candidates_center.py:24-36`)
```python
encoder_config = VitKeypointsEncoderConfig(
    vit=vit_model_config,
    descriptor_dim=384,
    keypoints=kp_config,
    prior_type=PriorTypes.NONE,
    projection=ProjectionTypes.GLOBAL_HARD,
    taper_sm=25,
    cosine_distance=True,
    use_spatial_expectation=True,
    threshold_keypoint_dist=None,
    overshadow_keypoints=False,
    add_noise_scale=None,
)
```

#### **Keypoint Configuration** (`conf/encoder/vit_keypoints/candidates_center.py:13-15`)
```python
kp_config = KeypointsConfig(
    n_sample=len(obj_labels),  # Number of keypoints to extract
)
```

### 2. Training and Inference

#### **Training Pipeline**
1. **Pretraining**: Dense correspondence learning with contrastive loss
2. **Policy Training**: End-to-end training with TP-GMM policy
3. **Fine-tuning**: Task-specific adaptation

#### **Inference Pipeline**
1. **Keypoint Extraction**: Real-time keypoint detection from camera feeds
2. **Trajectory Prediction**: TP-GMM-based action sequence generation
3. **Execution**: Motion planning and robot control

## Conclusion

The TAPAS keypoint extraction system provides a comprehensive framework for learning and tracking semantically meaningful keypoints for robot manipulation. The system's strength lies in its:

1. **Unified Architecture**: Common pipeline for different keypoint types
2. **Robustness**: Multiple filtering and consistency mechanisms
3. **Flexibility**: Configurable extraction methods and projections
4. **Integration**: Seamless connection with TP-GMM trajectory prediction

The combination of ViT-based dense correspondence learning with task-parametrized trajectory prediction enables effective learning from few demonstrations while maintaining robustness to visual variations and environmental changes.