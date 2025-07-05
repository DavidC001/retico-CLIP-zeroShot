# retico-CLIP-zeroShot

Retico modules for zero-shot classification of scenes and objects using CLIP models.

## Overview

The `retico-CLIP-zeroShot` module provides functionality for zero-shot classification of scenes and objects using CLIP models. 
It leverages the power of CLIP to understand and classify images without the need for extensive retraining, making it ideal for dynamic environments and real-time applications.
If needed, it also allow to define a small set of training examples to fine-tune the predictions using the [CoOp](https://arxiv.org/abs/2109.01134) method.

## Installation

### Step 1: Install retico dependencies
First, ensure you have the `retico-core` and `retico-vision` modules installed.
The `retico-vision` module needs to be installed and added to your Python path:
```bash
git clone https://github.com/retico-team/retico-vision.git
```
**Important**: Make sure to add the path to the `retico-vision` and `retico-core` libraries to your `PYTHONPATH` environment variable. This is required for the module to properly import the vision components.

### Step 2: Install the package

```bash
pip install git+https://github.com/retico-team/retico-CLIP-zeroShot.git
```

## Usage
For a basic example of how to use the `retico-CLIP-zeroShot` module, refer to the `example_scene_classification.py` and `example_object_classification.py` files inside the `examples` directory of the repository.
Note that you will also need to install and add to the environment the `retico-yolov11` module to provide the object detection capabilities for the object classification example.

## Configuration
Both the image and object classification modules can be configured with the following parameters:
- `class_labels`: List of class labels to classify against. For example, ["dog", "cat", "car"].
- `template`: Template for class labels with {class_name} placeholder. For example, "a photo of a {class_name}".
- `model_name`: HuggingFace CLIP model name
- `device`: Device to run the model on (auto-detect if None)
- `ema_alpha`: EMA smoothing factor (0.0-1.0, higher = more responsive) used to smooth the model predictions for stability between frames
- `confidence_threshold`: Minimum confidence threshold for class changes
- `stability_frames`: Number of frames a class must be stable before COMMIT
- `debug_prints`: Whether to print debug information
- `use_coop`: Whether to use Context Optimization (CoOp) to train the token embeddings for the class labels.
- `coop_n_ctx`: Number of context tokens for CoOp
- `coop_ctx_init`: Initial context text for CoOp (empty for random init)
- `coop_examples_folder`: Path to folder with training images for CoOp
- `coop_epochs`: Number of training epochs for CoOp
- `coop_lr`: Learning rate for CoOp training

The `CLIPImageClassificationModule` and `CLIPObjectClassificationModule` can be instantiated with these parameters to perform zero-shot classification on images and objects, respectively.

The `CLIPObjectClassificationModule` has an additional parameter:
- `max_object_memory`: Maximum number of objects to track in memory (default: 1000). This limits the number of objects the module can remember across frames.
- `similarity_threshold`: Cosine similarity threshold for matching objects across frames (default: 0.8). This controls how similar features must be to consider objects the same across frames.
Note that the module uses a greedy approach to match objects across frames, the order of processing is the same as the order of the DetectedObjectsIU input.

## Update Logic (ADD/COMMIT/REVOKE)

The retico-CLIP-zeroShot modules implement a sophisticated update logic system that ensures stable and reliable classification results in real-time scenarios. This system uses three types of updates: **ADD**, **COMMIT**, and **REVOKE**, based on confidence levels and temporal stability.

### Core Concepts

- **EMA Smoothing**: Both modules use Exponential Moving Average (EMA) to smooth predictions over time, controlled by the `ema_alpha` parameter
- **Confidence Threshold**: Predictions below `confidence_threshold` are considered unreliable
- **Stability Frames**: A class must remain stable for `stability_frames` consecutive frames before being committed

### Image Classification Update Logic

The `CLIPImageClassificationModule` tracks scene-level classifications with the following logic:

#### ADD (New Classification)
- **Trigger**: First time a class is predicted with confidence above threshold
- **Condition**: `confidence >= confidence_threshold` AND `current_class != predicted_class`
- **Action**: Sends new classification result to downstream modules

#### COMMIT (Stable Classification)
- **Trigger**: A class has remained stable for the required number of frames
- **Condition**: Same class predicted for `stability_frames` consecutive frames with sufficient confidence
- **Action**: Confirms the classification is stable and reliable

#### REVOKE (Unreliable Classification)
- **Trigger**: Confidence drops below threshold or significant class change
- **Conditions**:
  - `confidence < confidence_threshold` (low confidence)
  - Major class change detected
- **Action**: Withdraws previous classification, indicating uncertainty

#### Example Sequence:
```
Frame 1: "dog" (conf: 0.8) → ADD "dog"
Frame 2: "dog" (conf: 0.82) → (no update, building stability)
Frame 3: "dog" (conf: 0.85) → COMMIT "dog" (stable for 3 frames)
Frame 4: "cat" (conf: 0.7) → REVOKE "dog", then ADD "cat"
Frame 5: "unknown" (conf: 0.05) → REVOKE "cat"
```

### Object Classification Update Logic

The `CLIPObjectClassificationModule` handles multiple objects simultaneously with more complex tracking:

#### Object Tracking
- **Feature Matching**: Uses CLIP feature similarity to match objects across frames 
- **Persistent IDs**: Maintains consistent object IDs using `similarity_threshold`
- **Per-Object State**: Each object has independent EMA smoothing and stability tracking

#### ADD (New Objects or Changes)
- **Triggers**:
  - First objects detected in the scene
  - New objects appear
  - Existing objects change classification
- **Action**: Updates object classifications and IDs

#### COMMIT (All Objects Stable)
- **Trigger**: All detected objects have stable classifications
- **Condition**: Every object has maintained the same class for `stability_frames` consecutive frames
- **Action**: Confirms all object classifications are stable

#### REVOKE (Objects Disappeared or Unreliable)
- **Triggers**:
  - All objects disappear from the scene
  - Object classifications become unreliable (low confidence)
- **Action**: Withdraws object classifications

#### Memory Management
- **Object Memory**: Maintains feature vectors for up to `max_object_memory` objects
- **State Cleanup**: Automatically removes states for disappeared objects
- **Feature Similarity**: Uses cosine similarity to match objects across frames

#### Example Sequence:
```
Frame 1: [obj1: "dog" (conf: 0.8)] → ADD objects
Frame 2: [obj1: "dog" (conf: 0.82)] → (building stability)
Frame 3: [obj1: "dog" (conf: 0.85), obj2: "car" (conf: 0.9)] → ADD objects (new car detected)
Frame 4: [obj1: "dog" (conf: 0.87), obj2: "car" (conf: 0.91)] → (building stability)
Frame 5: [obj1: "dog" (conf: 0.86), obj2: "car" (conf: 0.89)] → COMMIT objects (all stable)
Frame 6: [] → REVOKE objects (all disappeared)
```

### Configuration Tips

- **Higher `ema_alpha`**: More responsive to changes, less stable
- **Lower `ema_alpha`**: More stable, slower to adapt
- **Higher `confidence_threshold`**: More conservative, fewer false positives
- **More `stability_frames`**: More stable output, slower response to changes
- **Higher `similarity_threshold`**: Stricter object matching, more new IDs assigned

### Project Structure

```
retico-CLIP-zeroShot/
├── retico_CLIP_zeroShot/
│   ├── __init__.py
│   ├── version.py              # Version information
│   ├── constants.py             # Constants used in the module
|   |
|   ├── incremental_units.py  # Incremental units for scene and object classification
|   |
│   ├── base_CLIP.py            # Base class for CLIP modules
│   ├── CoOp_encoder.py         # CoOp encoder for fine-tuning CLIP
|   |
│   ├── image_classification.py  # Image classification using CLIP
|   ├── object_classification.py  # Object classification using CLIP
|   |
│   └── debug_modules.py       # Debugging modules to print the output IUs from the modules
│
├── examples/
│   ├── example_scene_classification.py  # Example for scene classification
│   └── example_object_classification.py # Example for object classification
|
├── setup.py                    # Package setup
├── README.md                   # This file
└── LICENSE                     # License file
```

## Related Projects

- [ReTiCo Core](https://github.com/retico-team/retico-core) - The core ReTiCo framework
- [ReTiCo Vision](https://github.com/retico-team/retico-vision) - Vision components for ReTiCo
- [Hugging Face Transformers](https://github.com/huggingface/transformers) - The transformers library used for CLIP
