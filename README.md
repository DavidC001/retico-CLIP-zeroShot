# retico-CLIP-zeroShot

Retico modules for zero-shot classification of scenes and objects using CLIP models.

## Overview

The `retico-CLIP-zeroShot` module provides functionality for zero-shot classification of scenes and objects using CLIP models. 
It leverages the power of CLIP to understand and classify images without the need for extensive retraining, making it ideal for dynamic environments and real-time applications.
If needed, it also allow to define a small set of training examples to fine-tune the predictions using the [CoOp](https://arxiv.org/abs/2109.01134) method.

## Installation

### Step 1: Install the package

```bash
pip install git+https://github.com/retico-team/retico-CLIP-zeroShot.git
```

### Step 2: Install retico-vision dependency
Since this module depends on `retico-vision`, you need to install it and add it to your Python path:
```bash
git clone https://github.com/retico-team/retico-vision.git
```
**Important**: Make sure to add the path to the `retico-vision` library to your `PYTHONPATH` environment variable. This is required for the module to properly import the vision components.

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
