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

### Project Structure

```
retico-CLIP-zeroShot/
├── retico_CLIP_zeroShot/
│   ├── __init__.py
│   ├── objects_feat_extr.py    # Main feature extraction module
│   ├── version.py              # Version information
│   └── test_module.py          # Test utilities
├── examples/
│   ├── example_scene_classification.py  # Example for scene classification
│   └── example_object_classification.py # Example for object classification
├── setup.py                    # Package setup
├── README.md                   # This file
└── LICENSE                     # License file
```

## Related Projects

- [ReTiCo Core](https://github.com/retico-team/retico-core) - The core ReTiCo framework
- [ReTiCo Vision](https://github.com/retico-team/retico-vision) - Vision components for ReTiCo
- [Hugging Face Transformers](https://github.com/huggingface/transformers) - The transformers library used for vision models
