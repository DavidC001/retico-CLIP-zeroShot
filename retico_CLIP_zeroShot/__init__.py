"""
CLIP Zero-Shot Classification Module for Retico

This package provides zero-shot image and object classification using CLIP models
with optional Context Optimization (CoOp) for improved performance.
"""

# Import main modules
from .image_classification import CLIPImageClassificationModule
from .object_classification import CLIPObjectClassificationModule

# Import incremental units
from .incremental_units import CLIPClassificationIU, CLIPObjectFeaturesIU

# Import base functionality (for advanced users)
from .base_CLIP import BaseCLIPModule
from .CoOp_encoder import CoOpTextEncoder

# Import constants (for configuration)
from .constants import (
    DEFAULT_TEMPLATE,
    DEFAULT_MODEL_NAME,
    DEFAULT_SLEEP_TIME,
    CONTEXT_INIT_STD,
    SUPPORTED_IMAGE_EXTENSIONS
)

# Version information
from .version import __version__
