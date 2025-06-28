"""Constants and configuration for CLIP zero-shot classification modules."""

# Default configuration values
DEFAULT_TEMPLATE = "A photo of a {class_name}"
DEFAULT_MODEL_NAME = "openai/clip-vit-base-patch32"
DEFAULT_SLEEP_TIME = 0.5
CONTEXT_INIT_STD = 0.02

# Supported image file extensions for CoOp training
SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

# Default hyperparameters
DEFAULT_EMA_ALPHA = 0.3
DEFAULT_CONFIDENCE_THRESHOLD = 0.1
DEFAULT_STABILITY_FRAMES = 3
DEFAULT_COOP_N_CTX = 16
DEFAULT_COOP_EPOCHS = 50
DEFAULT_COOP_LR = 0.002
DEFAULT_SIMILARITY_THRESHOLD = 0.8
DEFAULT_MAX_OBJECT_MEMORY = 10
