from retico_core import *
from retico_vision.vision import WebcamModule 

from retico_CLIP_zeroShot.image_classification import CLIPImageClassificationModule
from retico_CLIP_zeroShot.debug_modules import ClassificationConsumer

video = WebcamModule()

template = "A scene from {class_name}."
classes = ["outdoor", "indoor", "kitchen", "bedroom", "living room",
           "office", "bathroom", "restaurant", "street", "park", 
           "beach", "forest", "urban area", "rural area", "cityscape"]
classifier = CLIPImageClassificationModule(
    template=template,
    class_labels=classes,
    model_name="openai/clip-vit-base-patch32",  # Use HuggingFace model name
    ema_alpha=0.3,  # EMA smoothing factor
    confidence_threshold=0.15,  # Minimum confidence for classifications
    stability_frames=3  # Frames required for stability
)
debug_classifier = ClassificationConsumer()


video.subscribe(classifier)
classifier.subscribe(debug_classifier)

network.run(video)

input("Network is running. Press Enter to stop...")

network.stop(video)

print("Network stopped.")
