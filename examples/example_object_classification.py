from retico_core import *
from retico_vision.vision import WebcamModule, ExtractObjectsModule 

from retico_yolov11.yolov11 import Yolov11

from retico_CLIP_zeroShot.object_classification import CLIPObjectClassificationModule
from retico_CLIP_zeroShot.debug_modules import ObjectClassificationConsumer

video = WebcamModule()
yolo = Yolov11() 
extractor = ExtractObjectsModule(num_obj_to_display=5)

object_classifier = CLIPObjectClassificationModule(
    model_name="openai/clip-vit-base-patch32",  # Use HuggingFace model name
    template="A photo of a {class_name}.",
    class_labels=["cat", "dog", "car", "bicycle", "person", "mouse", "keyboard", "phone", "laptop", "book"],
    ema_alpha=0.4,  # More responsive for objects
    confidence_threshold=0.2,  # Minimum confidence for classifications
    stability_frames=2,  # Frames required for stability
    debug_prints=False,  # Enable debug prints for object classification
)
debug_obj_classifier = ObjectClassificationConsumer()


video.subscribe(yolo)
yolo.subscribe(extractor)
extractor.subscribe(object_classifier)
object_classifier.subscribe(debug_obj_classifier)


network.run(video)

input("Network is running. Press Enter to stop...")

network.stop(video)

print("Network stopped.")
