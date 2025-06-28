from retico_core import AbstractConsumingModule, UpdateType
from retico_clip_zeroShot.incremental_units import CLIPObjectFeaturesIU, CLIPClassificationIU

class ObjectClassificationConsumer(AbstractConsumingModule):
        @staticmethod
        def name():
            return "Object Classification Consumer"
        
        @staticmethod
        def description():
            return "Prints object classification results"
        
        @staticmethod
        def input_ius():
            return [CLIPObjectFeaturesIU]
        
        def process_update(self, update_message):
            for iu, ut in update_message:
                print(f"\n[{ut.name}] Object Classification Results:")
                print(f"Number of objects: {iu.num_objects}")
                for i, (obj_class, confidence) in enumerate(zip(iu.object_classes, iu.object_confidences)):
                    print(f"  Object {i+1}: {obj_class} (confidence: {confidence:.3f})")
    
# Create a simple consumer to print results
class ClassificationConsumer(AbstractConsumingModule):
    @staticmethod
    def name():
        return "Classification Consumer"
    
    @staticmethod
    def description():
        return "Prints classification results"
    
    @staticmethod
    def input_ius():
        return [CLIPClassificationIU]
    
    def process_update(self, update_message):
        for iu, ut in update_message:
            print(f"\n[{ut.name}] Classification Results:")
            print(f"Top Class: {iu.top_class}")
            print(f"Confidence: {iu.confidence:.3f}")
            print("Top 3 predictions:")
            sorted_probs = sorted(iu.class_probabilities.items(), 
                                key=lambda x: x[1], reverse=True)[:3]
            for label, prob in sorted_probs:
                print(f"  {label}: {prob:.3f}")