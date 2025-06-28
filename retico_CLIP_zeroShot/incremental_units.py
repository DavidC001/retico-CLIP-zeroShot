"""Incremental Units for CLIP classification results."""

import datetime
import json
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

import retico_core
from retico_vision.vision import ObjectFeaturesIU


class CLIPClassificationIU(retico_core.IncrementalUnit):
    """An incremental unit that contains CLIP classification results for an image.

    Attributes:
        creator: The module that created this IU
        previous_iu: A link to the IU created before the current one
        grounded_in: A link to the IU this IU is based on
        created_at: The UNIX timestamp of the moment the IU is created
        image: The original image
        classifications: Dictionary containing classification results
        top_class: The class with highest confidence
        confidence: Confidence score for the top class
        class_probabilities: Dictionary mapping class names to probabilities
    """

    @staticmethod
    def type() -> str:
        return "CLIP Classification IU"

    def __init__(
        self,
        creator: Optional[retico_core.AbstractModule] = None,
        iuid: int = 0,
        previous_iu: Optional[retico_core.IncrementalUnit] = None,
        grounded_in: Optional[retico_core.IncrementalUnit] = None,
        **kwargs
    ) -> None:
        super().__init__(
            creator=creator,
            iuid=iuid,
            previous_iu=previous_iu,
            grounded_in=grounded_in,
            payload=None
        )
        self.image: Optional[Image.Image] = None
        self.classifications: Dict = {}
        self.top_class: Optional[str] = None
        self.confidence: float = 0.0
        self.class_probabilities: Dict[str, float] = {}

    def set_classification_results(
        self, 
        image: Image.Image, 
        classifications: Dict, 
        top_class: str, 
        confidence: float, 
        class_probabilities: Dict[str, float]
    ) -> None:
        """Set the classification results for the IU."""
        self.image = image
        self.payload = classifications
        self.classifications = classifications
        self.top_class = top_class
        self.confidence = confidence
        self.class_probabilities = class_probabilities

    def to_zmq(self, update_type: retico_core.UpdateType) -> Dict:
        """Return a formatted string that can be sent across zeromq."""
        payload = {
            "originatingTime": datetime.datetime.now().isoformat(),
            "update_type": str(update_type)
        }
        
        message = {
            'image': np.array(self.image).tolist() if self.image else None,
            'classifications': self.classifications,
            'top_class': self.top_class,
            'confidence': float(self.confidence),
            'class_probabilities': self.class_probabilities
        }
        payload["message"] = json.dumps(message)
        return payload

    def from_zmq(self, zmq_data: Dict) -> None:
        """Load data from zmq message."""
        zmq_data = json.loads(zmq_data['message'])
        
        if zmq_data['image']:
            self.image = Image.fromarray(np.array(zmq_data['image'], dtype='uint8'))
        
        self.classifications = zmq_data['classifications']
        self.payload = self.classifications
        self.top_class = zmq_data['top_class']
        self.confidence = zmq_data['confidence']
        self.class_probabilities = zmq_data['class_probabilities']


class CLIPObjectFeaturesIU(ObjectFeaturesIU):
    """An incremental unit that extends ObjectFeaturesIU with CLIP classification results.

    Attributes:
        creator: The module that created this IU
        previous_iu: A link to the IU created before the current one
        grounded_in: A link to the IU this IU is based on
        created_at: The UNIX timestamp of the moment the IU is created
        image: The original image
        object_features: List of feature vectors for each object
        object_classifications: List of classification results for each object
        object_classes: List of top predicted classes for each object
        object_confidences: List of confidence scores for each object
        num_objects: Number of classified objects
    """

    @staticmethod
    def type() -> str:
        return "CLIP Object Features IU"

    def __init__(
        self,
        creator: Optional[retico_core.AbstractModule] = None,
        iuid: int = 0,
        previous_iu: Optional[retico_core.IncrementalUnit] = None,
        grounded_in: Optional[retico_core.IncrementalUnit] = None,
        **kwargs
    ) -> None:
        super().__init__(
            creator=creator,
            iuid=iuid,
            previous_iu=previous_iu,
            grounded_in=grounded_in,
            **kwargs
        )
        self.object_classifications: List[Dict] = []
        self.object_classes: List[str] = []
        self.object_confidences: List[float] = []

    def set_object_classifications(
        self, 
        image: Image.Image, 
        object_features: List, 
        object_classifications: List[Dict], 
        object_classes: List[str], 
        object_confidences: List[float]
    ) -> None:
        """Set the classification results for detected objects."""
        self.image = image
        self.object_features = object_features
        self.payload = object_features
        self.object_classifications = object_classifications
        self.object_classes = object_classes
        self.object_confidences = object_confidences
        self.num_objects = len(object_classifications)

    def to_zmq(self, update_type: retico_core.UpdateType) -> Dict:
        """Return a formatted string that can be sent across zeromq."""
        payload = {
            "originatingTime": datetime.datetime.now().isoformat(),
            "update_type": str(update_type)
        }
        
        message = {
            'image': np.array(self.image).tolist() if self.image else None,
            'object_features': self.object_features,
            'object_classifications': self.object_classifications,
            'object_classes': self.object_classes,
            'object_confidences': [float(conf) for conf in self.object_confidences],
            'num_objects': self.num_objects
        }
        payload["message"] = json.dumps(message)
        return payload

    def from_zmq(self, zmq_data: Dict) -> None:
        """Load data from zmq message."""
        zmq_data = json.loads(zmq_data['message'])
        
        if zmq_data['image']:
            self.image = Image.fromarray(np.array(zmq_data['image'], dtype='uint8'))
        
        self.object_features = zmq_data['object_features']
        self.payload = self.object_features
        self.object_classifications = zmq_data['object_classifications']
        self.object_classes = zmq_data['object_classes']
        self.object_confidences = zmq_data['object_confidences']
        self.num_objects = zmq_data['num_objects']
