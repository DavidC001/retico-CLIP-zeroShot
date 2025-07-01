"""
CLIP Image Classification Module

This module contains the CLIPImageClassificationModule class that performs
zero-shot image classification using CLIP.
"""

import threading
import time
from typing import List, Dict, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

import retico_core
from retico_vision.vision import ImageIU

from .base_CLIP import BaseCLIPModule
from .incremental_units import CLIPClassificationIU
from .constants import DEFAULT_TEMPLATE, DEFAULT_MODEL_NAME, DEFAULT_SLEEP_TIME


class CLIPImageClassificationModule(BaseCLIPModule):
    """A module that performs zero-shot image classification using CLIP."""

    @staticmethod
    def name() -> str:
        return "CLIP Image Classification Module"

    @staticmethod
    def description() -> str:
        return "A module that performs zero-shot image classification using CLIP"

    @staticmethod
    def input_ius() -> List:
        return [ImageIU]

    @staticmethod
    def output_iu() -> type:
        return CLIPClassificationIU

    def __init__(
        self, 
        class_labels: List[str], 
        template: str = DEFAULT_TEMPLATE, 
        model_name: str = DEFAULT_MODEL_NAME, 
        device: Optional[str] = None, 
        ema_alpha: float = 0.3, 
        confidence_threshold: float = 0.1, 
        stability_frames: int = 3, 
        debug_prints: bool = False,
        use_coop: bool = False, 
        coop_n_ctx: int = 16, 
        coop_ctx_init: str = "", 
        coop_examples_folder: Optional[str] = None,
        coop_epochs: int = 50, 
        coop_lr: float = 0.002, 
        **kwargs
    ) -> None:
        """Initialize the CLIP Image Classification Module."""
        super().__init__(
            class_labels, template, model_name, device, ema_alpha, confidence_threshold, 
            stability_frames, debug_prints, use_coop, coop_n_ctx, coop_ctx_init, 
            coop_examples_folder, coop_epochs, coop_lr, **kwargs
        )
        
        # Image-specific state tracking for stability
        self.ema_probabilities: Optional[np.ndarray] = None
        self.current_class: Optional[str] = None
        self.stable_frames_count: int = 0
        self.last_output_iu: Optional[CLIPClassificationIU] = None
        
        # CoOp name for logging
        self._coop_name = "Image CoOp"

    def _classifier_thread(self) -> None:
        """Main classification thread for processing images."""
        while self._classifier_thread_active:
            if len(self.queue) == 0:
                time.sleep(DEFAULT_SLEEP_TIME)
                continue

            input_iu = self.queue.popleft()
            
            if input_iu.image is None:
                continue
                
            try:
                # Process the image and get classification results
                classification_results = self._classify_image(input_iu.image)
                
                if classification_results:
                    top_class, confidence, class_probabilities, classifications = classification_results
                    
                    # Determine update type based on stability
                    update_type = self._determine_update_type(top_class, confidence)
                    
                    if update_type is not None:
                        self._send_classification_update(
                            input_iu, input_iu.image, classifications, 
                            top_class, confidence, class_probabilities, update_type
                        )
                
            except Exception as e:
                print(f"Error processing image classification: {e}")
                continue
    
    def _classify_image(self, image: Union[Image.Image, np.ndarray]) -> Optional[Tuple]:
        """Classify a single image and return results."""
        # Preprocess image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Process image with HuggingFace processor
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        # Get image features and calculate probabilities
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
            logits = torch.matmul(image_features, self.text_embeddings.T) * self.model.logit_scale.exp()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        
        # Apply EMA smoothing
        self._update_ema_probabilities(probs)
        
        # Get classification results from smoothed probabilities
        top_idx = np.argmax(self.ema_probabilities)
        top_class = self.class_labels[top_idx]
        confidence = float(self.ema_probabilities[top_idx])
        
        # Create probability dictionary and classification results
        class_probabilities = {
            label: float(prob) for label, prob in zip(self.class_labels, self.ema_probabilities)
        }
        
        classifications = {
            "predicted_class": top_class,
            "confidence": confidence,
            "all_probabilities": class_probabilities
        }
        
        return top_class, confidence, class_probabilities, classifications
    
    def _update_ema_probabilities(self, probs: np.ndarray) -> None:
        """Update EMA probabilities with new observation."""
        if self.ema_probabilities is None:
            self.ema_probabilities = probs.copy()
        else:
            self.ema_probabilities = (
                self.ema_alpha * probs + 
                (1 - self.ema_alpha) * self.ema_probabilities
            )
    
    def _send_classification_update(
        self, 
        input_iu: ImageIU, 
        image: Image.Image, 
        classifications: Dict, 
        top_class: str, 
        confidence: float, 
        class_probabilities: Dict[str, float], 
        update_type: retico_core.UpdateType
    ) -> None:
        """Send classification update message."""
        output_iu = self.create_iu(input_iu)
        output_iu.set_classification_results(
            image, classifications, top_class, confidence, class_probabilities
        )
        
        
        # Update state tracking
        if update_type in [retico_core.UpdateType.ADD, retico_core.UpdateType.COMMIT]:
            um = retico_core.UpdateMessage.from_iu(output_iu, update_type)
            self.append(um)
            self.last_output_iu = output_iu
        elif update_type == retico_core.UpdateType.REVOKE:
            self.revoke(self.last_output_iu)
            self.last_output_iu = None

    def _determine_update_type(self, predicted_class: str, confidence: float) -> Optional[retico_core.UpdateType]:
        """Determine the appropriate update type based on classification stability."""
        
        # Check if confidence is above threshold
        if confidence < self.confidence_threshold:
            return self._handle_low_confidence()
        
        # High confidence classification
        if self.current_class == predicted_class:
            return self._handle_same_class()
        else:
            return self._handle_class_change(predicted_class)
    
    def _handle_low_confidence(self) -> Optional[retico_core.UpdateType]:
        """Handle low confidence classification."""
        if self.current_class is not None:
            self.current_class = None
            self.stable_frames_count = 0
            return retico_core.UpdateType.REVOKE
        return None
    
    def _handle_same_class(self) -> Optional[retico_core.UpdateType]:
        """Handle classification with same class as before."""
        self.stable_frames_count += 1
        
        if self.stable_frames_count == 1:
            # First time seeing this class - ADD
            return retico_core.UpdateType.ADD
        elif self.stable_frames_count >= self.stability_frames:
            # Class has been stable long enough - COMMIT
            self.stable_frames_count = self.stability_frames  # Cap the counter
            return retico_core.UpdateType.COMMIT
        else:
            # Still building stability - no update
            return None
    
    def _handle_class_change(self, predicted_class: str) -> retico_core.UpdateType:
        """Handle classification with different class."""
        revoke_needed = self.current_class is not None
        
        # Update to new class
        self.current_class = predicted_class
        self.stable_frames_count = 1
        
        if revoke_needed:
            # Send REVOKE for old class first, then we'll ADD the new one next frame
            return retico_core.UpdateType.REVOKE
        else:
            # No previous class - directly ADD
            return retico_core.UpdateType.ADD

    def prepare_run(self) -> None:
        """Initialize the CLIP model and start the classifier thread."""
        self._classifier_thread_active = True
        
        if not self._initialize_clip_model():
            return
            
        threading.Thread(target=self._classifier_thread, daemon=True).start()
