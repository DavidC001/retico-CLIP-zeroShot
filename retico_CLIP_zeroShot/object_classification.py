"""
CLIP Object Classification Module

This module contains the CLIPObjectClassificationModule class that performs
zero-shot classification of detected objects using CLIP.
"""

import threading
import time
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from scipy.spatial.distance import cosine

import retico_core
from retico_vision.vision import ExtractedObjectsIU

from .base_CLIP import BaseCLIPModule
from .incremental_units import CLIPObjectFeaturesIU
from .constants import DEFAULT_TEMPLATE, DEFAULT_MODEL_NAME


class CLIPObjectClassificationModule(BaseCLIPModule):
    """A module that performs zero-shot classification of detected objects using CLIP."""

    @staticmethod
    def name() -> str:
        return "CLIP Object Classification Module"

    @staticmethod
    def description() -> str:
        return "A module that performs zero-shot classification of detected objects using CLIP"

    @staticmethod
    def input_ius() -> List:
        return [ExtractedObjectsIU]

    @staticmethod
    def output_iu() -> type:
        return CLIPObjectFeaturesIU

    def __init__(
        self, 
        class_labels: List[str], 
        template: str = DEFAULT_TEMPLATE, 
        model_name: str = DEFAULT_MODEL_NAME, 
        device: Optional[str] = None,
        ema_alpha: float = 0.3, 
        confidence_threshold: float = 0.1, 
        stability_frames: int = 3, 
        similarity_threshold: float = 0.8, 
        max_object_memory: int = 10, 
        debug_prints: bool = False,
        use_coop: bool = False, 
        coop_n_ctx: int = 16, 
        coop_ctx_init: str = "", 
        coop_examples_folder: Optional[str] = None,
        coop_epochs: int = 50, 
        coop_lr: float = 0.002, 
        **kwargs
    ) -> None:
        """Initialize the CLIP Object Classification Module."""
        super().__init__(
            class_labels, template, model_name, device, ema_alpha, confidence_threshold, 
            stability_frames, debug_prints, use_coop, coop_n_ctx, coop_ctx_init, 
            coop_examples_folder, coop_epochs, coop_lr, **kwargs
        )

        # Object tracking parameters
        self.similarity_threshold = similarity_threshold
        self.max_object_memory = max_object_memory
        
        # Object-specific state tracking (per consistent object ID)
        self.object_states: Dict[int, Dict] = {}
        self.object_features_memory: Dict[int, List[float]] = {}
        self.next_object_id: int = 0
        self.last_object_count: int = 0
        self.last_output_iu: Optional[CLIPObjectFeaturesIU] = None
        
        # CoOp name for logging
        self._coop_name = "Object CoOp"

    def _classifier_thread(self):
        while self._classifier_thread_active:
            if len(self.queue) == 0:
                time.sleep(0.5)
                continue

            input_iu = self.queue.popleft()
            image = input_iu.image
            detected_objects = input_iu.extracted_objects
            object_count = 0
            
            if image is None or detected_objects is None:
                continue
                
            try:
                object_features = []
                object_classifications = []
                object_classes = []
                object_confidences = []
                raw_probabilities = []  # Store raw probabilities for EMA
                
                # Process each object crop
                for i, obj in enumerate(detected_objects):
                    try:
                        crop = detected_objects[obj]
                        # Skip if image has width or height of 0
                        if (not isinstance(crop, Image.Image) or
                                crop.width == 0 or crop.height == 0):
                            object_features.append([])
                            object_classifications.append({})
                            object_classes.append("unknown")
                            object_confidences.append(0.0)
                            raw_probabilities.append(None)
                            continue
                        
                        object_count += 1
                        # Preprocess the crop with HuggingFace processor
                        inputs = self.processor(images=crop, return_tensors="pt").to(self.device)
                        
                        # Get image features
                        with torch.no_grad():
                            image_features = self.model.get_image_features(**inputs)
                            
                            # Validate image features
                            if image_features is None or image_features.numel() == 0:
                                print(f"Warning: Invalid image features for object {i}")
                                raise ValueError("Empty image features")
                                
                            # Normalize features
                            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                            
                            # Calculate similarities with precomputed text embeddings
                            logits = torch.matmul(image_features, self.text_embeddings.T) * self.model.logit_scale.exp()
                            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                        
                        # Store raw probabilities for EMA processing
                        raw_probabilities.append(probs)
                        
                        # Store results (will be updated with EMA later)
                        feature_vector = image_features.cpu().numpy().flatten().tolist()
                        if len(feature_vector) > 0:
                            object_features.append(feature_vector)
                        else:
                            object_features.append([])
                            if self.debug_prints: print(f"Warning: Empty feature vector for object {i}")
                        object_classifications.append({})  # Will be filled later
                        object_classes.append("")  # Will be filled later
                        object_confidences.append(0.0)  # Will be filled later
                        
                    except Exception as e:
                        print(f"Error processing object {i}: {e}")
                        # Add empty results for failed objects
                        object_features.append([])
                        object_classifications.append({})
                        object_classes.append("unknown")
                        object_confidences.append(0.0)
                        raw_probabilities.append(None)
                
                # Match objects to previous frame using feature similarity
                valid_features_count = len([f for f in object_features if f is not None and len(f) > 0])
                if self.debug_prints: print(f"Processing {object_count} objects with {valid_features_count} valid feature vectors")
                matched_ids = self._match_objects_to_previous(object_features)
                
                # Update object feature memory
                self._update_object_memory(matched_ids, object_features)
                
                # Apply EMA and determine update type with consistent object IDs
                update_type, final_results = self._process_object_states(raw_probabilities, matched_ids)
                
                if update_type is not None:
                    # Update results with EMA-smoothed values
                    for i, (ema_probs, top_class, confidence) in enumerate(final_results):
                        if ema_probs is not None:
                            object_classifications[i] = {
                                label: float(prob) for label, prob in zip(self.class_labels, ema_probs)
                            }
                            object_classes[i] = top_class
                            object_confidences[i] = confidence
                
                    # Create output IU
                    output_iu = self.create_iu(input_iu)
                    output_iu.set_object_classifications(
                        image, object_features, object_classifications, object_classes, object_confidences
                    )
                    
                    # Store the matched object IDs in the output IU for debugging/tracking
                    output_iu.object_ids = matched_ids
                    
                    
                    # Update state tracking
                    if update_type == retico_core.UpdateType.ADD or update_type == retico_core.UpdateType.COMMIT:
                        um = retico_core.UpdateMessage.from_iu(output_iu, update_type)
                        self.append(um)
                        self.last_output_iu = output_iu
                    elif update_type == retico_core.UpdateType.REVOKE:
                        self.revoke(self.last_output_iu)
                        self.last_output_iu = None

                self.last_object_count = object_count

            except Exception as e:
                print(f"Error processing object classification: {e}")
                continue

    def _process_object_states(self, raw_probabilities, matched_ids):
        """Process object states with EMA and determine update type using consistent object IDs."""
        current_object_count = len([p for p in raw_probabilities if p is not None])
        
        # Use matched object IDs as keys for state tracking instead of indices
        active_object_ids = set()
        final_results = []
        any_changes = False
        all_stable = True
        
        for i, (probs, object_id) in enumerate(zip(raw_probabilities, matched_ids)):
            if probs is None or object_id is None:
                final_results.append((None, "unknown", 0.0))
                continue
                
            active_object_ids.add(object_id)
            
            # Initialize object state if new
            if object_id not in self.object_states:
                self.object_states[object_id] = {
                    'ema_probabilities': probs.copy(),
                    'current_class': None,
                    'stable_frames_count': 0
                }
                any_changes = True
                if self.debug_prints: print(f"Initialized state for new object ID {object_id}")
            else:
                # Apply EMA smoothing
                self.object_states[object_id]['ema_probabilities'] = (
                    self.ema_alpha * probs + 
                    (1 - self.ema_alpha) * self.object_states[object_id]['ema_probabilities']
                )
            
            # Get classification results from smoothed probabilities
            ema_probs = self.object_states[object_id]['ema_probabilities']
            top_idx = np.argmax(ema_probs)
            top_class = self.class_labels[top_idx]
            confidence = float(ema_probs[top_idx])
            
            # Update object stability
            if confidence >= self.confidence_threshold:
                if self.object_states[object_id]['current_class'] == top_class:
                    self.object_states[object_id]['stable_frames_count'] += 1
                else:
                    self.object_states[object_id]['current_class'] = top_class
                    self.object_states[object_id]['stable_frames_count'] = 1
                    any_changes = True
                    if self.debug_prints: print(f"Object ID {object_id} changed class to {top_class} (confidence: {confidence:.3f})")
                
                # Check if this object is stable
                if self.object_states[object_id]['stable_frames_count'] < self.stability_frames:
                    all_stable = False
                elif self.object_states[object_id]['stable_frames_count'] == self.stability_frames:
                    if self.debug_prints: print(f"Object ID {object_id} is now stable as {top_class}")
            else:
                # Low confidence
                if self.object_states[object_id]['current_class'] is not None:
                    self.object_states[object_id]['current_class'] = None
                    self.object_states[object_id]['stable_frames_count'] = 0
                    any_changes = True
                    if self.debug_prints: print(f"Object ID {object_id} became unstable (low confidence: {confidence:.3f})")
                all_stable = False
            
            final_results.append((ema_probs, top_class, confidence))
        
        # Clean up states for objects that are no longer present
        objects_to_remove = []
        for object_id in self.object_states.keys():
            if object_id not in active_object_ids:
                objects_to_remove.append(object_id)
        
        for object_id in objects_to_remove:
            del self.object_states[object_id]
            if self.debug_prints: print(f"Removed state for disappeared object ID {object_id}")
            any_changes = True
        
        # Determine update type based on overall state
        if current_object_count == 0:
            if self.last_object_count > 0:
                if self.debug_prints: print("All objects disappeared - sending REVOKE")
                return retico_core.UpdateType.REVOKE, final_results
            else:
                return None, final_results
        elif self.last_object_count == 0:
            # First objects detected
            if self.debug_prints: print("First objects detected - sending ADD")
            return retico_core.UpdateType.ADD, final_results
        elif any_changes:
            # Objects changed
            if self.debug_prints: print("Object changes detected - sending ADD")
            return retico_core.UpdateType.ADD, final_results
        elif all_stable and all(self.object_states[obj_id]['stable_frames_count'] >= self.stability_frames 
                               for obj_id in active_object_ids 
                               if obj_id in self.object_states and self.object_states[obj_id]['current_class'] is not None):
            # All objects are stable
            if self.debug_prints: print("All objects stable - sending COMMIT")
            return retico_core.UpdateType.COMMIT, final_results
        else:
            # No significant changes
            return None, final_results
    
    def _clear_object_states(self):
        """Clear all object states."""
        self.object_states.clear()

    def prepare_run(self) -> None:
        """Initialize the CLIP model and start the classifier thread."""
        self._classifier_thread_active = True
        
        if not self._initialize_clip_model():
            return
            
        threading.Thread(target=self._classifier_thread, daemon=True).start()

    def _match_objects_to_previous(self, current_features):
        """Match current objects to previous objects using feature similarity."""
        if not self.object_features_memory or not current_features:
            # No previous objects or no current objects - assign new IDs
            matched_ids = []
            for i, features in enumerate(current_features):
                if features is not None:
                    object_id = self.next_object_id
                    self.next_object_id += 1
                    matched_ids.append(object_id)
                else:
                    matched_ids.append(None)
            return matched_ids
        
        matched_ids = []
        used_memory_ids = set()
        
        for current_features_vec in current_features:
            if current_features_vec is None or len(current_features_vec) == 0:
                matched_ids.append(None)
                continue
                
            best_match_id = None
            best_similarity = -1
            
            # Compare with all remembered objects
            for memory_id, memory_features in self.object_features_memory.items():
                if memory_id in used_memory_ids:
                    continue  # Already matched this memory object
                
                # Validate feature vectors before similarity calculation
                if (memory_features is None or len(memory_features) == 0 or 
                    len(memory_features) != len(current_features_vec)):
                    continue  # Skip invalid or mismatched feature vectors
                    
                # Calculate cosine similarity
                try:
                    similarity = 1 - cosine(current_features_vec, memory_features)
                    if similarity > best_similarity and similarity >= self.similarity_threshold:
                        best_similarity = similarity
                        best_match_id = memory_id
                except Exception as e:
                    print(f"Error calculating similarity between vectors of length {len(current_features_vec)} and {len(memory_features)}: {e}")
                    continue
            
            if best_match_id is not None:
                # Found a good match
                matched_ids.append(best_match_id)
                used_memory_ids.add(best_match_id)
                # print(f"Matched object with similarity {best_similarity:.3f} to ID {best_match_id}")
            else:
                # No good match found - assign new ID
                object_id = self.next_object_id
                self.next_object_id += 1
                matched_ids.append(object_id)
                # print(f"Assigned new object ID {object_id}")

        return matched_ids

    def _update_object_memory(self, matched_ids, current_features):
        """Update the object features memory with current features."""
        # Update features for matched objects
        for object_id, features in zip(matched_ids, current_features):
            if object_id is not None and features is not None and len(features) > 0:
                # Apply EMA to feature memory as well
                if object_id in self.object_features_memory:
                    # Validate stored features before EMA update
                    old_features = self.object_features_memory[object_id]
                    if (old_features is not None and len(old_features) > 0 and 
                        len(old_features) == len(features)):
                        # EMA update of feature memory
                        old_features = np.array(old_features)
                        new_features = np.array(features)
                        self.object_features_memory[object_id] = (
                            self.ema_alpha * new_features + 
                            (1 - self.ema_alpha) * old_features
                        ).tolist()
                    else:
                        # Replace with new features if old ones are invalid
                        self.object_features_memory[object_id] = features
                        print(f"Replaced invalid features for object ID {object_id}")
                else:
                    # New object
                    self.object_features_memory[object_id] = features
        
        # Clean up old objects if memory is too large
        if len(self.object_features_memory) > self.max_object_memory:
            # Remove oldest objects (simple strategy - could be improved)
            oldest_ids = sorted(self.object_features_memory.keys())
            for old_id in oldest_ids[:-self.max_object_memory]:
                del self.object_features_memory[old_id]
                if old_id in self.object_states:
                    del self.object_states[old_id]
                print(f"Removed old object ID {old_id} from memory")
