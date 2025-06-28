"""Base module with shared CLIP functionality."""

import glob
import os
import threading
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

import retico_core

from .constants import (
    DEFAULT_TEMPLATE, DEFAULT_MODEL_NAME, DEFAULT_EMA_ALPHA,
    DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_STABILITY_FRAMES,
    DEFAULT_COOP_N_CTX, DEFAULT_COOP_EPOCHS, DEFAULT_COOP_LR,
    SUPPORTED_IMAGE_EXTENSIONS
)
from .CoOp_encoder import CoOpTextEncoder


class BaseCLIPModule(retico_core.AbstractModule):
    """Base class for CLIP modules with shared functionality."""
    
    def __init__(
        self, 
        class_labels: List[str], 
        template: str = DEFAULT_TEMPLATE, 
        model_name: str = DEFAULT_MODEL_NAME, 
        device: Optional[str] = None,
        ema_alpha: float = DEFAULT_EMA_ALPHA, 
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD, 
        stability_frames: int = DEFAULT_STABILITY_FRAMES, 
        debug_prints: bool = False,
        use_coop: bool = False, 
        coop_n_ctx: int = DEFAULT_COOP_N_CTX, 
        coop_ctx_init: str = "", 
        coop_examples_folder: Optional[str] = None, 
        coop_epochs: int = DEFAULT_COOP_EPOCHS, 
        coop_lr: float = DEFAULT_COOP_LR, 
        **kwargs
    ) -> None:
        """Initialize the base CLIP module.
        
        Args:
            class_labels: List of class labels to classify against
            template: Template for class labels with {class_name} placeholder
            model_name: HuggingFace CLIP model name
            device: Device to run the model on (auto-detect if None)
            ema_alpha: EMA smoothing factor (0.0-1.0, higher = more responsive)
            confidence_threshold: Minimum confidence threshold for class changes
            stability_frames: Number of frames a class must be stable before COMMIT
            debug_prints: Whether to print debug information
            use_coop: Whether to use Context Optimization (CoOp) for improved prompts
            coop_n_ctx: Number of context tokens for CoOp
            coop_ctx_init: Initial context text for CoOp (empty for random init)
            coop_examples_folder: Path to folder with training images for CoOp
            coop_epochs: Number of training epochs for CoOp
            coop_lr: Learning rate for CoOp training
        """
        super().__init__(**kwargs)
        
        # Core parameters
        self.class_labels = class_labels
        self.template = template
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.debug_prints = debug_prints
        
        # Processing queue
        self.queue = deque(maxlen=1)
        
        # EMA and stability parameters
        self.ema_alpha = ema_alpha
        self.confidence_threshold = confidence_threshold
        self.stability_frames = stability_frames
        
        # CoOp parameters
        self.use_coop = use_coop
        self.coop_n_ctx = coop_n_ctx
        self.coop_ctx_init = coop_ctx_init
        self.coop_epochs = coop_epochs
        self.coop_lr = coop_lr
        
        # Load CoOp examples from folder
        self.coop_examples: Dict[str, List[Image.Image]] = {}
        if coop_examples_folder:
            self.coop_examples = self._load_coop_examples_from_folder(coop_examples_folder)
        
        # Initialize CLIP model components
        self.model: Optional[CLIPModel] = None
        self.processor: Optional[CLIPProcessor] = None
        self.text_embeddings: Optional[torch.Tensor] = None
        self.coop_text_encoder: Optional[CoOpTextEncoder] = None
        self._classifier_thread_active = False

    def process_update(self, update_message: retico_core.UpdateMessage) -> None:
        """Process incoming update messages."""
        for iu, ut in update_message:
            if ut == retico_core.UpdateType.ADD:
                self.queue.append(iu)


    def _load_coop_examples_from_folder(self, folder_path: str) -> Dict[str, List[Image.Image]]:
        """Load CoOp training examples from a folder structure.
        
        Expected folder structure:
        folder_path/
        ├── class1/
        │   ├── img1.jpg
        │   ├── img2.png
        │   └── ...
        ├── class2/
        │   ├── img3.jpg
        │   └── ...
        └── ...
        
        Args:
            folder_path: Path to the base folder containing class subfolders
            
        Returns:
            Dictionary mapping class names to lists of PIL Images
        """
        if not os.path.exists(folder_path):
            print(f"Warning: CoOp examples folder '{folder_path}' does not exist")
            return {}
        
        coop_examples = {}
        
        # Check each class label
        for class_label in self.class_labels:
            class_folder = os.path.join(folder_path, class_label)
            
            if not os.path.exists(class_folder) or not os.path.isdir(class_folder):
                if self.debug_prints:
                    print(f"Warning: No folder found for class '{class_label}' at '{class_folder}'")
                continue
            
            # Load all supported image files from the class folder
            class_images = self._load_images_from_folder(class_folder, class_label)
            
            if class_images:
                coop_examples[class_label] = class_images
                print(f"Loaded {len(class_images)} images for class '{class_label}'")
            elif self.debug_prints:
                print(f"Warning: No valid images found for class '{class_label}' in '{class_folder}'")
        
        total_images = sum(len(images) for images in coop_examples.values())
        print(f"Total CoOp training images loaded: {total_images} across {len(coop_examples)} classes")
        
        return coop_examples
    
    def _load_images_from_folder(self, class_folder: str, class_label: str) -> List[Image.Image]:
        """Load images from a specific class folder."""
        class_images = []
        
        for ext in SUPPORTED_IMAGE_EXTENSIONS:
            # Use glob to find all files with this extension (case insensitive)
            for pattern in [f"*{ext}", f"*{ext.upper()}"]:
                files = glob.glob(os.path.join(class_folder, pattern), recursive=False)
                
                for img_path in files:
                    try:
                        img = Image.open(img_path)
                        if img:
                            class_images.append(img)
                            if self.debug_prints:
                                print(f"Loaded image: {img_path} ({img.width}x{img.height})")
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")
        
        return class_images


    def _train_coop(self) -> None:
        """Train CoOp context vectors using provided examples."""
        if not self.coop_examples:
            if self.debug_prints:
                print("Warning: No CoOp examples provided, skipping CoOp training")
            return
        
        print(f"Training CoOp with {self.coop_epochs} epochs...")
        
        # Prepare training data
        train_images, train_labels = self._prepare_training_data()
        
        if not train_images:
            print("Warning: No valid training images found for CoOp")
            return
        
        # Set up training components
        optimizer = torch.optim.AdamW([self.coop_text_encoder.ctx], lr=self.coop_lr)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self._run_training_loop(train_images, train_labels, optimizer, criterion)
        
        # Finalize training
        self._finalize_coop_training()
    
    def _prepare_training_data(self) -> Tuple[List[Image.Image], torch.Tensor]:
        """Prepare training data for CoOp."""
        train_images = []
        train_labels = []
        
        for class_name, examples in self.coop_examples.items():
            if class_name in self.class_labels:
                class_idx = self.class_labels.index(class_name)
                for img in examples:
                    train_images.append(img)
                    train_labels.append(class_idx)
        
        return train_images, torch.tensor(train_labels, device=self.device)
    
    def _run_training_loop(
        self, 
        train_images: List[Image.Image], 
        train_labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ) -> None:
        """Run the CoOp training loop."""
        self.coop_text_encoder.train()
        for epoch in tqdm(range(self.coop_epochs)):
            total_loss = 0
            correct = 0
            
            # Shuffle training data
            indices = torch.randperm(len(train_images))
            
            for idx in indices:
                loss, is_correct = self._train_single_sample(
                    train_images[idx], train_labels[idx], optimizer, criterion
                )
                total_loss += loss
                correct += is_correct
            
            # Log progress
            avg_loss = total_loss / len(train_images)
            accuracy = correct / len(train_images)
            tqdm.write(f"Epoch {epoch + 1}/{self.coop_epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    def _train_single_sample(
        self, 
        image: Image.Image, 
        label: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ) -> Tuple[float, int]:
        """Train on a single sample."""
        optimizer.zero_grad()
        
        # Get image features
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        
        # Get text features from CoOp
        text_features = self.coop_text_encoder()
        
        # Calculate logits and loss
        logits = torch.matmul(image_features, text_features.T) * self.model.logit_scale.exp()
        loss = criterion(logits, label.unsqueeze(0))
        
        loss.backward()
        optimizer.step()
        
        # Check if prediction is correct
        pred = logits.argmax(dim=1)
        is_correct = (pred == label).item()
        
        return loss.item(), is_correct
    
    def _finalize_coop_training(self) -> None:
        """Finalize CoOp training and update text embeddings."""
        self.coop_text_encoder.eval()
        with torch.no_grad():
            self.text_embeddings = self.coop_text_encoder()
        
        print("CoOp training completed!")


    def _initialize_clip_model(self) -> bool:
        """Initialize the CLIP model and text embeddings."""
        try:
            # Load HuggingFace CLIP model and processor
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            
            if self.use_coop:
                self._initialize_coop()
            else:
                self._initialize_standard_text_embeddings()
            
            print(f"HuggingFace CLIP model {self.model_name} loaded successfully on {self.device}")
            print(f"Class labels: {self.class_labels}")
            
            return True
            
        except Exception as e:
            print(f"Error loading CLIP model: {e}")
            self.model = None
            return False
    
    def _initialize_coop(self) -> None:
        """Initialize CoOp text encoder."""
        coop_name = getattr(self, '_coop_name', 'CoOp')
        print(f"Initializing {coop_name} text encoder...")
        
        self.coop_text_encoder = CoOpTextEncoder(
            self.model, 
            self.class_labels,
            template=self.template,
            n_ctx=self.coop_n_ctx,
            ctx_init=self.coop_ctx_init,
            device=self.device
        ).to(self.device)
        
        # Initialize the CoOp encoder with the processor
        self.coop_text_encoder.initialize_with_processor(self.processor)
        
        # Train CoOp if examples are provided
        if self.coop_examples:
            self._train_coop()
        else:
            # Get initial text embeddings from CoOp
            self.coop_text_encoder.eval()
            with torch.no_grad():
                self.text_embeddings = self.coop_text_encoder()
        
        print(f"{coop_name} initialized with {self.coop_n_ctx} context tokens")
        print(f"Text embeddings shape: {self.text_embeddings.shape}")
    
    def _initialize_standard_text_embeddings(self) -> None:
        """Initialize standard CLIP text embeddings."""
        text_inputs = [self.template.format(class_name=label) for label in self.class_labels]
        text_tokens = self.processor(text=text_inputs, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**text_tokens)
            # Normalize text features
            self.text_embeddings = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        
        print(f"Standard CLIP text embeddings shape: {self.text_embeddings.shape}")


    def shutdown(self) -> None:
        """Shutdown the module."""
        self._classifier_thread_active = False
