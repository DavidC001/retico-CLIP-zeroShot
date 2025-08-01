"""Context Optimization (CoOp) text encoder for CLIP."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from .constants import DEFAULT_TEMPLATE, CONTEXT_INIT_STD


class CoOpTextEncoder(nn.Module):
    """Context Optimization (CoOp) for CLIP text encoder.
    
    This module learns a small set of context tokens that are prepended to class names
    to improve classification performance. Based on the paper:
    "Learning to Prompt for Vision-Language Models" (Zhou et al., 2022)
    """
    
    def __init__(
        self, 
        clip_model: CLIPModel, 
        class_names: List[str], 
        template: str = DEFAULT_TEMPLATE, 
        n_ctx: int = 16, 
        ctx_init: str = "", 
        device: str = "cuda"
    ) -> None:
        """Initialize CoOp text encoder.
        
        Args:
            clip_model: The CLIP model
            class_names: List of class names
            template: Template for class labels with {class_name} placeholder
            n_ctx: Number of context tokens to learn
            ctx_init: Initialization text for context
            device: Device to run on
        """
        super().__init__()
        
        self.clip_model = clip_model
        self.device = device
        self.n_ctx = n_ctx
        self.class_names = class_names
        self.template = template
        self.n_cls = len(class_names)
        
        # make sure the model is in evaluation mode
        self.clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad_(False)

        
        # Get text encoder components from CLIP
        self.text_encoder = clip_model.text_model
        self.text_projection = clip_model.text_projection
        self.token_embedding = self.text_encoder.embeddings.token_embedding
        self.positional_embedding = self.text_encoder.embeddings.position_embedding
        
        # Initialize context vectors
        embed_dim = self.token_embedding.embedding_dim
        ctx_vectors = torch.empty(n_ctx, embed_dim, device=device)
        nn.init.normal_(ctx_vectors, std=CONTEXT_INIT_STD)
        
        # Store initialization text for later use
        self._ctx_init_text = ctx_init if ctx_init else None
        
        # Make context vectors learnable parameters
        self.ctx = nn.Parameter(ctx_vectors)
        
        # Initialize placeholders for class embeddings
        self.name_lens: List[int] = []
        self.class_token_embeddings: List[torch.Tensor] = []
        self._processor: Optional[CLIPProcessor] = None
        
        
    def initialize_with_processor(self, processor: CLIPProcessor) -> None:
        """Initialize class embeddings using the processor after model loading."""
        self._processor = processor
        
        # Re-initialize context vectors with text if provided
        if self._ctx_init_text:
            self._initialize_context_from_text(processor)
        
        # Create class name embeddings using the template
        self._create_class_embeddings(processor)
        
        # Register special tokens
        self._register_special_tokens(processor)
    
    def _initialize_context_from_text(self, processor: CLIPProcessor) -> None:
        """Initialize context vectors from text."""
        try:
            ctx_init_tokens = processor.tokenizer(
                self._ctx_init_text, return_tensors="pt"
            )["input_ids"]
            
            if len(ctx_init_tokens[0]) > self.n_ctx:
                ctx_init_tokens = ctx_init_tokens[:, :self.n_ctx]
                
            ctx_vectors = self.token_embedding(ctx_init_tokens.to(self.device))
            ctx_vectors = ctx_vectors.squeeze(0)  # Remove batch dimension
            
            # Pad if needed
            if ctx_vectors.shape[0] < self.n_ctx:
                embed_dim = self.token_embedding.embedding_dim
                padding = torch.zeros(
                    self.n_ctx - ctx_vectors.shape[0], embed_dim, device=self.device
                )
                ctx_vectors = torch.cat([ctx_vectors, padding], dim=0)
            
            # Update the parameter
            with torch.no_grad():
                self.ctx.copy_(ctx_vectors)
                
        except Exception as e:
            print(f"Warning: Could not initialize context with text '{self._ctx_init_text}': {e}")
            print("Using random initialization instead.")
    
    def _create_class_embeddings(self, processor: CLIPProcessor) -> None:
        """Create embeddings for each class name."""
        self.name_lens = []
        self.class_token_embeddings = []
        
        for name in self.class_names:
            formatted_text = self.template.format(class_name=name)
            tokens = processor.tokenizer(
                formatted_text, 
                return_tensors="pt", 
                padding=False, 
                truncation=True,
                add_special_tokens=False # Avoid adding [CLS] and [SEP] automatically
            )["input_ids"].to(self.device)
            
            class_embeddings = self.token_embedding(tokens).squeeze(0)
            
            # Store length and embeddings
            self.name_lens.append(len(tokens[0]))  
            self.class_token_embeddings.append(class_embeddings) 
    
    def _register_special_tokens(self, processor: CLIPProcessor) -> None:
        """Register special token embeddings."""
        tokenizer = processor.tokenizer
        start_token_id = torch.tensor([tokenizer.bos_token_id], device=self.device)
        end_token_id = torch.tensor([tokenizer.eos_token_id], device=self.device)
        
        self.register_buffer("start_token", self.token_embedding(start_token_id))
        self.register_buffer("end_token", self.token_embedding(end_token_id))
        
        
    def forward(self) -> torch.Tensor:
        """Forward pass to generate text embeddings with learned context."""
        ctx = self.ctx  # [n_ctx, embed_dim]
        
        # Create prompts for all classes
        prompts = self._build_prompts(ctx)
        
        # Stack and pad prompts
        padded_prompts, attention_masks, eos_pos = self._pad_prompts(prompts)
        
        # Pass through text encoder and get final features
        return self._encode_text(padded_prompts, attention_masks, eos_pos)


    def _build_prompts(self, ctx: torch.Tensor) -> List[torch.Tensor]:
        """Build prompt embeddings for all classes."""
        prompts = []
        
        for class_embeddings in self.class_token_embeddings:
            # Construct: [CLS] + context + class_tokens + [SEP]
            prompt = torch.cat([
                self.start_token,    # [CLS]
                ctx,                 # Learned context
                class_embeddings,    # Class name tokens
                self.end_token,      # [SEP]
            ], dim=0)
            prompts.append(prompt)
        
        return prompts
    
    def _pad_prompts(self, prompts: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad prompts to the same length."""
        max_len = max(p.shape[0] for p in prompts)
        embed_dim = prompts[0].shape[1]
        
        padded_prompts = torch.zeros(len(prompts), max_len, embed_dim, device=self.device)
        attention_masks = torch.zeros(len(prompts), max_len, device=self.device)
        eos_pos = []
        
        for i, prompt in enumerate(prompts):
            length = prompt.shape[0]
            eos_pos += [length - 1]  # Position of [SEP] token
            padded_prompts[i, :length] = prompt
            attention_masks[i, :length] = 1

        return padded_prompts, attention_masks, eos_pos
    
    def _encode_text(self, text_embeddings: torch.Tensor, attention_masks: torch.Tensor, eos_pos: List[int]) -> torch.Tensor:
        """Encode text embeddings through CLIP text encoder."""
        # For CLIP, we need to manually pass through the text encoder layers
        # since it doesn't support inputs_embeds directly
        
        # Add positional embeddings
        seq_length = text_embeddings.shape[1]
        # Create position_ids that maintain gradients
        position_ids = torch.arange(seq_length, dtype=torch.long, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(text_embeddings.shape[0], -1)
        position_embeddings = self.positional_embedding(position_ids)
        
        # Combine token and position embeddings
        hidden_states = text_embeddings + position_embeddings

        # Convert 2D attention mask to 4D for multi-head attention
        # Shape: [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
        extended_attention_mask = attention_masks.unsqueeze(1).unsqueeze(1)
        
        # Create causal mask for text (lower triangular)
        batch_size, seq_len = attention_masks.shape
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)
        
        # Combine masks: apply both padding mask and causal mask
        combined_mask = extended_attention_mask * causal_mask
        
        # Convert to the format expected by the encoder (0 for masked, large negative for unmasked)
        combined_mask = (1.0 - combined_mask) * -10000.0

        hidden_states = self.text_encoder.encoder(hidden_states, attention_mask=combined_mask).last_hidden_state

        # Apply final layer norm
        hidden_states = self.text_encoder.final_layer_norm(hidden_states)
        
        pooled_output = hidden_states[
            torch.arange(hidden_states.shape[0], device=hidden_states.device),
            eos_pos
        ]
        
        # Apply final projection (like in original CLIP)
        text_features = self.text_projection(pooled_output)
        
        # Normalize features
        return text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    
    def train(self, mode: bool = True) -> 'CoOpTextEncoder':
        """Set training mode while keeping CLIP frozen."""
        super().train(mode)
        
        # Keep CLIP model frozen but allow text encoder to process gradients
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad_(False)
        
        # Set text encoder to train mode to allow gradient flow through embeddings
        if mode:
            self.clip_model.text_model.train()
        else:
            self.clip_model.text_model.eval()
            
        # Ensure context vectors remain learnable
        self.ctx.requires_grad_(True)
        
        return self
