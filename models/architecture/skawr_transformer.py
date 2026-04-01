"""
SKAWR Custom Transformer Model

A BERT-style transformer designed specifically for marketplace search.
Supports multi-task learning for:
- Masked language modeling (general understanding)
- Domain classification (automotive, electronics, etc.)
- Query specificity prediction
- Contrastive learning for query-product matching
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from typing import Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass

@dataclass
class SKAWRConfig:
    """Configuration for SKAWR Transformer model."""

    vocab_size: int = 30000
    hidden_size: int = 512
    num_attention_heads: int = 8
    num_hidden_layers: int = 6
    intermediate_size: int = 2048
    max_position_embeddings: int = 512

    embedding_dim: int = 512
    num_domains: int = 4
    num_specificity_levels: int = 3

    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12

    pad_token_id: int = 0
    cls_token_id: int = 1
    sep_token_id: int = 2
    mask_token_id: int = 3

    def __post_init__(self):
        # Validate configuration
        assert self.hidden_size % self.num_attention_heads == 0, \
            "hidden_size must be divisible by num_attention_heads"

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""

    def __init__(self, config: SKAWRConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape tensor for multi-head attention."""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Compute Q, K, V
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply attention mask
        if attention_mask is not None:
            attention_scores += attention_mask

        # Normalize attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # Reshape back to original
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # Output projection
        attention_output = self.dense(context_layer)
        attention_output = self.output_dropout(attention_output)

        # Residual connection and layer norm
        attention_output = self.LayerNorm(attention_output + hidden_states)

        return attention_output

class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, config: SKAWRConfig):
        super().__init__()
        self.dense_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Feed-forward transformation
        ff_output = self.dense_1(hidden_states)
        ff_output = F.gelu(ff_output)  # Use GELU activation like BERT
        ff_output = self.dense_2(ff_output)
        ff_output = self.dropout(ff_output)

        # Residual connection and layer norm
        ff_output = self.LayerNorm(ff_output + hidden_states)

        return ff_output

class TransformerLayer(nn.Module):
    """Single transformer layer (attention + feed-forward)."""

    def __init__(self, config: SKAWRConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attention_output = self.attention(hidden_states, attention_mask)
        layer_output = self.feed_forward(attention_output)
        return layer_output

class SKAWRTransformerEncoder(nn.Module):
    """Stack of transformer layers."""

    def __init__(self, config: SKAWRConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_hidden_layers)
        ])

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states

class SKAWREmbeddings(nn.Module):
    """Input embeddings: token + position embeddings."""

    def __init__(self, config: SKAWRConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Register position_ids buffer
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_length = input_ids.size(1)

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # Get embeddings
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        # Combine embeddings
        embeddings = word_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class SKAWRPooler(nn.Module):
    """Pooling layer to create sequence representation from [CLS] token."""

    def __init__(self, config: SKAWRConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Take [CLS] token representation (first token)
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class SKAWRTransformer(nn.Module):
    """
    SKAWR Custom Transformer for Marketplace Search

    Multi-task transformer that learns:
    1. General language understanding (masked LM)
    2. Domain classification
    3. Query specificity prediction
    4. Contrastive embeddings for search
    """

    def __init__(self, config: SKAWRConfig):
        super().__init__()
        self.config = config

        # Core transformer components
        self.embeddings = SKAWREmbeddings(config)
        self.encoder = SKAWRTransformerEncoder(config)
        self.pooler = SKAWRPooler(config)

        # Task-specific heads
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.domain_classifier = nn.Linear(config.hidden_size, config.num_domains)
        self.specificity_predictor = nn.Linear(config.hidden_size, config.num_specificity_levels)
        self.embedding_projector = nn.Linear(config.hidden_size, config.embedding_dim)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def create_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create attention mask from input_ids."""
        # Create mask where 1 = real token, 0 = padding
        attention_mask = (input_ids != self.config.pad_token_id).float()

        # Convert to additive mask for attention scores
        # Shape: (batch_size, 1, 1, seq_len)
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        domain_labels: Optional[torch.Tensor] = None,
        specificity_labels: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:

        batch_size, seq_length = input_ids.shape

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = self.create_attention_mask(input_ids)

        # Forward pass through transformer
        embeddings = self.embeddings(input_ids)
        encoder_outputs = self.encoder(embeddings, attention_mask)
        pooled_output = self.pooler(encoder_outputs)

        outputs = {
            "last_hidden_state": encoder_outputs,
            "pooler_output": pooled_output
        }

        # Task-specific predictions
        lm_logits = self.lm_head(encoder_outputs)
        domain_logits = self.domain_classifier(pooled_output)
        specificity_logits = self.specificity_predictor(pooled_output)

        outputs.update({
            "lm_logits": lm_logits,
            "domain_logits": domain_logits,
            "specificity_logits": specificity_logits
        })

        # Generate embeddings for search
        if return_embeddings:
            search_embeddings = self.embedding_projector(pooled_output)
            search_embeddings = F.normalize(search_embeddings, p=2, dim=1)
            outputs["search_embeddings"] = search_embeddings

        # Calculate losses if labels provided
        total_loss = 0

        # Masked language modeling loss
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only compute loss on masked tokens
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            outputs["masked_lm_loss"] = masked_lm_loss
            total_loss += masked_lm_loss

        # Domain classification loss
        if domain_labels is not None:
            loss_fct = CrossEntropyLoss()
            domain_loss = loss_fct(domain_logits, domain_labels)
            outputs["domain_loss"] = domain_loss
            total_loss += 0.5 * domain_loss  # Weight domain loss

        # Specificity prediction loss
        if specificity_labels is not None:
            loss_fct = CrossEntropyLoss()
            specificity_loss = loss_fct(specificity_logits, specificity_labels)
            outputs["specificity_loss"] = specificity_loss
            total_loss += 0.5 * specificity_loss  # Weight specificity loss

        if total_loss > 0:
            outputs["loss"] = total_loss

        return outputs

    def get_search_embeddings(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get normalized embeddings for search/similarity tasks."""
        outputs = self.forward(input_ids, attention_mask, return_embeddings=True)
        return outputs["search_embeddings"]

    def predict_domain(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict domain classification."""
        outputs = self.forward(input_ids, attention_mask)
        return F.softmax(outputs["domain_logits"], dim=-1)

    def predict_specificity(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict query specificity level."""
        outputs = self.forward(input_ids, attention_mask)
        return F.softmax(outputs["specificity_logits"], dim=-1)

class ContrastiveLoss(nn.Module):
    """Contrastive loss for learning query-product similarity."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, query_embeddings: torch.Tensor, product_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss between query and product embeddings.

        Args:
            query_embeddings: Shape (batch_size, embedding_dim)
            product_embeddings: Shape (batch_size, embedding_dim)
        """
        # Normalize embeddings
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        product_embeddings = F.normalize(product_embeddings, p=2, dim=1)

        # Compute similarity matrix
        logits = torch.matmul(query_embeddings, product_embeddings.T) / self.temperature

        # Create labels (positive pairs are on diagonal)
        labels = torch.arange(logits.size(0), device=logits.device)

        # Symmetric loss: query->product + product->query
        loss_q2p = F.cross_entropy(logits, labels)
        loss_p2q = F.cross_entropy(logits.T, labels)

        return (loss_q2p + loss_p2q) / 2

def create_model(config: Dict[str, Any]) -> SKAWRTransformer:
    """Factory function to create SKAWR transformer model."""
    model_config = SKAWRConfig(**config)
    return SKAWRTransformer(model_config)