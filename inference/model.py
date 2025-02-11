# coding=utf-8
# Copyright 2025 Arsh Team
# Apache License Version 2.0

import torch
import torch.nn as nn
from transformers import (
    PreTrainedModel,
    GenerationMixin,
    Cache,
    DynamicCache,
    StaticCache,
    FlashAttentionKwargs,
    ArshConfig
)
from typing import Optional, Tuple, Union, List


class ArshRMSNorm(nn.Module):
    """
    RMS Normalization layer customized for Arsh architecture

    Args:
        hidden_size (int): Dimension of hidden states
        eps (float): Epsilon value for numerical stability

    Example:
        >>> norm = ArshRMSNorm(768)
        >>> x = torch.randn(1, 10, 768)
        >>> output = norm(x)
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class ArshRotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding implementation for Arsh model

    Args:
        config (ArshConfig): Model configuration

    Attributes:
        max_seq_len_cached (int): Maximum cached sequence length
        attention_scaling (float): Scaling factor for attention

    Example:
        >>> config = ArshConfig()
        >>> rotary_emb = ArshRotaryEmbedding(config)
    """

    def __init__(self, config: ArshConfig):
        super().__init__()
        self.config = config
        self.max_seq_len_cached = config.max_position_embeddings
        self.rope_type = config.rope_scaling.get("type", "default")
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(config)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _update_frequency(self, position_ids: torch.Tensor):
        """Dynamically update frequency based on input sequence length"""
        seq_len = position_ids.max() + 1
        if seq_len > self.max_seq_len_cached:
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, seq_len=seq_len
            )
            self.register_buffer("inv_freq", inv_freq)
            self.max_seq_len_cached = seq_len

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if "dynamic" in self.rope_type:
            self._update_frequency(position_ids)

        # Compute cosine and sine embeddings
        inv_freq_expanded = self.inv_freq[None, :, None]
        position_ids_expanded = position_ids[:, None, :].float()

        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos * self.attention_scaling, sin * self.attention_scaling


class ArshMLP(nn.Module):
    """
    Gated MLP Block for Arsh model

    Args:
        config (ArshConfig): Model configuration

    Example:
        >>> config = ArshConfig()
        >>> mlp = ArshMLP(config)
        >>> x = torch.randn(1, 10, 768)
        >>> output = mlp(x)
    """

    def __init__(self, config: ArshConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class ArshAttention(nn.Module):
    """
    Multi-head Attention layer with RoPE support

    Args:
        config (ArshConfig): Model configuration
        layer_idx (int): Layer index

    Example:
        >>> config = ArshConfig()
        >>> attn = ArshAttention(config, layer_idx=0)
    """

    def __init__(self, config: ArshConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)

        self.rotary_emb = ArshRotaryEmbedding(config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape

        # Project inputs
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Attention computation
        attn_output, attn_weights = scaled_dot_product_attention(
            q, k, v, attention_mask=attention_mask
        )

        # Output projection
        output = self.o_proj(attn_output.view(batch_size, seq_len, -1))
        return output, attn_weights


class ArshDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer

    Args:
        config (ArshConfig): Model configuration
        layer_idx (int): Layer index

    Example:
        >>> config = ArshConfig()
        >>> layer = ArshDecoderLayer(config, layer_idx=0)
    """

    def __init__(self, config: ArshConfig, layer_idx: int):
        super().__init__()
        self.self_attn = ArshAttention(config, layer_idx)
        self.mlp = ArshMLP(config)
        self.input_norm = ArshRMSNorm(config.hidden_size)
        self.post_attn_norm = ArshRMSNorm(config.hidden_size)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention block
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)
        attn_output, _ = self.self_attn(hidden_states, attention_mask, position_ids)
        hidden_states = residual + attn_output

        # MLP block
        residual = hidden_states
        hidden_states = self.post_attn_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class ArshModel(PreTrainedModel):
    """
    Main Arsh model architecture

    Args:
        config (ArshConfig): Model configuration

    Example:
        >>> config = ArshConfig()
        >>> model = ArshModel(config)
    """

    def __init__(self, config: ArshConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [ArshDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = ArshRMSNorm(config.hidden_size)

        self.post_init()

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids
            )

        return self.norm(hidden_states)


class ArshForCausalLM(ArshModel, GenerationMixin):
    """
    Arsh model for causal language modeling

    Args:
        config (ArshConfig): Model configuration

    Example:
        >>> config = ArshConfig()
        >>> model = ArshForCausalLM(config)
        >>> inputs = {"input_ids": torch.randint(0, 100, (1, 10))}
        >>> outputs = model(**inputs)
    """

    def __init__(self, config: ArshConfig):
        super().__init__(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
    ) -> dict:
        hidden_states = super().forward(input_ids, attention_mask)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return {"loss": loss, "logits": logits}
