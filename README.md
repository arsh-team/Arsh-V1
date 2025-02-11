
# Arsh-V1 Model Card

We introduce our first reasoning and chat model based on phi weights. The Arsh architecture is built on new technologies, making this model optimized for performance and extendability in chat applications. Our team has utilized high-quality datasets for training to ensure reliable and robust performance.

Our ongoing efforts include frequent updates and enhancements that aim to blend multiple architectures, striving to build the best model possible. We are making significant progress and are dedicated to refining Arsh-V1 further.

---
<a href="https://huggingface.co/arshiaafshani/Arsh-V1" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Arsh%20V1%20AI-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>


## Model Overview

The Arsh-V1 model belongs to the `Causal Language Model (LLM)` family and is designed for text generation tasks. It comprises a stack of transformer blocks and is optimized for high throughput and low latency in language generation.

### Architecture

- **Model Name:** ArshForCausalLM
- **Model Type:** Llama
- **Hidden Size:** 5120
- **Number of Layers:** 40
- **Attention Heads:** 40
- **Maximum Sequence Length:** 16384
- **Vocabulary Size:** 100352
- **Activation Function:** SiLU (Gaussian Error Linear Unit)

This architecture enables the model to capture complex language patterns, making it suitable for conversation and reasoning tasks.

---

## Key Components

### 1. **RMS Normalization**

The `ArshRMSNorm` layer is employed for normalization within the model, enhancing training stability and speed.

```python
norm = ArshRMSNorm(hidden_size=5120)
x = torch.randn(1, 10, 5120)
output = norm(x)
```

### 2. **Rotary Position Embedding**

ArshRotaryEmbedding leverages rotary embeddings for efficient positional encoding in transformer architectures.

```python
config = ArshConfig(max_position_embeddings=16384)
rotary_emb = ArshRotaryEmbedding(config)
```


### **3. Gated MLP Block**

The ArshMLP component is responsible for non-linear transformations, incorporating a gating mechanism.

```python
mlp = ArshMLP(config)
x = torch.randn(1, 10, 5120)
output = mlp(x)
```



### **4. Multi-Head Attention**

The ArshAttention layer implements multi-head attention with support for rotary positional embeddings, enhancing context understanding.

```python
attention_layer = ArshAttention(config, layer_idx=0)
hidden_states = torch.randn(1, 10, 5120)
attn_output, attn_weights = attention_layer(hidden_states)
```


### **5. Transformer Decoder Layer**

The ArshDecoderLayer integrates self-attention and feed-forward neural network components in series.

```python
decoder_layer = ArshDecoderLayer(config, layer_idx=0)
hidden_states = decoder_layer(hidden_states)
```


## License

Our model is based on MIT license, which allows for modifications, creation, fine-tuning, and commercial usage. We appreciate your contributions to make Arsh-V1 better!
