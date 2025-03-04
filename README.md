# Fine-Tuning ModernBERT: Exploring a Lightweight Approach to Prompt Guardrails

*Decoder-only* and *encoder-decoder* models have become the standard choice for Generative AI applications. However, *encoder-only* models remain essential in AI pipelines due to their attractive balance between performance and inference requirements in non-generative tasks such as classification, retrieval and QA, where generation of new text is not the primary goal.

In this article, we explore [ModernBERT](https://arxiv.org/abs/2412.13663) [1], a significant advancement in *encoder-only* models. We first outline the key architectural improvements underpinning this model, and then demonstrate how to fine-tune the  [ModerBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) version for implementing a lightweight classifier that discriminates malicious prompts. This provides a baseline approach for adding custom, cheap safety checks to AI pipelines without trading off significant latency.

#### Table of Contents

1. [A Primer on Encoder-Only Models](#a-primer-on-encoder-only-models)
2. [From BERT to ModernBERT](#from-bert-to-modernbert)
    1. [Technical Evolution](#technical-evolution)
    2. [Alternating Attention](#alternating-attention)
    3. [Flash Attention](#flash-attention)
4. [Guardrails Dataset](#guardrails-dataset)
5. [Fine-Tuning](#fine-tuning)
6. [Evaluation](#evaluation)

## A Primer on Encoder-Only Models

*Encoder-only* models, such as [BERT](https://arxiv.org/abs/1810.04805) [2], are built entirely from the encoder component of the *[Transformer](https://arxiv.org/abs/1706.03762)* architecture [3]. The encoder consists of multiple stacked layers, each comprising a bidirectional multi-head self-attention sublayer and feed-forward neural networks. In practice, input sequences are first tokenized and converted into embedding vectors, with positional encodings added to represent token order. These embeddings pass through the encoder layers, where self-attention heads learn different aspects of the input in form of weighted attention scores, creating updated embeddings that capture contextual dependencies and semantical understanding across the entire sequence.

At its core, this architecture differs from *decoder-only* models in that: (i) it processes input tokens bidirectionally, considering the full context of a sequence during both training and inference, whereas decoder models generate tokens sequentially in an autoregressive fashion, limiting paralellization; (ii) it requires only a single forward pass to produce contextualized representations of the entire input, instead of one pass for each generated token; and (iii) it typically has fewer parameters ([ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large) has 395M parameters, while [Llama 3.3](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) has 70B) due to its simpler objective, focused on understanding input rather than generating output.

This enables *encoder-only* models to efficiently process corpora of documents at scale and quickly perform non-generative tasks.

## From BERT to ModernBERT

#### Technical Evolution

Introduced in December 2024 by [Answer.AI](https://huggingface.co/answerdotai) and [LightOn.AI](https://huggingface.co/lightonai), [ModernBERT](https://arxiv.org/abs/2412.13663) is a state-of-the-art *encoder-only* model that advances upon the original [BERT](https://arxiv.org/abs/1810.04805) architecture by replacing some of its building blocks:

| | BERT | ModernBERT | Relevance |
|---------|------|-------------|--------------|
| **Max Sequence Length** | 512 tokens | 8,192 tokens | *Larger Context (16x), Better Understanding and Downstream Performance* |
| **Bias Terms** | All Layers | Final Decoder | *More Efficient Usage of Parameter Capacity* |
| **Positional Encoding** | Absolute | Rotary (RoPE) | *Scale to Sequences longer than those provided in Training* |
| **Normalization** | Post-LN | Pre-LN & Extra-N after Embeddings | *Enhance Training Stability* |
| **Activation** | GeLU | GeGLU (Gated GeLU) | *Enhance Training and Model Performance* |
| **Attention Mechanism** | Full Global | Global (1/3) & Local (2/3) with 128-token sliding window | *Improve Computational Efficiency from O(n^2) to O(seq_length × window)* |
| **Batch Processing** | Padding | Unpadding & Sequence Packing | *Avoid Waste Computation on Empty Tokens* |
| **Flash Attention** | N/A | Flash | *Minimize GPU Transfers, Speed Up Inference* |

By incorporating these architectural advances, [ModernBERT](https://arxiv.org/abs/2412.13663) improves over [BERT](https://arxiv.org/abs/1810.04805) models across both *computational efficiency* and *accuracy* without the traditional tradeoffs between these metrics. Among all technical improvements, we found the integration of [Alternating Attention](https://arxiv.org/abs/1706.03762) along [FlashAttention](https://arxiv.org/abs/2205.14135) to be particular impactful, as they reduced the memory requirements of our training process by nearly 70%.

#### Alternating Attention

[Transformer](https://arxiv.org/abs/1706.03762) models face scalability challenges when working with long inputs as the self-attention mechanism has quadratic time and memory complexity in sequence length.

In the next figures we can see that while self-attention enables the model to correctly learn contextual dependencies and semantic understanding across each input sequence, the computational complexity is indeed quadratic. For each attention head in a single layer, *attention* requires to perform *Query (Q)* and *Key (K)* matrix multiplications, creating an attention matrix where each entry represents the attention score between a pair of tokens in the sequence (*dark blue boxes indicate higher attention scores*):

<p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/64a13b68b14ab77f9e3eb061/_U5_6EP69bCvQGsQ8WIhp.png">
</p>

<p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/64a13b68b14ab77f9e3eb061/379jyReT0dsyIHrYojitV.png">
</p>

<p align="center">Full Attention Visualization (source: https://github.com/dcarpintero/generative-ai-101)</p>

To address this limitation, *alternating attention patterns* have been introduced to scale language models with longer contexts. [ModernBERT](https://arxiv.org/abs/2412.13663) builds upon [Sliding Window Alternating Attention](https://arxiv.org/abs/1706.03762) [4]. This means that attention layers alternate between *global attention*, where every token within a sequence attends to every other token (as in the original *Transformer* implementation), and *local attention*, where each token only attends to the 128 tokens nearest to itself. This approach resembles the way we naturally switch between two modes of understanding when reading a book. That is, when reading a particular chapter, our primary focus is on the immediate context (local attention), whereas periodically, our mind performs broader understanding by connecting the current chapter to the main plot (global attention).

Technically, this implementation enables [ModernBERT](https://arxiv.org/abs/2412.13663) to (i) improve the computational efficiency by reducing the number of attention calculations, (ii) scale to contexts of thousands of tokens, and (iii) simplify the implementation of downstream tasks by eliminating the need to chunk or truncate long inputs.

#### Flash Attention

Beyond the known quadratic complexity of self-attention in [Transformer](https://arxiv.org/abs/1706.03762) models, the authors of [FlashAttention](https://arxiv.org/abs/2205.14135) [5] identified another critical efficiency challenge related to modern GPU memory architectures. These architectures are built upon two distinct memory levels: (i) *on-chip, ultra-fast, very small Static Random Access Memory (SRAM)*, and (ii) *off-chip, slower, larger High Bandwidth Memory (HBM)*.

The key insight of their work is that the difference in speed between these two memory levels creates a bottleneck as GPUs spend significant time waiting for data to move between *HBM* and *SRAM*. Traditional attention implementations do not take into account this memory hierarchy that requires moving large matrices between *HBM* and *SRAM*. [FlashAttention](https://arxiv.org/abs/2205.14135) strategically organizes computation to minimize these expensive memory transfers, even if it means doing some calculations more than once. In practice, [FlashAttention](https://arxiv.org/abs/2205.14135) optimizes I/O operations by applying:

- `Tiling`: splits input matrices into smaller blocks that fit into on-chip *SRAM*, allowing attention to be computed incrementally by looping these blocks without materializing the large N×N sequence attention matrix in the slower *HBM*;
- `Recomputation`: avoids storing intermediate values during the forward pass by recalculating them during the backward pass when needed. This trades off more computation for significantly fewer memory accesses; and
- `Kernel fusion`: combines multiple operations (matrix multiplication, softmax, masking, dropout) into a single GPU kernel, further reducing memory transfers between *HBM* and *SRAM*.

<p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/64a13b68b14ab77f9e3eb061/7GncpQZkov06h_FjMDbtS.png">
</p>

<p align="center">FlashAttention (source: https://arxiv.org/abs/2205.14135)</p>

Further optimizations were proposed in the follow-up [FlashAttention-2](https://arxiv.org/abs/2307.08691) [6] by: (i) refining the original algorithm to reduce the number of non-matrix multiplications as they take longer to perform, (ii) parallelizing computation along the sequence length dimension, in addition to the batch and number of heads dimension, to make full use of GPU resources, and (iii) reducing shared memory access by inverting the split scheme and partitioning *Q* while keeping *K, V* matrices accesible.

<p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/64a13b68b14ab77f9e3eb061/dwoHPoeH0Ul5JVbvDsK5u.png">
</p>

<p align="center">FlashAttention-2 split scheme (source: https://arxiv.org/abs/2307.08691)</p>

## Guardrails Dataset

LLM-based applications are susceptible to security challenges in form of prompt attacks – carefully crafted inputs designed to subvert the models' intended behavior by exploiting their reliance on natural language inputs. These prompt injection attacks can result in models exposing sensitive data or deviating from their intended behavior.

A common defense approach is the use of guardrails to identify and filter out potentially malicious prompts. In this example, we will fine-tune [ModernBERT](https://arxiv.org/abs/2412.13663) to discriminate malicious prompts using the [InjecGuard](https://arxiv.org/abs/2410.22770) [5] dataset, which provides over 75k samples of both legitimate interactions and documented attack attempts.

We will use the `load_dataset()` method from the 🤗 Datasets library to load the [InjecGuard](https://arxiv.org/abs/2410.22770) dataset:

```
[...]
```

Let’s check out an example:

```
[...]
```

To train our model, we need to convert the text prompts to token IDs. This is done by a `Tokenizer`, which tokenizes the inputs (including converting the tokens to their corresponding IDs in the pre-trained vocabulary):

```
[...]
```

## Fine Tuning

In this section, we adapt the [ModerBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) model and a feedforward classification head to discriminate user prompts.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/64a13b68b14ab77f9e3eb061/V4DMZ2gR53Gl0yas_Vc9U.png)

## Evaluation

[...]

## Citation

```
@article{modernbert-prompt-guardrails
  author = { Diego Carpintero},
  title = {Fine-Tuning ModernBERT: Exploring a Lightweight Approach to Prompt Guardrails},
  journal = {Hugging Face Blog},
  year = {2025},
  note = {https://huggingface.co/blog/dcarpintero/fine-tuning-modernbert},
}
```

## References

- [1] Clavié, et al. 2024. *Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference*. [arXiv:2412.13663](https://arxiv.org/abs/2412.13663)
- [2] Devlin, et al. 2018. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- [3] Vaswani, et al. 2017. *Attention Is All You Need*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).
- [4] Beltagy, et al. 2020. *Longformer: The Long-Document Transformer*. [arXiv:2004.05150](https://arxiv.org/abs/2004.05150v2)
- [5] Dao, et al. 2022. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135).
- [6] Dao. 2023. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691).
- [7] Li, et al. 2024. *InjecGuard: Benchmarking and Mitigating Over-defense in Prompt
Injection Guardrail Models*. [arXiv:2410.22770](https://arxiv.org/abs/2410.22770).
- [8] Carpintero. 2025. *Fine-Tuning ModernBERT - Codebase Repository*. [github.com/dcarpintero/fine-tuning-modernbert](https://github.com/dcarpintero/fine-tuning-modernbert)

## Author

Diego Carpintero (https://github.com/dcarpintero)
