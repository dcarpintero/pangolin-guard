# Fine-Tuning ModernBERT: Exploring a Lightweight Approach to Prompt Guardrails

*Decoder-only* and *encoder-decoder* Large Language Models (LLMs) have become the standard choice for Generative AI applications. However, *encoder-only* models remain essential in AI pipelines due to their attractive balance between performance and inference requirements in non-generative tasks such as classification, named entity recognition, and semantic similarity, where generation of new text is not the primary goal.

In this article, we explore [ModernBERT](https://arxiv.org/abs/2412.13663) [1], a significant advancement in *encoder-only* models. We first outline the key architectural improvements underpinning this model, and then demonstrate how to fine-tune  [ModerBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) version for implementing a lightweight classifier that discriminates malicious prompts. This might provide a baseline approach for adding custom, cheap safety checks to AI pipelines without trading off significant latency.

#### Table of Contents

1. [A Primer on Encoder-Only Models](#A-Primer-on-Encoder-Only-Models)
2. [ModernBERT](#ModernBERT)
3. [Guardrails Dataset](#Guardrails-Dataset)
4. [Fine-Tuning](#Fine-Tuning)
5. [Evaluation](#Evaluation)

## A Primer on Encoder-Only Models

Encoder-only models, such as [BERT](https://arxiv.org/abs/1810.04805) [2], are built entirely from the encoder component of the *[Transformer](https://arxiv.org/abs/1706.03762)* architecture [3]. The encoder consists of multiple stacked layers, each comprising a bidirectional multi-head self-attention sublayer and feed-forward neural networks. In practice, input sequences are first tokenized and converted into embedding vectors, with positional encodings added to represent token order. These embeddings pass through the encoder layers, where self-attention heads learn different aspects of the input in form of weighted attention scores, creating updated embeddings that capture contextual dependencies and semantical understanding across the entire sequence.

At its core, this architecture differs from decoder-only models in that: (i) it processes input tokens bidirectionally, considering the full context of a sequence during both training and inference, whereas decoder-only models generate tokens sequentially in an autoregressive fashion, limiting paralellization; (ii) it requires only a single forward pass to produce contextualized representations of the entire input, instead of one pass for each generated token as in a decoder architecture; and (iii) it typically has fewer parameters (ModernBERT-large has 395M parameters, while Llama 3.3 has 70B) due to its simpler objective, focused on understanding input rather than generating output.

This enables encoder-only models to efficiently process corpora of documents at scale and quickly perform discriminative tasks.

## ModernBERT

Introduced in December 2024 by [Answer.AI](https://huggingface.co/answerdotai) and [LightOn.AI](https://huggingface.co/lightonai), ModernBERT is a state-of-the-art *encoder-only* model that advances upon the original BERT architecture by replacing some of its building blocks:

| Feature | BERT | ModernBERT | Effect |
|---------|------|-------------|--------------|
| **Max Sequence Length** | 512 tokens | 8,192 tokens | *Larger Context (16x), Better Understanding and Performance* |
| **Bias Terms** | All Layers | Final Decoder | *More Efficient Usage of Parameter Budget* |
| **Positional Encoding** | Absolute | Rotary (RoPE) | *Scale to Sequences longer than in Training* |
| **Normalization** | Post-LN | Pre-LN & Extra-N after Embeddings | *Enhance Training Stability* |
| **Activation** | GeLU | GeGLU (Gated GeLU) | *Enhance Training and Model Performance* |
| **Attention Mechanism** | Full Global | Global (1/3) & Local (2/3) with 128-token sliding window | *Improve Computational Efficiency from O(n^2) to O(seq_length × window)* |
| **Batch Processing** | Padding | Unpadding & Sequence Packing | *Avoid Waste Computation on Empty Tokens* |
| **Flash Attention** | N/A | Flash | *Minimize GPU Transfers, Speed Up Processing* |

## Guardrails Dataset

LLM-based applications are susceptible to security challenges in the form of prompt attacks – carefully crafted inputs designed to subvert the models' intended behavior by exploiting their reliance on natural language inputs. These prompt injection attacks can result in models exposing sensitive data or deviating from their core objectives, similar to social engineering exploits.

A common defense approach is the use of guardrails to identify and filter out potentially malicious prompts. In this example, we will fine-tune ModernBERT to discriminate malicious prompts using the [InjecGuard](https://arxiv.org/abs/2410.22770) [4] dataset, which provides over 75k samples of both legitimate interactions and documented attack attempts.

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

[...]

## Evaluation

[...]

## Citation

```
@article{carpintero2025
  author = { Diego Carpintero},
  title = {Fine-Tuning ModernBERT: A Lightweight Approach to AI Safety Guardrails},
  journal = {Hugging Face Blog},
  year = {2025},
  note = {https://huggingface.co/blog/dcarpintero/fine-tuning-modernbert},
}
```

## References

- [1] Clavié, et al. 2024. *Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference*. [arXiv:2412.13663](https://arxiv.org/abs/2412.13663)
- [2] Devlin, et al. 2018. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- [3] Vaswani, et al. 2017. *Attention Is All You Need*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).
- [4] Li, et al. 2024. *InjecGuard: Benchmarking and Mitigating Over-defense in Prompt
Injection Guardrail Models*. [arXiv:2410.22770](https://arxiv.org/abs/2410.22770).
- [5] [GitHub Repo](https://github.com/dcarpintero/fine-tuning-modernbert)

## Author

Diego Carpintero (https://github.com/dcarpintero)