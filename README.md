## PangolinGuard

LLM applications face critical security challenges in form of prompt injections and jailbreaks. This can result in models leaking sensitive data or deviating from their intended behavior. Existing safeguard models are not fully open and have limited context windows (e.g., only 512 tokens in LlamaGuard).

PangolinGuard is a ModernBERT (Large), lightweight model that discriminates malicious prompts. It closely approximates the performance of Claude 3.7 and Gemini Flash 2.0 (84.72% accuracy vs. 86.81% and 86.11%) on a mixed benchmark targeting prompt safety and malicious input detection (NotInject, BIPIA, Wildguard-Benign, PINT), while maintaining low latency (<40ms) and supporting 8192 input tokens.

🤗 [Tech-Blog on Hugging Face](https://huggingface.co/blog/dcarpintero/pangolin-fine-tuning-modern-bert)

### Intended uses

- Adding custom, self-hosted safety checks to AI agents and conversational interfaces
- Topic and content moderation
- Mitigating risks when connecting AI pipelines to external services

### Evaluation data

Evaluated on unseen data from a subset of specialized benchmarks targeting prompt safety and malicious input detection, while testing over-defense behavior:

- NotInject: Designed to measure over-defense in prompt guard models by including benign inputs enriched with trigger words common in prompt injection attacks.
- BIPIA: Evaluates privacy invasion attempts and boundary-pushing queries through indirect prompt injection attacks.
- Wildguard-Benign: Represents legitimate but potentially ambiguous prompts.
- PINT: Evaluates particularly nuanced prompt injection, jailbreaks, and benign prompts that could be misidentified as malicious.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/64a13b68b14ab77f9e3eb061/bODii0VbPk-siqftXFVHV.png)

## Model Cards

- [PangolinGuard-Base](https://huggingface.co/dcarpintero/pangolin-guard-base)
- [PangolinGuard-Large](https://huggingface.co/dcarpintero/pangolin-guard-large)

## References

- [1] Clavié, et al. 2024. *Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference*. [arXiv:2412.13663](https://arxiv.org/abs/2412.13663)
- [2] Devlin, et al. 2018. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- [3] Vaswani, et al. 2017. *Attention Is All You Need*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).
- [4] Beltagy, et al. 2020. *Longformer: The Long-Document Transformer*. [arXiv:2004.05150](https://arxiv.org/abs/2004.05150v2)
- [5] Dao, et al. 2022. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135).
- [6] Dao. 2023. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691).
- [7] Li, et al. 2024. *InjecGuard: Benchmarking and Mitigating Over-defense in Prompt
Injection Guardrail Models*. [arXiv:2410.22770](https://arxiv.org/abs/2410.22770).
- [8] Groeneveld et al. 2024. *Accelerating the science of language models*. [arXiv:2402.00838](https://arxiv.org/abs/2402.00838) 
- [9] Hugging Face, *Methods and tools for efficient training on a single GPU* [hf-docs-performance-and-scalability](https://huggingface.co/docs/transformers/v4.49.0/perf_train_gpu_one)