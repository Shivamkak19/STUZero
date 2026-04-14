# Spectral Transformer JAX

`spectral-transformer-jax` is a JAX-only training stack for spectral transformer style language, vision, and video models. It implements three complementary sequence mixing mechanisms — **Attention**, **STU (Spectral Temporal Unit)**, and **Mamba** — and systematically ablates their composition across language modeling, image classification, and video action recognition tasks.

The stack keeps the same broad control surface as the Torch codebase: attention/STU/Mamba block families, mixed block types, remat and precision dials, optimizer and scheduler choices, deterministic memmap input order, checkpoint resume, and an export path for later HF or vLLM-oriented inference work.

---

## What Is Implemented

- A JAX-native config system built on OmegaConf dataclasses.
- Flax NNX model modules for attention, STU, Mamba, and mixed blocks.
- A functional JAX training loop using `nnx.split`/`nnx.merge`, `jax.jit`, and `nnx.value_and_grad`.
- TPU-oriented sharding helpers based on `jax.distributed.initialize`, `Mesh`, `NamedSharding`, and `jax.make_array_from_process_local_data`.
- A deterministic memmap data iterator with restart-from-step semantics.
- Orbax checkpoint save and restore.
- A lightweight HF/PyTorch export bundle writer that emits flattened NumPy weights plus config metadata.

---

## Quickstart

```bash
cd spectral-transformer-jax
uv run --with ".[dev]" python scripts/train.py configs/tiny-test/tiny.yaml
```

To enable Weights & Biases logging, pass overrides such as `wandb.enabled=true wandb.mode=offline`. Set `wandb.mode=online` for normal synced runs. WandB is initialized on process 0 only.

To point the trainer at real data, set `data.train_paths` and optionally `data.eval_paths` in your YAML config. Each memmap file is expected to store fixed-length token records of size `max_sequence_length + 1`, so each row can be split into `input_ids` and next-token `labels`.

---

## Experiments

The codebase contains three experiment families, each with their own set of ablations over mixer composition.

### Sequence Mixers

Three distinct mixing mechanisms are combined in all experiments:

| Mixer | Description | Based on |
|---|---|---|
| **Attention** | Standard multi-head scaled dot-product attention with RoPE | [Dosovitskiy et al. 2020](https://arxiv.org/abs/2010.11929), [Su et al. 2021](https://arxiv.org/abs/2104.09864) |
| **STU** | Spectral filtering via Hankel matrix eigendecomposition; fixed convolutional filters derived from the spectral basis | [Agarwal et al. 2023](https://arxiv.org/abs/2312.06837), [Liu et al. 2024](https://arxiv.org/abs/2409.10489) |
| **Mamba** | Selective state space model with input-dependent state transitions (supports Mamba-1 and Mamba-2) | [Gu & Dao 2023](https://arxiv.org/abs/2312.00752), [Dao & Gu 2024](https://arxiv.org/abs/2405.21060) |

Mixed blocks combine these in parallel with configurable contribution ratios (e.g. `attn_ratio=4, stu_ratio=4, mamba_ratio=4` for equal-weight three-way mixing). Layer-wise ablations instead interleave discrete block types across layers.

---

### 1. Language Modeling (150M parameters)

**Task:** Autoregressive language modeling on OLMo-mix v1.6 (Wikipedia + Dolma).  
**Architecture:** OLMo-style transformer, d_model=768, 5 layers, 12 heads, SwiGLU FFN.  
**Training:** 14,300 steps, batch size 1024, sequence length 2048, AdamW + cosine schedule.  
**Eval:** Perplexity on Dolma-wiki and WikiText-103.  
**Config dir:** `configs/head-ratio-ablations/`, `configs/layer-ablations/`, `configs/moe-ablations/`

The OLMo training setup follows [Groeneveld et al. 2024](https://arxiv.org/abs/2402.00838).

#### Head-Ratio Ablations (`configs/head-ratio-ablations/`)

Tests seven block compositions at fixed 150M parameter budget. Each run uses the same number of total "heads" but varies their type:

| Config | Block composition |
|---|---|
| `150m-12a` | 12 attention heads (pure attention baseline) |
| `150m-12m` | 12 Mamba blocks (pure Mamba baseline) |
| `150m-12s` | 12 STU blocks (pure STU baseline) |
| `150m-6a6m` | 6 attention + 6 Mamba (parallel mix, equal weight) |
| `150m-6a6s` | 6 attention + 6 STU (parallel mix, equal weight) |
| `150m-6s6m` | 6 STU + 6 Mamba (parallel mix, equal weight) |
| `150m-4a4s4m` | 4 attention + 4 STU + 4 Mamba (three-way parallel mix) |

The naming convention reflects the ratio values passed to the mixed block (e.g. `6a6s` means `attn_ratio=6, stu_ratio=6`).

#### Layer Ablations (`configs/layer-ablations/`)

Tests whether interleaving distinct block types across layers is better or worse than parallel mixing within each layer:

| Config | Layer sequence (9 layers) |
|---|---|
| `150m-layer-ablation-attn-stu` | `[STU, Attn, STU, Attn, ...]` alternating |
| `150m-layer-ablation-attn-mamba` | `[Attn, Mamba, Attn, Mamba, ...]` alternating |
| `150m-layer-ablation-stu-mamba` | `[STU, Mamba, STU, Mamba, ...]` alternating |

#### MoE Ablations (`configs/moe-ablations/`)

Tests Mixture-of-Experts scaling for each block type. Following [Fedus et al. 2021](https://arxiv.org/abs/2101.03961), a top-k router with load-balancing auxiliary loss replaces the standard FFN. Sweeps over 2, 4, and 8 experts for each base mixer:

| Block type | Configs |
|---|---|
| Attention | `150m-attn-moe-{2,4,8}e` |
| Mamba | `150m-mamba-moe-{2,4,8}e` |
| STU (sandwich) | `150m-stu-moe-{2,4,8}e` |
| 4A+4S+4M (mixed) | `150m-4a4s4m-moe-{2,4,8}e` |

---

### 2. Image Classification (ImageNet-1K)

**Task:** ImageNet-1K top-1 accuracy.  
**Dataset:** 1.2M training images, 1000 classes.  
**Config dir:** `configs/vision-ablations/`

#### ViT-Small Ablations (`vit-s-*`)

Architecture: ViT-S, d_model=384, 12 layers, 6 heads, 16×16 patches, 224×224 images.  
Training: 300 epochs, batch size 1024, AdamW + cosine schedule.

Same seven block compositions as language modeling, plus MoE variants:

| Config | Block composition |
|---|---|
| `vit-s-12a` | Pure attention |
| `vit-s-12m` | Pure Mamba |
| `vit-s-12s` | Pure STU |
| `vit-s-6a6m` | Attention + Mamba (parallel) |
| `vit-s-6a6s` | Attention + STU (parallel) |
| `vit-s-6s6m` | STU + Mamba (parallel) |
| `vit-s-4a4s4m` | Attention + STU + Mamba (three-way parallel) |
| `vit-s-attn-moe-4e` | Attention + 4-expert MoE FFN |
| `vit-s-mamba-moe-4e` | Mamba + 4-expert MoE FFN |
| `vit-s-stu-moe-4e` | STU + 4-expert MoE FFN |
| `vit-s-4a4s4m-moe-4e` | Three-way mixed + 4-expert MoE FFN |

The ViT baseline architecture follows [Dosovitskiy et al. 2020](https://arxiv.org/abs/2010.11929).

#### ViT-5-Base Pretraining and Fine-tuning (`vit5-b-*`)

Architecture: ViT-B, d_model=768, 12 layers, 12 heads. Uses the ViT-5 modernizations from [Wang et al. 2026](https://arxiv.org/abs/2602.08071):
- **QK-Norm**: per-head query/key normalization for training stability
- **LayerScale**: per-layer learnable scalar (init 1e-4) to stabilize deep network training
- **2D RoPE**: two-dimensional rotary positional encoding for image patches
- **Register tokens**: 4 additional non-patch tokens to absorb low-information global states, following [Darcet et al. 2023](https://arxiv.org/abs/2309.16588)

Training: 800 epochs pretraining at 192×192, then fine-tuning at 224×224. LAMB optimizer with binary cross-entropy loss, EMA decay 0.99996, mixup/cutmix augmentation.

Block composition ablations are run at ViT-B scale as well:

| Config | Block composition |
|---|---|
| `vit5-b-pretrain` / `vit5-b-finetune-224` | Attention (full ViT-5 baseline) |
| `vit5-b-12m` | Mamba |
| `vit5-b-12s` | STU |
| `vit5-b-6a6m` | Attention + Mamba |
| `vit5-b-6a6s` | Attention + STU |
| `vit5-b-6s6m` | STU + Mamba |
| `vit5-b-4a4s4m` | Attention + STU + Mamba |

---

### 3. Video Action Recognition (Kinetics-400)

**Task:** Top-1/Top-5 accuracy on Kinetics-400 action classification.  
**Dataset:** ~246K training videos across 400 action classes ([Kinetics dataset](https://github.com/cvdfoundation/kinetics-dataset)).  
**Architecture:** Factorized spatiotemporal transformer — temporal mixing (across frames) followed by spatial mixing (within frames), each using the same configurable block types.  
**Model size:** d_model=384, 12 layers, 6 heads (ViT-S scale).  
**Training:** 30 epochs, batch size 256, 16 frames at 4-frame temporal stride, 16×16 patches on 224×224 frames, AdamW + cosine schedule.  
**Config dir:** `configs/video-ablations/`

The factorized video baseline is motivated by [Pătrăucean et al. 2024 (TRecViT)](https://arxiv.org/abs/2412.14294), which demonstrates that time-space-channel factorization with dedicated blocks for each dimension achieves strong results with far fewer parameters and FLOPs than non-factorized video transformers.

Seven block compositions are tested (same grid as language and vision):

| Config | Block composition |
|---|---|
| `k400-s-12a` | Pure attention |
| `k400-s-12m` | Pure Mamba |
| `k400-s-12s` | Pure STU |
| `k400-s-6a6m` | Attention + Mamba (alternating layers) |
| `k400-s-6a6s` | Attention + STU (alternating layers) |
| `k400-s-6s6m` | STU + Mamba (alternating layers) |
| `k400-s-4a4s4m` | Attention + STU + Mamba (grouped: 4 attn → 4 stu → 4 mamba) |

---

## References

| Paper | Relevance |
|---|---|
| Agarwal, Suo, Chen, Hazan. [Spectral State Space Models](https://arxiv.org/abs/2312.06837). 2023. | Foundation for the STU mixer: spectral filtering via Hankel eigendecomposition with provable robustness guarantees. |
| Liu, Nguyen, Devre, Dogariu, Majumdar, Hazan. [Flash STU: Fast Spectral Transform Units](https://arxiv.org/abs/2409.10489). 2024. | Efficient hybrid STU + sliding-window attention; motivates the STU/attention mixed block designs. |
| Gu, Dao. [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752). 2023. | Mamba-1: selective SSMs with input-dependent state transitions and hardware-efficient parallel scan. |
| Dao, Gu. [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060). ICML 2024. | Mamba-2: unifies SSMs and attention via structured semiseparable matrices; 2–8× faster than Mamba-1. |
| Dosovitskiy et al. [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929). ICLR 2021. | Original ViT: patch-based image transformer baseline for all vision experiments. |
| Wang, Ren, Zhang, Neskovic, Bhattad, Xie, Yuille. [ViT-5: Vision Transformers for The Mid-2020s](https://arxiv.org/abs/2602.08071). 2026. | ViT-5 modernizations used in `vit5-b-*` configs: QK-Norm, LayerScale, 2D RoPE, register tokens, LAMB optimizer. |
| Darcet, Oquab, Mairal, Bojanowski. [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588). ICLR 2024. | Register tokens added to ViT-5 configs to absorb low-information global computations and clean up feature maps. |
| Fedus, Zoph, Shazeer. [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961). JMLR 2022. | Load-balanced top-k MoE routing used in all MoE ablation configs. |
| Groeneveld et al. [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838). ACL 2024. | OLMo architecture and training data (OLMo-mix v1.6) used for all language modeling experiments. |
| Pătrăucean et al. [TRecViT: A Recurrent Video Transformer](https://arxiv.org/abs/2412.14294). 2024. | Factorized time-space-channel video architecture; baseline and design motivation for `k400-*` video configs. |
| CVDF. [Kinetics Dataset](https://github.com/cvdfoundation/kinetics-dataset). | Kinetics-400 dataset used for all video experiments. |

---

## Layout

- `src/stj/config`: config schema and serialization helpers
- `src/stj/models`: OLMo-style JAX model, attention, STU, Mamba, and block composition
- `src/stj/train`: optimizer, scheduler, partitioning, loss, runtime state, and trainer loop
- `src/stj/data`: memmap datasets and deterministic sharded iterators
- `src/stj/checkpoint`: Orbax checkpoint manager
- `src/stj/export`: HF/PyTorch-oriented export bundle writer
- `scripts`: train, eval, export, and legacy import placeholder CLIs
- `configs/head-ratio-ablations`: 7 language model mixer composition configs
- `configs/layer-ablations`: 3 language model layer-interleaving configs
- `configs/moe-ablations`: 12 language model MoE scaling configs
- `configs/vision-ablations`: 16 ImageNet configs (ViT-S + ViT-5-B)
- `configs/video-ablations`: 7 Kinetics-400 factorized video configs

