# FeatherLLM Initial Architecture Design

This document preserves the earliest FeatherLLM architecture artifact recovered from the project workspace: the **FeatherLLM Development Dashboard**. It records the design intent without treating the original dashboard image as the current implementation status.

## Original design intent

The initial design described FeatherLLM as a modular, memory-aware inference engine with on-demand tensor streaming from model checkpoints stored on SSD.

The original flow was:

```text
Model Checkpoint (Safetensors)
        |
        v
Storage Layer
  - bounded file reads
  - Safetensors index / tensor metadata
  - tensor validation
        |
        v
Memory Management
  - RAM / VRAM budgets
  - reusable buffers
  - eviction / release policy
        |
        v
Runtime / Execution
  - CPU / CUDA backend interface
  - tensor operations
  - layer-by-layer execution
        |
        v
Inference
  - scheduling
  - KV cache
  - sampling
```

The design explicitly called out **on-demand tensor streaming** and releasing/reusing buffers as a core mechanism for operating under constrained memory.

## Planned repository structure in the recovered design

The original dashboard proposed these logical areas:

- `include/` — public C++ headers
- `src/` — C++ implementation
- `backends/` — CPU and CUDA backends
- `models/` — model implementations
- `bindings/python/` — Python bindings
- `python/featherllm/` — Python package
- `server/` — serving layer
- `tests/` — unit and integration tests
- `benchmarks/` — performance measurement
- `docs/` — documentation

The current repository uses `include/featherllm/` and focused storage/Safetensors modules instead of copying this historical layout verbatim. That is intentional: the recovered artifact is treated as architectural provenance, not as a requirement to reintroduce unused directories.

## Engineering principles recovered from the design

### Storage bottlenecks

Use bounded/sequential reads, indexing, caching, and fast storage paths rather than loading an entire checkpoint into memory.

### Memory limits

Use strict RAM/VRAM budgets, reusable buffers, and eviction policies.

### Slow inference

Use caching, prefetching, and asynchronous I/O where measurements demonstrate a benefit.

### Compute limits

Use optimized kernels and fused operations only after establishing correctness and benchmark baselines.

### Model compatibility

Keep the runtime modular and model-agnostic; model adapters should not impose a fixed parameter ceiling.

### MoE workloads

Use expert-aware loading/caching and routing once transformer execution and MoE streaming exist.

### Quantization

Evaluate multiple formats empirically rather than assuming a fixed compression or quality tradeoff.

### Concurrency

Bound request scheduling and KV-cache residency so concurrency cannot silently violate memory budgets.

### Engineering discipline

Develop incrementally with tests and benchmarks. Performance and energy claims must come from measurements, not architectural assumptions.

## Current-status correction

The recovered dashboard predates the current implementation. Its original statement that implementation had not started is historical only. The current repository now contains the C++20/CMake storage and Safetensors foundations, checkpoint manifest parsing/validation, bounded tensor reads, host tensor residency, tests, and storage/residency benchmarks developed after that artifact was created.
