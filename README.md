# FeatherLLM

Research-grade constrained-memory LLM inference engine.

## Current implementation

The repository currently contains the first storage/checkpoint-engine milestone: a C++20 Safetensors reader with header/index parsing, tensor metadata validation, bounded tensor reads, and a minimal executable test.

## Build

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

The runtime remains model-agnostic and is intended to expose a C++ core with a Python interface as subsequent milestones are implemented.

## Engineering direction

1. Safetensors indexing and bounded reads
2. Sharded checkpoint manifests
3. mmap/lazy loading and residency management
4. Async prefetch
5. Transformer execution
6. MoE streaming
7. Compute/memory reduction
8. FeatherCodec context compression
9. Energy telemetry and optimization

Claims about builds, tests, benchmarks, or energy measurements are only made after they are actually executed and verified.
