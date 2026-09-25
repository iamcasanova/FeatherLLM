# FeatherLLM

Research-grade constrained-memory LLM inference engine.

## Current implementation

The canonical repository currently contains the storage/checkpoint foundation:

- C++20/CMake core
- Safetensors header/index parsing with tensor metadata and byte-range validation
- sharded checkpoint manifests with confined shard paths
- bounded tensor reads
- bounded host-tensor residency with LRU eviction and hit/miss telemetry
- reusable sharded tensor reader
- native tests and storage/checkpoint/residency benchmarks
- Android Gradle/NDK/CMake/JNI build scaffold under `android/`

The transformer/inference runtime is not yet complete. The Android path currently exercises the native storage/checkpoint layer rather than full token generation.

## Build

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

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
10. Android local inference integration

Claims about builds, tests, benchmarks, or energy measurements are only made after they are actually executed and verified.
