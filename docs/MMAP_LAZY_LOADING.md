# Lazy tensor loading

FeatherLLM now has an explicit OS-backed lazy tensor access path.

## Flow

`ShardedTensorReader::view_tensor()` resolves a tensor through the checkpoint manifest and then:

1. Returns the existing immutable `HostTensorCache` storage when the tensor is resident.
2. Otherwise maps only the tensor's byte range from its shard.
3. Returns a `TensorView` that owns the mapping for its lifetime.

The operating system can page mapped tensor data on demand instead of requiring a full tensor copy into a new `std::vector`.

## Platform behavior

- Linux/POSIX: `mmap(MAP_PRIVATE)` with page-aligned offsets.
- Windows: `CreateFileMappingW` + `MapViewOfFile` with allocation-granularity alignment.
- Requested ranges are validated against the shard file size before mapping.
- Empty ranges are represented without creating an OS mapping.

## Relationship to residency

The existing host cache remains the explicit bounded RAM residency mechanism. Lazy mapping is the uncopied storage-access path. A later residency/prefetch layer can decide when mapped pages should be faulted in, retained, prefetched, or evicted.

## Verification

Native unit coverage includes:

- unaligned subrange mapping
- exact-range access
- empty mapping
- out-of-range rejection
- ShardedTensorReader integration

The standalone mapped-file implementation was also compiled and exercised locally with C++20, warnings enabled, after fixing empty-region construction.

Full repository CI remains the authoritative validation gate.
