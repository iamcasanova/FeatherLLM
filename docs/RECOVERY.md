# Historical Artifact Recovery

This file records the historical-artifact audit performed for the production engineering track.

## Canonical source

Only `iamcasanova/FeatherLLM` is authoritative. Other Feather/FeatherLLM repositories were not used as recovery sources.

## Recovered and integrated

### Initial architecture dashboard

- Source: project Library artifact `FeatherLLM Development Dashboard.png`
- Status: its architectural content has been recovered into `docs/architecture/initial-design.md`.
- The original image is a historical design artifact and is not used as evidence of current implementation status.

### Existing GitHub engineering history

The current `main` history contains the verified production-engineering sequence for:

- checkpoint manifest parsing and validation
- malformed-index rejection tests
- shard path confinement
- Unicode escape handling
- bounded tensor reads
- storage benchmarks
- bounded host tensor residency
- sharded-reader cache integration
- cache hit/miss telemetry
- host-cache benchmarks
- end-to-end cold/resident tensor-read benchmarking

These artifacts are already present on `main` and were audited through the canonical repository history rather than duplicated.

## Referenced but not recovered as source bytes

The project workspace contains other documents and code snippets mentioning unrelated projects (for example CodeBuddy/security tooling). They are not treated as FeatherLLM production artifacts because the available evidence does not establish that they belong to the FeatherLLM repository.

The original dashboard PNG itself could be inspected and its design recovered, but the current GitHub text-oriented write path did not provide a verified binary publication of that exact PNG. The binary artifact is therefore **not claimed as published**. Its recovered architectural content is preserved in `docs/architecture/initial-design.md`.

No source code, checkpoint weights, Safetensors model files, Python interface implementation, milestone archive, or benchmark archive was found in the available historical project-file search with sufficient evidence to publish it as a FeatherLLM artifact. Such artifacts remain unrecovered rather than being fabricated.

## Recovery rule

Future runs must distinguish between:

1. artifact bytes/source verified and published;
2. artifact content verified but represented as a derived documentation record; and
3. artifact referenced but unavailable.

Only category 1 is treated as recovered source code or executable project material.
