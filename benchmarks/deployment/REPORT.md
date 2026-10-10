# Resident-model batch inspection replay

**Measured workstation replay, not a deployed factory trial.**

Each backend processed 498 image instances across 2 sessions. Every request recomputed the same complete 83-image pool; these are not distinct new parts.

| Backend | Delivered images/s | Session seconds | First-request seconds | Later-request seconds | Peak allocated GiB |
|---|---:|---|---|---|---:|
| Original MuSc | 2.357 | 105.03, 106.29 | 35.12–35.39 | 34.80–35.49 | 6.29 |
| musc-gpu | 8.929 | 27.90, 27.88 | 9.68–9.74 | 9.06–9.13 | 7.14 |

Delivered-throughput ratio: **3.789×**. Per-session paired speedups: 3.765×, 3.813×.

![Measured delivery timeline and throughput](deployment.png)

Throughput divides actual completed images by total measured session time, including inter-request overhead. No hourly extrapolation, selected warmup discard, or percentile claim from this small sample.

## Setup outside the resident-service clock

| Session | Imports/CUDA setup (s) | Model load (s) | Worker ready to last result (s) |
|---|---:|---:|---:|
| Pair 1 Original MuSc | 13.36 | 4.69 | 123.11 |
| Pair 1 musc-gpu | 5.43 | 4.50 | 37.85 |
| Pair 2 musc-gpu | 5.51 | 4.38 | 37.79 |
| Pair 2 Original MuSc | 5.46 | 4.37 | 116.14 |

## Correctness and scope

Every request passed full-map and score agreement at 2e-5; maximum cross-backend map error was 4.3149543e-06. Repeated outputs were also checked against the first request in their session.

Resident model; request start through CPU maps/scores written and closed as .npy files. Includes image decode/preprocess, transfers, complete-pool inference, RsCIN, shape/finite checks and output serialization to OS-buffered local storage. First request includes first-use kernel setup; no discarded warmup. No ground-truth reads, quality metrics, network transport or camera acquisition. Cumulative session time also includes inter-request instrumentation/cleanup.

Input image content and runtime sources are hashed. Visual-forward and decode counts verify that every request processes every image. No defect annotations are read by the request adapter; zero placeholders satisfy unused upstream bookkeeping.

Runtime code is unchanged. The benchmark adapter uses the existing CLIP preprocessing and pipeline with one shared pool and batch size 4. Removing real ground-truth loading makes this a distinct timing protocol from the published fresh-process benchmark.

## Limits

- Repeated same complete pool, not unique production parts or larger reference pools.
- Finite saturated workstation replay, not factory deployment or long-duration reliability.
- OS disk caches not flushed; buffered output close is not durable fsync.
- No camera, network, PLC or actual production arrivals; all requests ready at session start.
- No inference feature caching; only model weights and runtime allocations remain resident.

A larger unique reference pool needs its own memory and accuracy measurement. Pool fill time, camera transfer, network delivery and PLC deadlines must be added for an actual production line. Two short sessions do not establish long-run stability or tail latency.

Authoritative evidence: [results.json](results.json).
