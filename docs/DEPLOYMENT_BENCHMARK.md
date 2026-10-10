# Batch inspection delivery benchmark

This benchmark measures a deployment-style workload: a resident model serves a
queue of complete inspection pools and writes the resulting scores and maps to
local storage. It compares the unchanged original and accelerated pipelines on
the same inputs. It is a finite workstation replay, not a factory installation.

## Reproduce

Run from the release root on native Windows with an idle supported CUDA GPU.
Use the environment in the README and a new output directory. The coordinator
uses the existing GPU mutex, supervisor/unload checks and bounded Windows Job
Object process ownership. It will refuse conflicting workloads rather than stop
them. The dataset remains read-only.

```powershell
python tools/benchmark_deployment.py --data-path C:/path/to/mvtec_anomaly_detection --category bottle --requests 3 --pairs 2 --plan-only
python tools/benchmark_deployment.py --data-path C:/path/to/mvtec_anomaly_detection --category bottle --requests 3 --pairs 2 --timeout-seconds 600 --output benchmarks/runs/deployment
python tools/report_deployment.py benchmarks/runs/deployment/results.json --output benchmarks/runs/deployment
python tools/test_deployment.py
```

For the retained measurements and chart, see the
[report](../benchmarks/deployment/REPORT.md) and
[authoritative JSON](../benchmarks/deployment/results.json).

## Workload and measurement

- Two alternating session pairs: original then GPU, followed by GPU then original.
- Each session starts a fresh process, loads one model, and serves three queued
  requests. Each request independently processes the same complete 83-image
  bottle pool. That is 249 image-processing completions per session and 498 per
  backend across both sessions, not 498 distinct images.
- Model weights stay resident. Every image is decoded, preprocessed and encoded
  again; there is no inter-request image-feature cache. Decode and visual-forward
  counts verify all 83 images are processed on every call.
- The first request is included. It pays first-use kernel setup. Later calls may
  reuse compiled kernels, model weights and runtime allocations. No image warmup
  is discarded and OS filesystem caches are not flushed.
- Request service time starts at the inference call and ends when full CPU maps,
  scores and image identifiers have been written and closed. It includes output
  shape/finiteness checks and buffered `.npy` serialization. Closing a file does
  not establish physical-disk durability; no `fsync` is measured.
- The cumulative delivery clock also includes between-request instrumentation,
  cleanup and bookkeeping. Throughput is completed image instances divided by
  the sum of measured session durations. Dots on throughput bars show individual
  session rates; they are not confidence intervals.
- Imports/CUDA initialization and model loading are recorded separately. The
  exact worker-ready-through-last-output time is
  `model_ready_seconds_since_worker_ready + summary.elapsed_seconds`; the raw
  `worker_ready_to_last_result_seconds` field also includes final instrumentation.
  The `result_write_seconds` field includes validation and exception unwinding,
  not just storage I/O.

The request adapter opens only manifest images with the existing CLIP preprocessing.
It supplies zero mask/label placeholders solely for unused upstream bookkeeping,
and intercepts outputs before evaluation metrics. No ground-truth directory or
defect annotation is read during service. This is a distinct scope from the older
fresh-process benchmark, which also loads real annotations; do not compare its
absolute times as if they were paired with this run.

The complete scoring contract remains batch4, 518px CLIP ViT-L-14-336/OpenAI,
layers 5/11/17/23, radii 1/3/5 and one complete shared pool. Runtime model code is
unchanged. The timer includes transfers, aggregation, scoring, interpolation and
RsCIN. Scores/maps pass the 2e-5 gate for every original/GPU request pair; repeated
requests are compared against the first same-backend output to detect state drift.
Failed or incomplete sessions cannot produce a successful throughput chart.

## Evidence and interpretation

The runner records ordered image content hashes, source hashes, environment,
per-request timings, observed memory allocation, GPU dispatch, raw outputs and
guard receipts. It checks image content again after the run. Capacity and free
disk are assessed before CUDA sessions. Each worker has a bounded timeout and
the coordinator owns the GPU mutex across the entire run.

On the measured RTX 3090, musc-gpu delivered **8.929 images/s** versus **2.357**,
or **3.789×** measured throughput. Both GPU sessions completed all 249 instances
in under 28 seconds; each original session took over 105 seconds. The first
original pool completed after 35 seconds, so both GPU sessions delivered three
complete pools before the corresponding original session delivered its first.

This demonstrates more completed processing in less time on this workload. It
does not prove that a larger *single* reference pool fits memory or maintains the
same rate. A larger pool requires its own capacity, agreement and timing run;
duplicating images to enlarge it would alter the scoring task.

For a real line, add measured pool-collection delay, camera and network transfer,
PLC integration, production images, acceptance thresholds, contention, and a
long-duration reliability trial. These two short sessions do not establish p95,
p99, uptime, defect acceptance, or an hour-long sustained rate. No hourly or
larger-pool extrapolation is presented as a measurement.
