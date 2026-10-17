# Captix ONNX runtime operations

Current promotion decision: **HOLD**. Real runtime benchmarks and labeled
accuracy are **NOT MEASURED**. No approved real checkpoint or private labeled
corpus is included. Synthetic tests exercise reporting and gates; their timings
and accuracy provide no evidence about the trained model. **Functional export
has not been validated against a real checkpoint.**

## Artifacts and service

Use an authorized local safetensors checkpoint at an immutable model revision.
Neither benchmarking nor startup downloads models, images, or CAPTCHAs. Export
on a trusted machine, then use `scripts/validate_onnx.py --help` and validate the
checkpoint/artifact pair before packaging:

```sh
uv run --extra export python scripts/export_onnx.py \
  --source-dir /private/checkpoint --out-dir /private/captix-v1 \
  --revision-sha MODEL_COMMIT_SHA --source-commit CAPTIX_COMMIT_SHA
```

Place verified artifacts at the Docker build's `models/model.onnx` and
`models/manifest.json` locations.
Supply `CAPTIX_API_TOKEN` through the operator's secret environment. Run
`docker compose -f deploy/compose.yaml build` and
`docker compose -f deploy/compose.yaml up -d`. The compose service binds to
localhost, limits CPU/RAM to 2 CPUs/1 GiB, uses an unprivileged user, read-only
filesystem, bounded tmpfs, dropped capabilities and a 35 second stop grace.
`CAPTIX_MODEL_DIR` selects local artifacts and `ORT_INTRA_OP_THREADS` defaults
to 2. `/health/live` reports liveness and `/health/ready` readiness after warmup.
`POST /v1/ocr` requires bearer authentication and multipart image input.
Saturation returns 429; callers should bound retries. Stop with
`docker compose -f deploy/compose.yaml stop`, preserving the grace interval.

`/metrics` is disabled by default. Enable with `CAPTIX_METRICS_ENABLED=true`
and a distinct `CAPTIX_METRICS_TOKEN`. Metrics contain aggregate durations and
bounded status labels, never images or recognized text. Never put secrets in
reports or manifests.

## Authorized held-out corpus

Build existing `CorpusRecord(path, label, source, license, sha256, mask=None,
length=None, case_sensitive=True)` entries and call
`freeze_corpus(records, Path('/private/test/manifest.json'), authorized=True,
split='test')`. Authorization and separation from training/calibration are
operator attestations; defaults are unconfirmed and cannot pass promotion.
The manifest contains private labels and paths and must stay local. Corpus
images are referenced, never copied. Duplicate bytes, missing frozen hashes,
changed contents, and invalid constraints are rejected.

Include varied widths, transparency, repeated characters, greedy/fixed-length
decoding and masks. Constraints must reflect known task metadata, rather than
being derived from the expected label solely to improve scores. Every unique
sample is evaluated for exact-match once per repetition, independently of timed
request count. A correctness change between repetitions invalidates the run.

## Fair benchmark procedure

Install `uv sync --extra benchmark --extra export`. Provision one isolated Linux
amd64 CPU environment with identical source, dependencies, affinity, **2 vCPU
quota / 1 GiB memory quota** for all variants. With cgroups v2 and a delegated
user systemd manager, an example is:

```sh
export CAPTIX_BENCH_CHECKPOINT_DIR=/private/checkpoint
export CAPTIX_MODEL_DIR=/private/captix-v1
export CAPTIX_BENCH_ISOLATED=true
systemd-run --user --scope -p CPUQuota=200% -p MemoryMax=1G \
  taskset -c 0,1 uv run --extra benchmark python benchmarks/bench_runtime.py \
  --variant pytorch-original --corpus /private/test/manifest.json \
  --out benchmarks/output/pytorch-original.json
```

Choose two CPUs allowed by host affinity; `0,1` is an example. A separate
benchmark container with `--cpus=2 --memory=1g --cpuset-cpus=... --network=none`
is also suitable after provisioning dependencies offline. The production
ONNX image excludes Torch and benchmark tooling. The runner reads its process
cgroups v2 scope and tighter inherited quotas; unavailable limits (including
cgroups v1) remain null and hold promotion. Confirm no other significant load;
the isolation flag is an operator attestation.

Run `pytorch-original`, `pytorch-compatible`, and `onnx-fp32` separately using
the same corpus, host, quotas, affinity, versions, commit and seed. Original
uses original transforms and CTC; compatible substitutes the exporter's height
mean pooling. All variants use CPU FP32, B=1, two intra-op threads and one
inter-op thread. Set `CAPTIX_BENCH_DOCKER_IMAGE` for each variant to its actual
locally built deployable image. Docker inspection records uncompressed bytes
without pulling. A PyTorch deployable image is an operator prerequisite; the
repository supplies only the ONNX production Dockerfile. Missing Docker/image
leaves its size unavailable; never substitute the candidate size for baseline.

Official runs have **100 warmups / 1000 measured requests / 3 independent
subprocesses**, seed 0 by default. Each subprocess makes its first prediction
for cold start, then performs the 100 warmups. Each repetition uses the same
seeded shuffled sequence. CLI count overrides mark a run exploratory and block
promotion. Images are preloaded; timed pipeline requests exclude file reads,
HTTP, authentication and transport. Cold start includes spawn, imports, input
loading, model initialization and the first prediction.

Reports include per-run p50/p95/p99 milliseconds for preprocessing, inference,
CTC (`decode`), and total, requests/s, Linux high-water RSS, PSS sampled from
`/proc/PID/smaps_rollup` every 10 ms, cold start and Docker bytes. RSS/PSS include
startup and accuracy evaluation. Sampled PSS is a lower bound on transient
peaks and is null when unavailable. Summaries give median, min, max and sample
standard deviation across independent runs, including each run's percentiles
rather than pooled percentiles. Metadata records versions, commit, hashed host
identity, CPU, physical RAM, kernel, seed, affinity, thread counts and quotas.

Raw local JSON includes hashes, sources and correctness booleans for pairing;
it contains no image bytes, paths, labels, predictions or tokens. Comparison
JSON/CSV/Markdown omit per-sample records. Source identifiers/hashes can still
identify private datasets; review files before sharing. Worker stderr is
suppressed, producing a sanitized `worker_failed` reason on failure.

## Operational evidence and promotion

Run actual API/container smoke, network-disabled startup/inference, payload
limits, authentication, saturation and graceful shutdown checks on the same
code/artifacts. Golden validation requires 100% deterministic textual equality;
retain numerical parity tolerances and validation output. Synthetic tests alone
do not establish parity for a trained checkpoint.

Set `CAPTIX_BENCH_EVIDENCE` to an operator-reviewed local JSON file:

```json
{
  "commit": "40-character Captix commit",
  "source_sha256": {
    "model.safetensors": "64-character SHA256",
    "config.json": "64-character SHA256"
  },
  "corpus_sha256": "64-character manifest SHA256",
  "checks": {
    "golden": null, "api_smoke": null, "container_smoke": null,
    "offline": null, "payload": null, "auth": null,
    "saturation": null, "shutdown": null
  }
}
```

Replace null with true only after the actual check passes, or false on failure.
Commit, source hashes and corpus digest must match the run. These are trusted
operator attestations, not signed audit records. After all three measurements:

```sh
uv run --extra benchmark python benchmarks/compare_results.py \
  --baseline benchmarks/output/pytorch-original.json \
  --candidate benchmarks/output/onnx-fp32.json \
  --out benchmarks/output/comparison
```

This writes `.json`, `.csv` and `.md`, distinguishing null/unavailable from zero.
The compatible report remains diagnostic and cannot replace the original
baseline. Promotion requires all of:

- Actual official measurements on the same isolated host/profile with matching
  checkpoint, corpus, versions, seed and source commit.
- **At least 1000 authorized unique held-out labeled samples**, with paired
  correctness matching actual measured results. Less evidence means **HOLD**.
- Overall ONNX minus PyTorch exact-match in **[-0.2, +0.2] percentage points**,
  inclusive. A **-0.3 p.p.** change means **HOLD**.
- No source with **at least 100 samples** losing **more than 1 p.p.**; a
  **-2 p.p.** change for 100 samples means **HOLD**, even at zero overall delta.
  There is no automatic review override.
- Golden equality and all listed operational checks passing.
- **p95 <= 0.90x baseline or peak RSS <= 0.80x baseline**, with the other
  metric **<= 1.05x baseline**. Thus p95 0.89x/RSS 1.05x passes value only when
  accuracy and all operational checks also pass.

Accuracy reports include sample counts and a seeded 10,000-draw paired
percentile bootstrap 95% confidence interval for the delta, overall/per source.
Resampling counts of paired differences preserves pairing. Confidence intervals
supplement the fixed product thresholds; significance never replaces policy.
Stretch goals (-30% p95, +50% throughput, -30% RAM) do not decide promotion.

Without approved artifacts, Docker or authorized labeled inputs, record blocked
checks and **HOLD / NOT MEASURED**. Retain the PyTorch reference until evidence
passes review. The tools never deploy or promote a model automatically.
