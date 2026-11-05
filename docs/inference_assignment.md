# Inference systems extension

Build an inference engine with the functionality of mini-sglang, beginning with your CS336 Transformer and ending with concurrent, streaming, multi-GPU serving. You implement the algorithms. The assistant maintains interfaces, fixtures, adapters, reference comparisons, and benchmark plumbing. A trained TinyStories run is not a prerequisite: small random models establish numerical correctness; pretrained weights establish integration behavior later.

The target is **mini-sglang commit `9a91cfafe754aa85daee49998176275667eb58f2`**, not a moving `main` branch or the much larger SGLang project. The [pinned source](https://github.com/sgl-project/mini-sglang/tree/9a91cfafe754aa85daee49998176275667eb58f2) and `inference_tests/upstream/source.json` define the reference inventory. Completing the first exercise does not complete this extension.

## Working agreement and scope

Keep the existing training path and assignment tests working. Put inference implementations in `cs336_basics/inference/`. Reuse weights and mathematical primitives; introduce explicit position IDs and cache-aware attention rather than forcing every inference optimization through the training API. Tests interact through `inference_tests/adapters.py`, which translates interfaces and must not contain the solution.

Functional parity includes request state, sampling, prefill/decode, variable-length batches, paged memory, naive and radix caches, chunked prefill, admission and cancellation, backend selection, custom memory kernels, CUDA graphs, CPU/GPU overlap, tensor parallelism, supported model loading and MoE, worker messages, tokenization, detokenization, offline generation, API serving, shell, configuration, and benchmarks. Preserve the relevant behavior and invariants; copying private class names or the original code layout is unnecessary.

The reference API declares fields such as `n`, `stop`, and repetition penalties that are not forwarded into its sampling implementation. Do not silently claim these work because the request schema accepts them. For this extension reject unsupported nondefault values, or implement them as separately tested additions. The registered Mistral3 name is not evidence of a working vision pipeline. Speculative decoding, quantization, LoRA, disaggregated prefill/decode, expert parallelism, and multimodal serving are additional projects, not requirements inferred from this reference.

## Test commands and current coverage

The scaffolding currently supplies M1 sampling and generation tests, plus six adapted upstream cache allocation tests for M4/M6. Later milestones below are acceptance specifications, **not implemented test suites yet**. Add their tests through the adapter as you reach them.

```bash
python -m pytest inference_tests --inference-stage=sampling -q
python -m pytest inference_tests --inference-stage=generation -q
python -m pytest inference_tests --inference-stage=cache -q
python -m pytest inference_tests --inference-stage=all -q
python -m pytest tests -q
```

Without `--inference-stage`, these new exercises are skipped, leaving ordinary test runs usable while the exercises are unfinished. Explicitly enabled unfinished stages fail with `NotImplementedError`; they are not marked as passing or expected failures. `all` means all tests currently supplied, not full engine certification.

The original cache test assertions are retained. Only the constructor, global-context setup, imports, and type hints are adapted. A compatibility view can expose `_allocate`, `_page_to_token`, `free_slots`, `prefix_cache.insert_prefix`, and `check_integrity` without imposing those names on your engine. The MIT license is retained beside the adapted test. Some upstream cases deliberately bypass normal ownership accounting to reproduce an eviction bug; do not treat every setup as a production request lifecycle.

Upstream tests are a mixture of pytest cases, directly executed scripts, GPU kernel checks, and a scheduler demonstration that loads Llama weights. They do not comprehensively test generation or server correctness. See the upstream coverage table below before trying to run everything as pytest.

## Milestone 1 Sampling and reference generation

Baseline prerequisite before model-level inference comparisons: the current regression run
has 19 passing tests and three failures in the selected model/utilities/optimizer/data/checkpoint suites.
`test_multihead_self_attention` and `test_multihead_self_attention_with_rope` fail because the
adapter uses `o_proj.weight` instead of the module's `output_proj.weight`.
`test_rope` fails because `x.shape[-3]` is not the sequence dimension for its 3D input.
Resolve these before M2; the synthetic M1 sampling/generation exercises can proceed independently.

**Deliverables:** implement `sampling_probs`, `sample_tokens`, and `generate_tokens` in the supplied stubs. Then connect your tokenizer and checkpoint loader through a generation CLI. Start with token IDs and synthetic logits; model quality is irrelevant to these tests.

Assignment contract for sampling:

- Input logits have shape `[batch, vocab]`. Return float32 probabilities on the same device, with every row summing to one. Do not mutate input logits. Compute normalization in float32.
- Require finite nonnegative temperature; `top_k` is `-1` or an integer in `[1, vocab]`; `top_p` is in `(0, 1]`. Reject invalid parameters with `ValueError`.
- Temperature zero selects the largest logit directly; ties choose the lowest token ID. It must not divide by zero. Otherwise divide logits by temperature.
- Keep the highest `k` logits when top-k is enabled. Use a stable token-ID tie rule for reproducible local tests. Normalize, then keep the smallest probability-sorted prefix whose cumulative probability reaches or exceeds top-p. Keep the crossing token and at least one token; renormalize.
- Our explicit joint-filter contract is **top-k then top-p**. This specifies the exercise, rather than promising identical randomness or tie behavior to every fused backend. Later backend comparisons must document any semantic differences.
- `sample_tokens` returns int64 `[batch]`, using the supplied device-compatible `torch.Generator`. Repeatability is tested with the same generator seed. Later add per-request parameters and RNG streams so changing a batch does not couple independent requests.

Reference generation accepts one nonempty int64 prompt, calls a logits-producing model on the full prefix, samples from the last position, and returns prompt plus generated IDs. Include a sampled EOS before stopping unless `ignore_eos=True`; the prompt ending in EOS does not itself consume generation budget. Zero new tokens returns the prompt without a model call. Reject negative budgets and prompt length plus requested output beyond context capacity. Do not silently crop context or reset RoPE positions. Run without gradients; the caller sets evaluation mode. These are explicit local conventions; an offline user-facing API can strip EOS when formatting output.

**Pass gate:** the supplied sampling and generation tests; add empty-prompt, negative-budget, noncontiguous logits, device, dtype, and empirical-distribution checks. Then verify greedy output against direct full-prefix evaluation of a tiny instance of your Transformer.

**Explain:** why probabilities need float32; why nucleus filtering keeps the threshold-crossing token; why only last-position logits produce the next token.

**Reference:** `core.py`, `engine/sample.py`, `llm/llm.py`.

## Milestone 2 Incremental decoding and contiguous KV cache

**Deliverables:** separate `prefill` and `decode`; retain K and V per layer with explicit sequence lengths and positions. A minimal interface is `prefill(ids) -> (last_logits, state)` and `decode(next_ids, state) -> (next_logits, state)`. The state represents tokens already processed, not tokens merely sampled.

Implement cache-aware causal attention for unequal query and key lengths. Query position `p` may see keys with positions at most `p`; a plain square triangular mask is insufficient for cached decoding. Rotate each new Q/K with its absolute position exactly once. Prefill computes all prompt KV; each decode step appends only the new token's KV. Validate capacity before writing.

Your existing Transformer generates positions from zero and MHA has a hardcoded RoPE capacity. Replace those assumptions on the inference path. Document cache shape, device, dtype, and ownership. Reset/free one request without affecting another.

**Pass gate:** cached next-token logits versus full-prefix logits at every step; prefill as one chunk versus several; single-token prompt; exact capacity; different initial offsets; two independent requests; no writes to previous KV slots; unchanged training tests. Use float32 first and record looser, measured tolerances for BF16. Greedy parity should be checked on cases without near-tied logits.

**Explain:** how prefill and decode differ in arithmetic intensity; the memory formula `2 * layers * tokens * kv_heads * head_dim * element_size`.

**Reference:** `layers/attention.py`, `layers/rotary.py`, `attention/base.py`, `kvcache/mha_pool.py`.

## Milestone 3 Request state and variable length batching

**Deliverables:** request IDs, sampling parameters, input and output lengths, cached length, finish status, and a batch descriptor carrying token IDs, positions, offsets and output locations. Pack ragged prefill inputs; decode one token per live request. Keep token storage separate from scheduler metadata.

Define transitions from queued to prefill to decode to finished/aborted. Sampling happens only after a request's final prompt chunk. Associate every result with its request ID even when batches reorder. Keep finished requests out of later computation.

**Pass gate:** ragged batch equals independent inference; batch permutation changes no request association; mixed EOS and length completion; zero live requests; no cross-request attention; greedy and sampled requests in the same batch. Test state transitions without a GPU using a fake engine.

**Reference:** `core.py`, `scheduler/table.py`, `scheduler/utils.py`, `engine/engine.py`.

## Milestone 4 Paged KV storage and allocation

**Deliverables:** a fixed-capacity KV pool, request-to-token/page table, free-page allocator, gather/scatter reference operations, and a naive cache manager. Distinguish a physical page ID, a page-aligned token offset, and a logical sequence position. Document these units in every interface.

A partially filled page must not be reused by another live request. Allocation either succeeds completely or leaves ownership unchanged. Release on completion and cancellation; prevent duplicate frees. Test page sizes greater than one early. For ownership accounting, include free, active, cached-evictable and cached-pinned memory without double-counting shared pages.

**Pass gate:** allocation/exhaustion/free/reuse; page sizes 1, 4, 8, 16; partial final pages; arbitrary noncontiguous physical placement; no overlapping live allocations; randomized allocate/free traces against a set-based oracle; paged attention logits equal the contiguous baseline. The adapted eviction suite becomes fully runnable after M6 adds prefix eviction.

**Reference:** `kvcache/mha_pool.py`, `scheduler/cache.py`, `scheduler/table.py`, upstream `tests/core/test_cache_allocate.py`.

## Milestone 5 Continuous batching and chunked prefill

**Deliverables:** pending and running queues, admission control, prefill token budget, maximum live requests, decode scheduling, and chunk continuation. Admit requests while older ones decode. Reserve enough pages for the chosen scheduling policy, including page rounding. Avoid busy loops when a request can never fit: return a clear capacity error.

Start with an explicit non-overlapped schedule. Document prioritization of waiting prefills versus ongoing decode, and a fairness policy. A chunk boundary must not emit a sampled token before the prompt is complete. Handle cancellation while queued, partially prefilling, and decoding. Free table slots, cache references and reserved capacity in each case.

**Pass gate:** deterministic fake-engine traces for arrivals, memory pressure and cancellation; total scheduled tokens within budget; no sampling from intermediate chunks; progress under mixed short/long prompts; continuous batch logits equal serial execution. Compare chunked and unchunked results.

**Reference:** `scheduler/prefill.py`, `scheduler/decode.py`, `scheduler/scheduler.py`, `scheduler/config.py`.

## Milestone 6 Radix prefix caching

**Deliverables:** token-keyed compressed radix tree; longest-prefix lookup; edge splitting on partial match; insertion; handles/refcounts; pin/unpin; eviction of unpinned entries; integration with the page allocator. Keep a naive cache option for comparisons.

Reuse only valid, computed KV. Align reusable regions to page boundaries where necessary. A matching complete prompt may still require computing its final token to obtain logits; do not assume KV alone contains next-token logits. Do not share a writable partial tail without an explicit ownership/copy policy. When two requests insert overlapping prefixes, reconcile duplicate physical storage without leaking pages or freeing pinned storage.

**Pass gate:** exact/shared/divergent/no-prefix matches; partial-edge split; duplicate insertion; nested pins; unpin and eviction; active-prefix protection; shared prefixes followed by different continuations; repeated cancellation; allocator integrity under randomized traces. Pass all six adapted upstream page-alignment tests.

**Explain:** which tokens save prefill work, why pin counts matter, and how token counts convert to page counts during eviction.

**Reference:** `kvcache/base.py`, `kvcache/naive_cache.py`, `kvcache/radix_cache.py`, `kernel/radix.py`, `kernel/csrc/src/radix.cpp`.

## Milestone 7 Attention backends and custom kernels

**Deliverables:** backend interface for metadata preparation, prefill and decode; Torch correctness backend; adapters for FlashAttention, FlashInfer and TensorRT-LLM attention where the installed GPU/software supports them. Support separate prefill/decode selection and reject incompatible page sizes or head dimensions explicitly.

Implement the reference project's indexing and KV-store operations in Triton or CUDA, with Torch oracles. Exercise masked/padded entries, noncontiguous inputs where supported, and page-boundary writes. Understand tensor descriptors, device ownership and FFI lifetimes if reproducing the C++/CUDA bindings. A C++ radix implementation may replace the Python tree after the same behavior tests pass.

**Pass gate:** backend logits agree with the reference path on supported configurations; offset causal masks; grouped-query attention; multiple page sizes; exact store/index results; repeated kernel execution without memory growth; compute-sanitizer runs for custom CUDA kernels. Compare memory throughput against the local Torch baseline rather than an arbitrary published number.

This is integration parity: upstream also delegates attention math to specialized libraries. Writing a FlashAttention kernel from scratch is a separate optional deep dive, not needed to reproduce the reference's functionality.

**Reference:** `attention/{base,fa,fi,trtllm,utils}.py`, `kernel/{index,store,tensor,utils}.py`, `kernel/csrc/`, upstream kernel tests.

## Milestone 8 CUDA graph capture and replay

**Deliverables:** warmup, static buffers, decode batch-size buckets, graph capture/replay, padding and eager fallback. Keep addresses stable; update values before replay. Dummy padded requests must not read or overwrite real requests' slots. Integrate attention backend graph metadata and make graph capture configurable.

**Pass gate:** eager versus replay logits; changing input values with fixed buffers; transitions between batch sizes; uncaptured-size fallback; EOS/cancellation with padding; no growth across repeated replays. Measure CPU launch overhead and decode latency after warmup, excluding capture cost from steady-state throughput but reporting it separately.

**Reference:** `engine/graph.py`, `engine/engine.py`, attention backend graph methods.

## Milestone 9 Overlap CPU scheduling and GPU execution

**Deliverables:** streams/events, host/device staging buffers, delayed result processing, and a schedule that prepares the next batch while the current one executes. Keep an explicit no-overlap mode. Distinguish device completion from host-visible token availability.

Define ownership and lifetime for each asynchronous buffer. Do not recycle pages, token slots, or pinned host buffers until readers finish. Handle cancellation during in-flight work and when the next scheduled batch references a request whose EOS is not yet known.

**Pass gate:** same greedy outputs and cleanup as non-overlapped scheduling; adversarial completion/cancellation traces; stress with tiny buffers; profiler timeline showing real overlap; no per-token global synchronization in the hot path. Report throughput and latency separately, since improvement is workload-dependent.

**Reference:** `scheduler/scheduler.py`, `scheduler/io.py`, `engine/engine.py`, `env.py`.

## Milestone 10 Tensor parallel execution

**Deliverables:** rank/world configuration, process groups, row- and column-parallel linear layers, vocab-parallel embedding and LM head, correct collectives, sharded weight loading, and rank-consistent request scheduling/sampling. Start with Torch distributed, then implement/integrate the reference PyNCCL path for graph-compatible communication.

Handle GQA/MQA KV head partitioning or replication explicitly. Ensure residual additions and biases are applied exactly once. Define which rank owns sampling and how selected token IDs reach the others. Fail early for unsupported dimension divisibility; bound waits on worker failure and tear down process groups.

**Pass gate:** tiny single-rank versus two-rank logits and greedy outputs; sharded weights reconstruct the originals; duplicate KV-head cases; repeated request traces; collective failure/shutdown handling; graph replay with communication. Multi-GPU tests must run explicitly with selected devices, not during ordinary CPU pytest.

**Reference:** `distributed/`, `layers/{linear,embedding,base}.py`, `models/weight.py`, `kernel/pynccl.py`, upstream `tests/kernel/test_comm.py`.

## Milestone 11 Pretrained architectures and mixture of experts

**Deliverables:** registry and config parsing; local and downloaded weight sources; safetensor shards; dtype handling; dummy weights; Llama, Qwen2/2.5, Qwen3, Qwen3 MoE and the supported Mistral text path. Read actual per-architecture details: GQA, projection biases, Q/K normalization, RoPE conventions, tied weights and head dimensions. Do not apply one architecture's assumptions to every checkpoint.

For MoE implement routing, top-expert selection, routing-weight normalization, token dispatch, expert computation and weighted combine. First use simple Torch operations; then connect the fused backend and custom dispatch kernels. Validate zero-token experts and repeated expert selections across a batch. Match the reference's parallel strategy; expert parallelism is not implied by tensor parallel support.

Support Hugging Face and ModelScope resolution via plumbing adapters, without embedding network calls in model forward. The TinyStories tokenizer and model remain a separate supported model family; HF chat templates and vocabulary are model-specific.

**Pass gate:** tiny random configurations versus Transformers eager logits, followed by one small public pretrained checkpoint; full-prefix and cached logits; tied and untied heads; shard mappings; single-rank and TP; MoE output against an explicit expert-loop oracle. Record unsupported config options rather than silently approximating them. No download of large models is necessary for unit tests.

**Reference:** `models/`, `layers/{norm,activation,rotary,moe}.py`, `moe/`, `kernel/moe_impl.py`, `kernel/triton/fused_moe.py`.

## Milestone 12 Worker processes and serving interfaces

**Deliverables:** typed request/reply/abort/shutdown messages; serialization; tokenizer and incremental detokenizer; ZMQ or equivalent queues; worker lifecycle; async frontend; offline `LLM.generate`; CLI and interactive shell with conversation history and reset. Reproduce shared/separate tokenizer-worker modes and rank-zero routing.

Implement `/generate`, `/v1/chat/completions` with streaming and nonstreaming responses, `/v1/models`, and the `/v1` status route. Make request IDs unique and map results correctly. Apply the appropriate tokenizer chat template. Stream valid incremental UTF-8 without losing text when byte sequences span tokens. Finish exactly once; propagate disconnects to cancellation and free resources. Distinguish length exhaustion from EOS in finish reasons and report actual usage. These last checks deliberately improve on reference placeholders rather than copying them.

**Pass gate:** message round trips including nested batches and tensors; offline results preserve input order; streamed concatenation equals nonstreamed output; Unicode boundaries; concurrent clients; disconnect cleanup; invalid-input errors; worker startup failure; shutdown without orphan processes; shell reset; slow consumer/backpressure. Use fake engines for most API tests, then a tiny real model. Do not expose a network server until local integration checks pass.

**Reference:** `message/`, `tokenizer/`, `server/`, `llm/`, `shell.py`, `utils/mp.py`, `__main__.py`, upstream `tests/misc/test_serialize.py`.

## Milestone 13 Performance experiments and completion audit

**Deliverables:** reproducible offline and online benchmarks, machine-readable results, NVTX ranges/profiler traces, and a feature coverage ledger. Include model revision, dtype, device, input/output length distributions, request concurrency/arrival rate, warmup, memory use and software versions.

Measure time to first token, inter-token latency, end-to-end latency, p50/p95/p99, input/output tokens per second, requests per second, prefix-hit fraction and peak allocated/reserved GPU memory. Specify whether throughput includes prefill. Distinguish CUDA-event kernel timings from wall-clock request timings. Keep prompts and generation lengths comparable across engines; warm-cache and cold-cache results must be separate.

Ablate contiguous versus paged KV, naive versus radix cache, full versus chunked prefill, eager versus graph, overlap off/on, backend choice and TP degree. A slower optimization is a result to explain, not a reason to change the workload until it wins. Do not require parity with published throughput on different hardware.

For completion, every module group below needs a corresponding implementation or documented architectural replacement, correctness evidence and applicable benchmark. Every unsupported setting must fail explicitly. Correctness, cancellation, resource ownership and reproducible tests precede throughput claims.

**Reference:** `benchmark/`, top-level `benchmark/`, `utils/{logger,arch,torch_utils,misc,registry,hf}.py`, `env.py`.

## Upstream coverage and adapter strategy

| Upstream source or test | Local milestone and treatment |
| --- | --- |
| `tests/core/test_cache_allocate.py` | M4/M6; six tests already adapted through `make_cache_manager`; MIT notice preserved |
| `tests/core/test_scheduler.py` | M5/M12; process demonstration requiring model weights; replace fixed model/ports with fixtures and assert outputs, termination and cleanup |
| `tests/kernel/test_index.py` | M7; preserve reference indexing assertions; adapt backend imports and device setup when kernels exist |
| `tests/kernel/test_store.py` | M7; preserve KV-store comparison; separate performance measurements from correctness |
| `tests/kernel/test_tensor.py` | M7; low-level FFI/device descriptor checks, relevant if exposing comparable bindings |
| `tests/kernel/test_comm.py` | M10; explicit multi-process GPU test/benchmark; compare collectives on selected devices |
| `tests/misc/test_serialize.py` | M12; message equality/round-trip tests through serialization adapters |
| `core`, `engine/sample`, `llm` | M1/M3/M12; local semantic tests are required beyond upstream smoke coverage |
| `kvcache`, `scheduler/cache`, `scheduler/table` | M2/M4/M6; memory accounting and randomized lifecycle tests |
| `scheduler/prefill`, `decode`, `scheduler`, `io`, `utils`, `config` | M3/M5/M9; deterministic traces plus GPU integration |
| `attention`, `engine/engine`, `engine/config`, `engine/graph` | M2/M7/M8; numeric comparisons and replay correctness |
| `distributed`, `layers`, `models`, `moe`, `kernel` | M7/M10/M11; model/backend/collective correctness and performance |
| `message`, `tokenizer`, `server`, `shell`, `__main__`, `utils/mp` | M12; process, API, streaming and shutdown tests |
| `benchmark`, remaining `utils`, `env`, packaging | M13; configuration, measurement and reproducibility plumbing |

The pinned upstream dependency specification requires `torch<2.10.0`, while this assignment specifies `torch~=2.11.0`. Do not install the upstream package into the assignment environment. Use a separate reference environment for upstream runs, and exchange small saved inputs/outputs or benchmark JSON. The local adapted CPU tests do not require importing `minisgl` or its CUDA extensions. Numerical equality tests use deterministic inputs; stochastic samplers are compared by distribution/support, not matching implementation-specific random draws.

## Start here

Implement `sampling_probs` first and run the sampling stage. Then implement `sample_tokens` and the uncached `generate_tokens` loop. We will review those before introducing the KV cache. The uncached path remains your numerical oracle for the entire project.
