# Correctness contract

`bench run` applies this worker-local contract before any performance sampling.
Trusted `spec.py`,
reference, and candidate code load inside an isolated worker; tensor and callable
objects never cross the JSON protocol. Candidate exceptions, numerical mismatch,
nondeterminism, OOM, unsupported, and correctness hard timeout are gate failures
(exit 1). Import/build protocol damage, worker crash, and artifact write failure
are infrastructure failures (exit 3).

```bash
./bench.sh run --mode correctness --operator OP --candidate CANDIDATE --case CASE --seed SEED
```

Correctness-only rows use `performance_status=skipped` and
`correctness_only_mode`. In performance mode, failed correctness uses
`correctness_gate_failed` and produces no performance samples or ranking input.

Historically, Phase 6 supplied the worker-local library and Phase 7 supplied
the CLI integration. Phase 8 retains the same contract as a mandatory gate
before its performance timers.

## Inputs and state

For each `CaseSpec`, the evaluator creates explicit CPU and requested CUDA
generators from the case seed. The reference spec builds one canonical
`correctness.InputBundle`, then returns independent clones for reference and
candidate. Sharing a mutable container or tensor storage across these clones
is an error. The stable case fingerprint covers the case metadata, generator
contract version, and bounded input structure summary.

An in-place effect is observable only when the mutated object appears in
`InputBundle.observed_state`. Preserve aliases within one clone when an object
is both an argument and observed state; the stock `clone_input_bundle()` does
this. Return values and post-call state are normalized together.

## Normalized outputs

Nested scalar, sequence, mapping, dataclass/namedtuple, tensor, and array
outputs become deterministic leaves such as `output`, `output[0]`,
`output.indices`, and `state.kv_cache`. Every leaf records dtype, shape,
stride, layout, device, and its runtime values. Mapping and state names must be
stable identifiers. Cycles, duplicate/unstable names, and unsupported types
fail with the exact output path.

Structure is checked before numbers: paths, dtype, shape, stride, and layout
must agree. Device is recorded for diagnosis; device placement policy belongs
to the operator execution contract.

## Comparators and tolerances

- `ExactComparator` checks types/codes and reports mismatch count/rate.
- `FloatingComparator` applies `abs(candidate-reference) <= atol +
  rtol*abs(reference)` and reports absolute-error quantiles, relative error,
  RMSE, relative L2, cosine, mismatches, and NaN/Inf counts.
- `TopKComparator` reports ordered/unordered equality, precision/recall@K,
  Jaccard, score error, and tie-aware validity. Duplicate, wrong-length, and
  out-of-range indices are invalid rather than silently collapsed by a set.
- `QuantizedComparator` either requires exact raw codes or uses explicit
  scale, zero point, axis, and layout to dequantize before floating comparison.
  It never guesses scale metadata. Scale/zero-point vectors require an axis;
  optional raw-code bounds are explicit and produce saturation rate in addition
  to zero rate and dequantization error.

Tolerance resolution is `case override > output override > operator default >
dtype fallback`. A formal operator should set `require_explicit=True`; FP8/FP4
have no fallback. Tolerances must be finite, non-negative numbers.

An empty tensor leaf or a leaf with no finite reference/candidate pair has null
distribution, relative-L2, and cosine metrics. When at least one finite pair is
present and both finite vectors are all zero, the practical exact-match
convention is relative-L2 `0` and cosine `1`. A zero reference norm with a
nonzero candidate norm leaves both metrics null. Same-sign infinities match,
and NaNs match only under `equal_nan`; these non-finite matches never masquerade
as zero-vector samples.

Strict JSON output contains finite numbers, null, and—where preserving the
identity of adversarial metadata matters—explicit `NaN`, `+Inf`, or `-Inf`
string tokens. Bounded comparison diagnostics may use null for non-finite sample
values.

## Evaluation and diagnostics

Reference and candidate run on isolated inputs. CUDA devices touched by a call
are synchronized before normalization, so asynchronous launch errors are
classified at the correct stage. Candidate repeats use new clones of the same
canonical input; exact drift is `nondeterministic`, distinct from comparison
`fail`. Exceptions are classified as `timeout`, `oom`, `unsupported`, or
`error`.

A failure diagnostic includes the output path, comparator thresholds and
metrics, bounded worst positions, short exception plus bounded full traceback,
and a stable single-case reproduction command. Full tensors are never copied
into diagnostics.
