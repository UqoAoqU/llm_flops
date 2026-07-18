# Troubleshooting

Start with the mirrored evaluation's `summary.md`, `logs/worker.jsonl`, bounded
stdout/stderr, and diagnostics. Do not infer success from compiler activity or
from a process that has not produced a terminal artifact.

## Import, JIT, and build

`worker.jsonl` identifies the active stage and heartbeat. Cold SGLang,
FlashInfer, DeepGEMM, Ninja, NVCC, or PTXAS work may be long but belongs to
import/build/first-call/graph-capture, never steady-state samples. A heartbeat
does not extend the hard stage deadline.

If `Python.h`, `ninja`, `rustc`, `cargo`, or CUDA compiler tools are missing,
repair the managed environment and rerun. Do not move JIT work into the timed
candidate body or classify a compiler failure as unavailable performance.

## Timeout, OOM, crash, and interruption

- `error`: structured Python/import/build exception;
- `timeout`: independent stage deadline exceeded; process group is terminated;
- `oom`: recognized CPU/CUDA out-of-memory condition;
- `crashed`: signal/nonzero exit without a valid response;
- `unsupported`: explicit unsupported contract/backend;
- evaluation `interrupted`: Ctrl-C cleanup; CLI exits 130.

The Controller terminates spawned compiler children with the worker. Resume an
interrupted normal run with `./bench.sh run --resume RUN_ID`. A legacy import
has no execution state and cannot be resumed.

## GPU lock and noisy measurements

GPU lock timeouts identify `.runtime/locks/<GPU UUID>.lock` and owner metadata.
Never delete a lock while its PID may exist. A well-formed dead-owner lock is
recoverable; malformed or permission-protected metadata is conservative.

Another compute process, timer-role mismatch, too few samples, high CV, failed
correctness, or an opted-in failed-correctness measurement makes performance
non-formal/non-rankable. Raw diagnostic samples remain visible. Re-run formal
measurement only after the selected GPU is idle; never kill unrelated jobs.

The B200 regression checks physical GPU 0's UUID before legacy measurement,
between frameworks, and after engine measurement. A task on another GPU does
not invalidate GPU 0, but any process on the recorded UUID does.

## Artifact errors

CSV headers, versions, enum values, row semantics, primary keys, and finite
numbers are strict. Same-key/different-row writes are conflicts. Atomic replace
failure leaves the previous complete file. A multi-directory legacy import may
publish earlier complete directories before a later rename fails; rerun the
same command to validate/reuse complete directories and publish the rest.

Do not edit imported artifacts: their SHA-256 inventory and source-derived
expected contents are checked on repeated import. A mismatch is a conflict,
not an invitation to overwrite history.

## Bounded logs and security

A log truncation marker means the byte ceiling was reached; pipe draining
continues to avoid deadlock. Failure isolation, path validation, timeouts, and
bounded logs are not a hostile-code sandbox. Use an external restricted
environment for untrusted candidate code.
