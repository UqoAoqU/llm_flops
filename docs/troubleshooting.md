# Troubleshooting managed workers

GPU lock timeouts identify `.runtime/locks/<GPU UUID>.lock` and its owner.
Never delete a lock whose PID may still exist; malformed or permission-protected
metadata times out conservatively. Non-formal performance commonly means too
few samples/high CV, role timer mismatch, opted-in correctness failure, or
another compute process detected before sampling. Such raw samples are retained
but never ranked.

- Import failures: inspect `logs/stderr.log` and `diagnostics/*-import.txt`.
- Candidate exceptions: correctness `error` plus a JSON diagnostic.
- Hard timeout: the whole worker process group is terminated; later cases run by default.
- Single-case diagnosis: copy the reproduction command from `bench summarize`.
- Interrupted run: `./bench.sh run --resume RUN_ID`.
- A correctness-only run records `performance_status=skipped` with
  `skip_reason=correctness_only_mode`; this is not a performance pass.

Historical note: Phase 5 introduced import/build isolation. The correctness
workflow was extended through Phase 9 with gated fair performance
measurement and explicit warmup/sampling stages.

## A worker times out during import or build

Inspect `logs/worker.jsonl` to identify the active stage and its heartbeat,
then inspect `logs/stdout.log` and `logs/stderr.log`. A heartbeat indicates
that the worker or compiler was observable; it does not extend the hard stage
timeout. The controller terminates the complete process group, including
Ninja/NVCC/PTXAS children, with TERM followed by KILL.

Increase a timeout only when the declared workload legitimately requires it.
Do not disable the timeout or treat compiler output as a successful stage.

## Outcome categories

- `error`: a structured Python/import/build exception.
- `timeout`: a stage exceeded its independent hard deadline.
- `oom`: `MemoryError` or a recognized CPU/CUDA out-of-memory diagnostic.
- `crashed`: signal/nonzero exit without a valid structured response.
- `unsupported`: an explicit `NotImplementedError` or unsupported backend.
- `interrupted`: controller cleanup following Ctrl-C (CLI exit code 130).

The response error is intentionally short. Follow `diagnostic_path` for the
full traceback or controller crash summary. A missing/invalid response is not
silently converted to `unavailable`.

When a result is retried, inspect its numbered attempt files and the matching
`DISCOVERED`-bounded event transcript. A response from an older attempt is
never reused for the current process, including when the current worker dies
before writing a structured response.

## Truncated output

Formal stdout/stderr files have a byte ceiling. A
`[benchmark-engine: log truncated ...]` marker means the process produced more
output than retained. Pipe draining continues after the marker, so truncation
does not deadlock the worker. The controller summary tail is separately
bounded.

## Security boundary

`start_new_session`, path validation, timeouts, and bounded logs provide
failure isolation only. They are not a hostile-code sandbox. Run untrusted
candidate code in a separately configured container or restricted account.
