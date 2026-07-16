# CSV schema v1

Applies to CSV schema version **1**. The Python authority is
`benchmark_engine.reporting.csv_writer`; this document defines the persisted
contract for readers and future evaluators.

All tables are UTF-8 comma-separated RFC 4180 files with a header and CRLF
record terminators. `schema_version` is always the first column and the value
is the integer `1`. Writers preserve the column order below. An empty field is
the only null representation: zero, `false`, and an empty JSON array are real
values and must not stand in for unavailable data. Booleans are lower-case
`true`/`false`; numbers are finite base-10 values; timestamps are UTC strings.
JSON-valued strings use compact JSON when populated.

String enums are closed sets and are validated on both write and read:

- `results.mode`: `all`, `correctness`, `performance`;
- `results.status`: `planned`, `running`, `passed`, `failed`, `skipped`,
  `unsupported`, `error`, `timeout`, `oom`, `crashed`;
- `results.correctness_status`: `planned`, `passed`, `failed`, `skipped`,
  `unsupported`, `error`, `timeout`, `oom`, `crashed`;
- `results.performance_status`: the correctness values above plus `unstable`;
- `performance_samples.implementation_role`: `reference`, `candidate`;
- `history.status`: `complete` only.

Unknown enum strings are schema errors rather than forward-compatible values;
adding an outcome therefore requires an explicit compatible schema update.

The controller is the sole writer. Each completed case/sample is persisted by
rewriting a same-directory temporary file, flushing and fsyncing it, then using
`os.replace()`. Repeating an identical primary key and row is idempotent;
reusing the key with different data is a conflict and never overwrites the old
row. Breaking changes require a new schema version; v1 readers reject unknown
headers and other schema versions.

## `results.csv`

One row is the summary of one `operator × candidate × case × seed`. Its
primary key is `result_id`.

| Fields (stable order) | Type | Null rule |
|---|---|---|
| `schema_version` | integer | required, always `1` |
| `run_id`, `evaluation_id`, `timestamp_utc`, `suite_id`, `mode` | string | required |
| `result_id`, `operator_id`, `candidate_id`, `reference_id` | string | required |
| `contract_version` | integer | required |
| `candidate_source_hash`, `reference_source_hash` | string | required |
| `environment_fingerprint` | string | required |
| `device`, `gpu_name`, `cuda_version`, `torch_version` | string | empty when unavailable |
| `case_id`, `case_hash` | string | required |
| `seed` | integer | required |
| `tags`, `input_summary` | JSON string | empty when unavailable |
| `status`, `correctness_status`, `performance_status` | string enum | required |
| `skip_reason` | string | empty unless skipped |
| `correctness_pass` | boolean | empty until evaluated |
| `failed_output_count`, `mismatch_count` | integer | empty until evaluated |
| `max_abs_error`, `max_rel_error`, `rmse`, `rel_l2`, `cosine_similarity`, `mismatch_rate` | number | empty when not applicable |
| `timer` | string | empty when performance was not run |
| `import_ms`, `build_ms`, `first_call_ms`, `warmup_ms`, `graph_capture_ms` | number | empty when not measured |
| `reference_median_ms`, `candidate_median_ms`, `candidate_p95_ms`, `candidate_stddev_ms`, `candidate_cv`, `speedup`, `slowdown_pct` | number | empty when not measured |
| `tflops`, `effective_bandwidth_gbps`, `throughput` | number | empty when no cost model applies |
| `peak_memory_bytes`, `workspace_bytes` | integer | empty when unavailable |
| `error_type`, `error_message`, `diagnostic_path`, `stdout_path`, `stderr_path` | string | empty on success |
| `profile_path` | string | reserved and empty while profiler support is excluded |

The exact flat order is the order shown across table rows, left to right within
each field list. Full tracebacks and mismatch samples belong in diagnostics,
not `error_message`.

## `correctness_outputs.csv`

One row describes one normalized output leaf. Its primary key is
`(result_id, output_path)`.

| Fields (stable order) | Type | Null rule |
|---|---|---|
| `schema_version` | integer | required, always `1` |
| `result_id`, `output_path`, `comparator` | string | required |
| `reference_dtype`, `candidate_dtype` | string | required |
| `reference_shape`, `candidate_shape` | JSON string | required |
| `rtol`, `atol` | number | empty when comparator has no tolerance |
| `passed` | boolean | required |
| `max_abs_error`, `mean_abs_error`, `p95_abs_error`, `max_rel_error`, `rmse`, `rel_l2`, `cosine_similarity`, `mismatch_rate` | number | empty when not computable |
| `mismatch_count`, `reference_nan_count`, `candidate_nan_count` | integer | empty when not computable |
| `diagnostic_path` | string | empty when no detailed diagnostic exists |

## `performance_samples.csv`

One row is one raw timing sample. Its primary key is
`(result_id, implementation_role, sample_index)`. Reference and candidate
samples remain separate so statistics can be recalculated later.

| Fields (stable order) | Type | Null rule |
|---|---|---|
| `schema_version` | integer | required, always `1` |
| `result_id` | string | required |
| `implementation_role` | string enum (`reference` or `candidate`) | required |
| `sample_index`, `inner_iterations`, `order_index` | integer | required |
| `elapsed_ms`, `per_call_ms` | number | required |
| `gpu_clock_mhz`, `memory_clock_mhz`, `temperature_c`, `power_w` | number | empty when telemetry is unavailable; never fake zero |

## Index CSVs

`results/run_index.csv` has the stable order `schema_version, run_id,
operator_id, candidate_id, evaluation_id, relative_path`. Its primary key is
the first four identity fields after the schema version and it contains no case
data.

Each candidate `history.csv` has the stable order `schema_version,
evaluation_id, run_id, operator_id, candidate_id, status, completed_at_utc,
relative_path, suite_id, mode, reference_source_hash,
candidate_source_hash, environment_fingerprint`. Its primary key is
`evaluation_id`, and it contains only complete evaluations.
