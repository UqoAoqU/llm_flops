# Benchmark Engine documentation

Phase 9: [CLI and exit codes](cli.md), [correctness](correctness.md),
[performance measurement](performance.md),
[result/resume layout](result-layout.md), [troubleshooting](troubleshooting.md),
[architecture](architecture.md), and [development](development.md).

The benchmark engine is being introduced alongside the existing DeepSeek V4
and GLM-5 launchers. Through Phase 6 it provides strict discovery,
deterministic plans, recoverable mirrored artifacts, isolated workers, and
worker-local correctness semantics. Correctness and staged performance
measurement are both connected to the CLI. Phase 9 adds fair interleaving,
UUID GPU locks, formal gates, and strict comparisons.

## Guides

- [Architecture](architecture.md) describes the package boundaries and the
  staged evaluation lifecycle.
- [Registry CLI](cli.md) documents `bench list` and `bench validate`.
- [Operator contract](operator-contract.md) defines `operator.yaml` schema v1.
- [Correctness](correctness.md) defines runtime inputs, normalized outputs,
  comparators, tolerances, diagnostics, and determinism checks.
- [Performance](performance.md) defines timers, stage separation, sampling,
  statistics, theoretical cost, gates, locking, and the exact Phase 9 boundary.
- [Candidate guide](candidate-guide.md) defines candidate layout and hashing.
- [Result layout](result-layout.md) defines source/result mirroring.
- [CSV schemas](csv-schema.md) define result, output, sample, and index
  tables.
- [ADR 0001](adr/0001-result-directory-key.md) records the result directory key.
- [Development](development.md) covers environment setup and verification.
- [Troubleshooting](troubleshooting.md) explains worker outcomes, timeout
  cleanup, diagnostics, and bounded logs.
- [Approved design](../design.md) is the normative long-form design.
- [Migration baseline](migration-baseline.md) records the preserved legacy
  behavior.
- [Contributing](../CONTRIBUTING.md) lists change and test expectations.

Legacy benchmark commands remain documented in the [repository README](../README.md).
