# Benchmark Engine documentation

The benchmark engine is being introduced alongside the existing DeepSeek V4
and GLM-5 launchers. Phase 2 adds strict, import-free filesystem discovery,
source identities, and mirrored result paths. It does not execute operators.

## Guides

- [Architecture](architecture.md) describes the package boundaries and the
  staged evaluation lifecycle.
- [Registry CLI](cli.md) documents `bench list` and `bench validate`.
- [Operator contract](operator-contract.md) defines `operator.yaml` schema v1.
- [Candidate guide](candidate-guide.md) defines candidate layout and hashing.
- [Result layout](result-layout.md) defines source/result mirroring.
- [ADR 0001](adr/0001-result-directory-key.md) records the result directory key.
- [Development](development.md) covers environment setup and verification.
- [Approved design](../design.md) is the normative long-form design.
- [Migration baseline](migration-baseline.md) records the preserved legacy
  behavior.
- [Contributing](../CONTRIBUTING.md) lists change and test expectations.

Legacy benchmark commands remain documented in the [repository README](../README.md).
