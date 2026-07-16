# Benchmark Engine documentation

The benchmark engine is being introduced alongside the existing DeepSeek V4
and GLM-5 launchers. Phase 1 establishes the installable package, immutable
cross-process metadata models, and the `bench` launcher. It does not yet
discover or execute operators.

## Guides

- [Architecture](architecture.md) describes the package boundaries and the
  staged evaluation lifecycle.
- [Development](development.md) covers environment setup and verification.
- [Approved design](../design.md) is the normative long-form design.
- [Migration baseline](migration-baseline.md) records the preserved legacy
  behavior.
- [Contributing](../CONTRIBUTING.md) lists change and test expectations.

Legacy benchmark commands remain documented in the [repository README](../README.md).
