"""Reproducible in-worker reference/candidate correctness evaluation."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from benchmark_engine.models import CaseSpec

from .comparators import ExactComparator
from .diagnostics import comparison_diagnostic, exception_diagnostic, reproduction_command
from .inputs import GENERATOR_VERSION, assert_input_isolation, case_fingerprint, input_summary, make_generator_context
from .models import ComparisonResult, CorrectnessResult, InputBundle, OracleGate, OutputBundle
from .normalization import attach_observed_state, normalize_output
from .protocols import OperatorSpec


def _walk(value: object, seen: set[int]):
    identity = id(value)
    if identity in seen:
        return
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return
    seen.add(identity)
    yield value
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _walk(item, seen)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item, seen)


def synchronize_cuda(output: object, inputs: InputBundle) -> None:
    """Synchronize CUDA devices touched by inputs/output; a no-op without torch."""

    try:
        import torch  # type: ignore
    except ImportError:
        return
    devices: set[str] = set()
    for value in _walk((output, inputs.args, inputs.kwargs, inputs.observed_state), set()):
        device = getattr(value, "device", None)
        if device is not None and str(device).startswith("cuda"):
            devices.add(str(device))
    for device in sorted(devices):
        torch.cuda.synchronize(torch.device(device))


class CorrectnessEvaluator:
    def __init__(self, *, determinism_repeats: int = 1, synchronizer: Callable[[object, InputBundle], None] = synchronize_cuda) -> None:
        if determinism_repeats < 1:
            raise ValueError("determinism_repeats must be at least one")
        self.determinism_repeats = determinism_repeats
        self.synchronizer = synchronizer

    @staticmethod
    def _normalize(spec: OperatorSpec, output: object, inputs: InputBundle) -> OutputBundle:
        normalized = spec.normalize_output(output)
        if isinstance(normalized, OutputBundle):
            return attach_observed_state(normalized, inputs.observed_state)
        return normalize_output(normalized, inputs.observed_state)

    def evaluate(
        self,
        *,
        spec: OperatorSpec,
        reference: Callable[..., object],
        candidate: Callable[..., object],
        case: CaseSpec,
        operator_id: str = "operator",
        candidate_id: str = "candidate",
        cuda_devices: tuple[str, ...] = (),
        case_overrides: Mapping[str, object] | None = None,
    ) -> CorrectnessResult:
        context = make_generator_context(case.seed, cuda_devices)
        reproduction = reproduction_command(operator_id, candidate_id, case.case_id, case.seed)
        summary: Mapping[str, object] = {}
        fingerprint = ""
        try:
            canonical = spec.make_inputs(case, context)
            if not isinstance(canonical, InputBundle):
                raise TypeError("OperatorSpec.make_inputs() must return correctness.InputBundle")
            summary = input_summary(canonical)
            fingerprint = case_fingerprint(case, summary)
            reference_inputs = spec.clone_inputs(canonical)
            candidate_inputs = spec.clone_inputs(canonical)
            if not isinstance(reference_inputs, InputBundle) or not isinstance(candidate_inputs, InputBundle):
                raise TypeError("OperatorSpec.clone_inputs() must return correctness.InputBundle")
            assert_input_isolation(canonical, reference_inputs)
            assert_input_isolation(canonical, candidate_inputs)
            assert_input_isolation(reference_inputs, candidate_inputs)
        except Exception as error:
            return self._exception_result(error, case, fingerprint, summary, "input", reproduction)

        try:
            reference_raw = reference(*reference_inputs.args, **reference_inputs.kwargs)
            self.synchronizer(reference_raw, reference_inputs)
            reference_output = self._normalize(spec, reference_raw, reference_inputs)
        except Exception as error:
            return self._exception_result(error, case, fingerprint, summary, "reference", reproduction)

        candidate_outputs: list[OutputBundle] = []
        candidate_input_clones: list[InputBundle] = [candidate_inputs]
        for repeat in range(self.determinism_repeats):
            try:
                current_inputs = candidate_inputs if repeat == 0 else spec.clone_inputs(canonical)
                if not isinstance(current_inputs, InputBundle):
                    raise TypeError("OperatorSpec.clone_inputs() must return correctness.InputBundle")
                if repeat:
                    assert_input_isolation(canonical, current_inputs)
                    assert_input_isolation(reference_inputs, current_inputs)
                    for prior_inputs in candidate_input_clones:
                        assert_input_isolation(prior_inputs, current_inputs)
                    candidate_input_clones.append(current_inputs)
                raw = candidate(*current_inputs.args, **current_inputs.kwargs)
                self.synchronizer(raw, current_inputs)
                candidate_outputs.append(self._normalize(spec, raw, current_inputs))
            except Exception as error:
                return self._exception_result(error, case, fingerprint, summary, "candidate", reproduction)

        first = candidate_outputs[0]
        reference_leaves = reference_output.by_path()
        candidate_leaves = first.by_path()
        output_contracts = {}
        for path in sorted(set(reference_leaves) | set(candidate_leaves)):
            reference_leaf = reference_leaves.get(path)
            candidate_leaf = candidate_leaves.get(path)
            output_contracts[path] = {
                "reference_dtype": None if reference_leaf is None else reference_leaf.dtype,
                "candidate_dtype": None if candidate_leaf is None else candidate_leaf.dtype,
                "reference_shape": None if reference_leaf is None else list(reference_leaf.shape),
                "candidate_shape": None if candidate_leaf is None else list(candidate_leaf.shape),
            }
        for repeated in candidate_outputs[1:]:
            drift = ExactComparator().compare(first, repeated)
            if not drift.passed:
                diagnostic = comparison_diagnostic(drift, reproduction=reproduction)
                diagnostic["kind"] = "nondeterministic"
                return CorrectnessResult("nondeterministic", case.case_id, case.seed, fingerprint, GENERATOR_VERSION, summary, drift, diagnostic, output_contracts)
        gates_factory = getattr(spec, "correctness_gates", None)
        if callable(gates_factory):
            try:
                comparison = self._evaluate_oracle_gates(
                    spec=spec,
                    case=case,
                    canonical=canonical,
                    reference_inputs=reference_inputs,
                    candidate_inputs=candidate_input_clones,
                    reference_output=reference_output,
                    candidate_output=first,
                    case_overrides=case_overrides or {},
                )
            except _OracleStageError as error:
                return self._exception_result(error.__cause__ or error, case, fingerprint, summary, error.stage, reproduction)
            except Exception as error:
                return self._exception_result(error, case, fingerprint, summary, "oracle_contract", reproduction)
        else:
            try:
                comparison = spec.comparator(case).compare(reference_output, first, case_overrides=case_overrides or {})
            except Exception as error:
                return self._exception_result(error, case, fingerprint, summary, "compare", reproduction)
        status = "pass" if comparison.passed else "fail"
        diagnostic = None if comparison.passed else comparison_diagnostic(comparison, reproduction=reproduction)
        return CorrectnessResult(status, case.case_id, case.seed, fingerprint, GENERATOR_VERSION, summary, comparison, diagnostic, output_contracts)

    def _evaluate_oracle_gates(
        self,
        *,
        spec: OperatorSpec,
        case: CaseSpec,
        canonical: InputBundle,
        reference_inputs: InputBundle,
        candidate_inputs: list[InputBundle],
        reference_output: OutputBundle,
        candidate_output: OutputBundle,
        case_overrides: Mapping[str, object],
    ) -> ComparisonResult:
        declared = tuple(spec.correctness_gates(case))
        if not declared:
            raise ValueError("correctness_gates() must return at least one gate")
        if not all(isinstance(gate, OracleGate) for gate in declared):
            raise TypeError("correctness_gates() must return OracleGate values")
        if len({gate.gate_id for gate in declared}) != len(declared):
            raise ValueError("correctness gate ids must be unique")
        oracles_factory = getattr(spec, "correctness_oracles", None)
        if not callable(oracles_factory):
            raise TypeError("multi-oracle spec must provide correctness_oracles()")
        oracles = oracles_factory(case)
        if not isinstance(oracles, Mapping):
            raise TypeError("correctness_oracles() must return a mapping")

        metrics: dict[str, object] = {"gates": {}}
        diagnostics: list[dict[str, object]] = []
        failed_path: str | None = None
        all_passed = True
        prior_inputs = [reference_inputs, *candidate_inputs]
        for gate in declared:
            oracle = oracles.get(gate.oracle_id)
            if not callable(oracle):
                raise ValueError(f"missing callable oracle {gate.oracle_id!r} for gate {gate.gate_id!r}")
            try:
                oracle_inputs = spec.clone_inputs(canonical)
                if not isinstance(oracle_inputs, InputBundle):
                    raise TypeError("OperatorSpec.clone_inputs() must return correctness.InputBundle")
                assert_input_isolation(canonical, oracle_inputs)
                for previous in prior_inputs:
                    assert_input_isolation(previous, oracle_inputs)
                prior_inputs.append(oracle_inputs)
                oracle_raw = oracle(oracle_inputs)
                self.synchronizer(oracle_raw, oracle_inputs)
                oracle_output = self._normalize(spec, oracle_raw, oracle_inputs)
            except Exception as error:
                raise _OracleStageError(f"oracle:{gate.gate_id}") from error
            try:
                reference_comparison = gate.comparator.compare(
                    oracle_output, reference_output, case_overrides=case_overrides
                )
                candidate_comparison = gate.comparator.compare(
                    oracle_output, candidate_output, case_overrides=case_overrides
                )
            except Exception as error:
                raise _OracleStageError(f"compare:{gate.gate_id}") from error
            gate_passed = reference_comparison.passed and candidate_comparison.passed
            metrics["gates"][gate.gate_id] = {
                "oracle_id": gate.oracle_id,
                "required": gate.required,
                "passed": gate_passed,
                "reference": _comparison_summary(reference_comparison),
                "candidate": _comparison_summary(candidate_comparison),
            }
            if not gate_passed:
                if gate.required:
                    all_passed = False
                failed_path = failed_path or reference_comparison.failed_path or candidate_comparison.failed_path
                for subject, result in (("reference", reference_comparison), ("candidate", candidate_comparison)):
                    if not result.passed:
                        diagnostics.extend(
                            {"gate_id": gate.gate_id, "oracle_id": gate.oracle_id, "subject": subject, **item}
                            for item in result.diagnostics
                        )
        return ComparisonResult(all_passed, "oracle_gates", metrics, tuple(diagnostics[:8]), failed_path)

    @staticmethod
    def _exception_result(error: Exception, case: CaseSpec, fingerprint: str, summary: Mapping[str, object], stage: str, reproduction: str) -> CorrectnessResult:
        if isinstance(error, TimeoutError):
            status = "timeout"
        elif isinstance(error, NotImplementedError):
            status = "unsupported"
        elif isinstance(error, MemoryError) or "out of memory" in str(error).lower():
            status = "oom"
        else:
            status = "error"
        return CorrectnessResult(status, case.case_id, case.seed, fingerprint, GENERATOR_VERSION, summary, None, exception_diagnostic(error, stage=stage, reproduction=reproduction))


class _OracleStageError(Exception):
    def __init__(self, stage: str) -> None:
        super().__init__(stage)
        self.stage = stage


def _comparison_summary(result: ComparisonResult) -> dict[str, object]:
    return {
        "passed": result.passed,
        "comparator": result.comparator,
        "metrics": result.metrics,
        "failed_path": result.failed_path,
    }
