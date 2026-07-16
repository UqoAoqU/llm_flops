"""Reproducible in-worker reference/candidate correctness evaluation."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from benchmark_engine.models import CaseSpec

from .comparators import ExactComparator
from .diagnostics import comparison_diagnostic, exception_diagnostic, reproduction_command
from .inputs import GENERATOR_VERSION, assert_input_isolation, case_fingerprint, input_summary, make_generator_context
from .models import CorrectnessResult, InputBundle, OutputBundle
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
        for repeated in candidate_outputs[1:]:
            drift = ExactComparator().compare(first, repeated)
            if not drift.passed:
                diagnostic = comparison_diagnostic(drift, reproduction=reproduction)
                diagnostic["kind"] = "nondeterministic"
                return CorrectnessResult("nondeterministic", case.case_id, case.seed, fingerprint, GENERATOR_VERSION, summary, drift, diagnostic)
        try:
            comparison = spec.comparator(case).compare(reference_output, first, case_overrides=case_overrides or {})
        except Exception as error:
            return self._exception_result(error, case, fingerprint, summary, "compare", reproduction)
        status = "pass" if comparison.passed else "fail"
        diagnostic = None if comparison.passed else comparison_diagnostic(comparison, reproduction=reproduction)
        return CorrectnessResult(status, case.case_id, case.seed, fingerprint, GENERATOR_VERSION, summary, comparison, diagnostic)

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
