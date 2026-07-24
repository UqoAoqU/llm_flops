"""Runtime correctness contracts and evaluators.

Runtime tensor objects remain inside a worker and never enter the Phase 5 JSON
protocol.
"""

from .comparators import CalcDiffComparator, ExactComparator, FloatingComparator, QuantizedComparator, TopKComparator, resolve_tolerance
from .evaluator import CorrectnessEvaluator, synchronize_cuda
from .inputs import GeneratorContext, assert_input_isolation, case_fingerprint, clone_input_bundle, input_summary, make_generator_context
from .models import ComparisonResult, CorrectnessResult, InputBundle, OracleGate, OutputBundle, OutputLeaf, QuantizationParameters, Tolerance
from .normalization import OutputNormalizationError, normalize_output
from .protocols import Comparator, OperatorSpec

__all__ = [
    "Comparator", "ComparisonResult", "CorrectnessEvaluator", "CorrectnessResult",
    "CalcDiffComparator", "ExactComparator", "FloatingComparator", "GeneratorContext", "InputBundle",
    "OperatorSpec", "OutputBundle", "OutputLeaf", "OutputNormalizationError",
    "OracleGate", "QuantizationParameters", "QuantizedComparator", "Tolerance", "TopKComparator",
    "assert_input_isolation", "case_fingerprint", "clone_input_bundle", "input_summary", "make_generator_context",
    "normalize_output", "resolve_tolerance", "synchronize_cuda",
]
