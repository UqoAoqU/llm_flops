"""Public package surface for the benchmark engine."""

from importlib.metadata import version

from .models import (
    CaseSpec,
    CorrectnessStatus,
    EvaluationIdentity,
    EvaluationJob,
    EvaluationPlan,
    ImplementationSpec,
    InputBundle,
    OutputBundle,
    OutputLeaf,
    PerformanceStatus,
    ResultStatus,
)

__version__ = version("benchmark-engine")

__all__ = [
    "__version__",
    "CaseSpec",
    "CorrectnessStatus",
    "EvaluationIdentity",
    "EvaluationJob",
    "EvaluationPlan",
    "ImplementationSpec",
    "InputBundle",
    "OutputBundle",
    "OutputLeaf",
    "PerformanceStatus",
    "ResultStatus",
]
