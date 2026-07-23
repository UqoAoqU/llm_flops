"""Public package surface for the benchmark engine."""

from importlib.metadata import PackageNotFoundError, version

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

try:
    __version__ = version("benchmark-engine")
except PackageNotFoundError:
    # The MI300X runtime executes the repository directly through a controlled
    # PYTHONPATH so the shared ROCm virtualenv remains untouched.
    __version__ = "0.1.0"

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
