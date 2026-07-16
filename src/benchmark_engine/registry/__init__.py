"""Static operator/candidate discovery and manifest validation."""

from .base import (
    BuildManifest,
    CandidateManifest,
    CorrectnessManifest,
    OperatorManifest,
    OperatorRegistry,
    OperatorSpecMetadata,
    PerformanceManifest,
    RegistryIssue,
    RegistrySnapshot,
)
from .filesystem import FilesystemRegistry
from .issues import registry_issue_owner, selected_registry_issues
from .validation import (
    ManifestValidationError,
    SourceHashError,
    compute_source_hash,
    parse_candidate_manifest,
    parse_operator_manifest,
)

__all__ = [
    "BuildManifest",
    "CandidateManifest",
    "CorrectnessManifest",
    "FilesystemRegistry",
    "ManifestValidationError",
    "OperatorManifest",
    "OperatorRegistry",
    "OperatorSpecMetadata",
    "PerformanceManifest",
    "RegistryIssue",
    "RegistrySnapshot",
    "SourceHashError",
    "compute_source_hash",
    "parse_candidate_manifest",
    "parse_operator_manifest",
    "registry_issue_owner",
    "selected_registry_issues",
]
