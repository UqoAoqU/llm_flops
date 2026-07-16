"""Validate and fingerprint the runtime used for benchmark baselines."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import platform
import copy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_LOCK = ROOT / "requirements" / "benchmark-lock.json"


def load_lock(path: Path = DEFAULT_LOCK) -> dict[str, Any]:
    lock = json.loads(path.read_text())
    if lock.get("schema_version") != 1:
        raise ValueError(f"unsupported schema_version: {lock.get('schema_version')}")
    return lock


def _has_symbol(name: str) -> bool:
    parts = name.split(".")
    module = None
    remainder: list[str] = []
    for split_at in range(len(parts), 0, -1):
        try:
            module = importlib.import_module(".".join(parts[:split_at]))
            remainder = parts[split_at:]
            break
        except ModuleNotFoundError as error:
            candidate = ".".join(parts[:split_at])
            if error.name != candidate and not candidate.startswith(f"{error.name}."):
                return False
    if module is None:
        return False
    value: Any = module
    for attribute in remainder:
        if not hasattr(value, attribute):
            return False
        value = getattr(value, attribute)
    return True


def collect_environment(
    lock: dict[str, Any], include_cuda: bool = True
) -> dict[str, Any]:
    packages: dict[str, dict[str, str | None]] = {}
    import_paths: dict[str, str | None] = {}
    for distribution, package in lock["packages"].items():
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = None
        module_name = package["module"]
        try:
            module = importlib.import_module(module_name)
            import_path = getattr(module, "__file__", None)
        except Exception:
            import_path = None
        packages[distribution] = {"version": version, "module": module_name}
        import_paths[module_name] = import_path

    sources = {}
    for name, source in lock.get("source", {}).items():
        package_name = source["package"]
        actual_commit = None
        try:
            direct_url = importlib.metadata.distribution(package_name).read_text(
                "direct_url.json"
            )
            if direct_url:
                actual_commit = json.loads(direct_url).get("vcs_info", {}).get(
                    "commit_id"
                )
        except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError):
            pass
        if actual_commit is None:
            try:
                generated = importlib.import_module(f"{package_name}._version")
                actual_commit = getattr(generated, "commit_id", None)
            except Exception:
                pass
        actual_commit = (actual_commit or "").removeprefix("g") or None
        expected_commit = source["commit"]
        if actual_commit and expected_commit.startswith(actual_commit):
            actual_commit = expected_commit
        sources[name] = {"commit": actual_commit}

    observed: dict[str, Any] = {
        "python": platform.python_version(),
        "cuda": None,
        "gpu": {"available": False, "capability": None, "name": None},
        "packages": packages,
        "source": sources,
        "symbols": {name: _has_symbol(name) for name in lock["required_symbols"]},
        "import_paths": import_paths,
    }
    if include_cuda:
        try:
            import torch

            observed["cuda"] = torch.version.cuda
            available = torch.cuda.is_available()
            observed["gpu"]["available"] = available
            if available:
                observed["gpu"]["capability"] = list(torch.cuda.get_device_capability())
                observed["gpu"]["name"] = torch.cuda.get_device_name()
        except Exception:
            pass
    return observed


def validate_environment(lock: dict[str, Any], observed: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not observed.get("python", "").startswith(f"{lock['python']}."):
        errors.append(
            f"Python mismatch: expected {lock['python']}.x, observed {observed.get('python')}"
        )
    if observed.get("cuda") != lock["cuda"]:
        errors.append(
            f"CUDA mismatch: expected {lock['cuda']}, observed {observed.get('cuda')}"
        )
    expected_gpu = lock["gpu"]
    observed_gpu = observed.get("gpu", {})
    if observed_gpu.get("capability") != expected_gpu["capability"]:
        errors.append(
            "GPU capability mismatch: expected "
            f"{expected_gpu['capability']}, observed {observed_gpu.get('capability')}"
        )
    if expected_gpu["name_contains"] not in (observed_gpu.get("name") or ""):
        errors.append(
            f"GPU name mismatch: expected *{expected_gpu['name_contains']}*, "
            f"observed {observed_gpu.get('name')}"
        )
    for distribution, expected in lock["packages"].items():
        actual = observed.get("packages", {}).get(distribution, {}).get("version")
        if expected["version"] is not None and actual != expected["version"]:
            errors.append(
                f"package {distribution} mismatch: expected {expected['version']}, observed {actual}"
            )
    for name, expected in lock.get("source", {}).items():
        actual = observed.get("source", {}).get(name, {}).get("commit")
        if actual != expected["commit"]:
            errors.append(
                f"source {name} mismatch: expected {expected['commit']}, observed {actual}"
            )
    for symbol in lock["required_symbols"]:
        if not observed.get("symbols", {}).get(symbol, False):
            errors.append(f"required symbol unavailable: {symbol}")
    return errors


def environment_fingerprint(observed: dict[str, Any]) -> str:
    stable = copy.deepcopy(
        {key: value for key, value in observed.items() if key != "import_paths"}
    )
    for source_name in stable.get("source", {}):
        if source_name in stable.get("packages", {}):
            stable["packages"][source_name].pop("version", None)
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    lock = load_lock(args.lock)
    observed = collect_environment(lock)
    errors = validate_environment(lock, observed)
    report = {
        "fingerprint": environment_fingerprint(observed),
        "valid": not errors,
        "errors": errors,
        "environment": observed,
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Benchmark environment: {report['fingerprint']}")
        print(f"Python {observed['python']}  CUDA {observed['cuda']}")
        print(
            f"GPU {observed['gpu']['name']}  capability={observed['gpu']['capability']}"
        )
        for error in errors:
            print(f"ERROR: {error}")
    return 1 if args.check and errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
