# Development

## Environment

Run the repository bootstrap from the repository root:

~~~bash
./bootstrap.sh
~~~

The script creates `.runtime/venv`, installs the locked dependencies, installs
this repository as an editable package without resolving extra dependencies,
validates the legacy GPU environment, verifies that `benchmark_engine` imports,
and only then records the lock marker.

Use either entry point:

~~~bash
.runtime/venv/bin/bench --help
./bench.sh --help
~~~

The shell launcher clears external `PYTHONPATH` state and keeps uv, extension,
FlashInfer, and XDG caches under `.runtime/cache`.

## Verification

Phase 1 changes should pass:

~~~bash
.runtime/venv/bin/python -m unittest discover -s tests -v
./run.sh check
git diff --check
~~~

Do not add generated runtime, cache, log, or benchmark result artifacts to Git.
Do not change legacy command behavior as part of an engine-only change.
