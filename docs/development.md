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

Every phase should pass:

~~~bash
.runtime/venv/bin/python -m unittest discover -s tests -v
./run.sh check
git diff --check
~~~

Do not add generated runtime, cache, log, or benchmark result artifacts to Git.
Do not change legacy command behavior as part of an engine-only change.

Phase 5 execution tests are Linux-sensitive because they validate sessions,
signals, process groups, segfault classification, and recursive child cleanup:

~~~bash
.runtime/venv/bin/python -m unittest -v \
  tests/test_worker_protocol.py \
  tests/test_controller_isolation.py \
  tests/test_worker_failures.py \
  tests/test_event_log.py
~~~

Keep fixture timeouts short and always allow the controller to perform its
TERM/KILL/reap cleanup. Fixtures belong under `tests/fixtures/workers/`; never
place crash/hang fixtures in the production operator registry.

Phase 6 correctness algorithms have a CPU-only mandatory suite:

~~~bash
.runtime/venv/bin/python -m unittest -v \
  tests/test_input_bundle.py \
  tests/test_output_normalization.py \
  tests/test_exact_comparator.py \
  tests/test_floating_comparator.py \
  tests/test_topk_comparator.py \
  tests/test_quantized_comparator.py \
  tests/test_correctness_evaluator.py
~~~

The evaluator test contains a real CUDA allocation/synchronization smoke when
CUDA is available and otherwise skips only that method. Correctness unit tests
must not require a GPU.
