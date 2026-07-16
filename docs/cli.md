# Registry CLI

Phase 2 exposes only import-free discovery commands in addition to global help
and version output:

```bash
bench list
bench list --operator 'minimal_*' --candidate 'task_*'
bench validate
bench validate --operator 'minimal_*'
```

Selectors use case-sensitive shell-style globs. Output is sorted by operator
and candidate ID. An invalid registry or a selector with no matches exits with
status 2. Discovery checks Python file locations but does not import reference,
spec, or candidate modules. Execution, environment, suite, and profiling
commands are not part of this phase.
