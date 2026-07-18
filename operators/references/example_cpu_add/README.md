# Example CPU add

This operator is the dependency-light framework example. The reference adds
two integer inputs; its candidates demonstrate a correct implementation,
numeric mismatch, exception, and timeout. It is intended for Registry,
Controller/Worker, correctness, reporting, resume, and wall-clock smoke tests,
not for hardware performance claims.

```bash
./bench.sh validate --operator example_cpu_add
./bench.sh run --mode correctness --operator example_cpu_add \
  --candidate quickstart__20260716T120000Z__4279e756 --case tiny
./bench.sh run --mode performance --operator example_cpu_add \
  --candidate quickstart__20260716T120000Z__4279e756 --case tiny \
  --timer wall_clock --warmup 5 --samples 30 --inner-iterations 20
```
