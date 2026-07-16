"""Pure metadata spec; importing it does not import the implementation."""

from benchmark_engine.models import CaseSpec


class MinimalAddSpec:
    operator_id = "minimal_cpu_add"

    def cases(self) -> tuple[CaseSpec, ...]:
        return (
            CaseSpec(
                case_id="small",
                symbols={"left": 1, "right": 2},
                seed=0,
                tags=frozenset({"smoke", "representative", "cpu"}),
            ),
            CaseSpec(
                case_id="negative",
                symbols={"left": -7, "right": 3},
                seed=0,
                tags=frozenset({"boundary", "full", "cpu"}),
            ),
        )


SPEC = MinimalAddSpec()


def cost_model(*_args: object, **_kwargs: object) -> None:
    return None
