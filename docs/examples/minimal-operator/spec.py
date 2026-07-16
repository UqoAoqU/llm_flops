"""Static example spec location; worker loading is introduced later."""


class MinimalAddSpec:
    operator_id = "minimal_cpu_add"


SPEC = MinimalAddSpec()


def cost_model(*_args: object, **_kwargs: object) -> None:
    return None
