"""Deliberately wrong fixture; never discovered as a formal candidate."""


def operator(
    activation,
    activation_scale,
    activation_scale_aligned,
    weight,
    weight_scale,
    output,
):
    del activation, activation_scale, activation_scale_aligned, weight, weight_scale
    output.zero_()
    return output
