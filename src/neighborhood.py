import math


def gaussian(
    bmu_row,
    bmu_col,
    neuron_row,
    neuron_col,
    sigma,
):

    d_sq = (
        (bmu_row - neuron_row) ** 2
        + (bmu_col - neuron_col) ** 2
    )

    return math.exp(
        -d_sq / (2 * sigma ** 2)
    )
