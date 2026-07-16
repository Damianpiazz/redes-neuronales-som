"""
Modulo de funciones de vecindad.

Define como se propaga la influencia del BMU a las neuronas vecinas.
La funcion gaussiana es la mas comun: mayor influencia en vecinos
cercanos, menor en lejanos, con decaimiento controlado por sigma.
"""

import math


def gaussian(
    bmu_row,
    bmu_col,
    neuron_row,
    neuron_col,
    sigma,
):
    """
    Funcion de vecindad gaussiana.

    Calcula el valor de la funcion de vecindad entre el BMU
    y una neurona de la grilla. El valor esta en (0, 1]:
    - 1.0 si la neurona es el BMU mismo
    - Menor a medida que la distancia aumenta
    - Controlado por sigma (radio de vecindad)

    Formula: h = exp(-d² / (2σ²))
    donde d es la distancia euclidiana en la grilla.
    """
    d_sq = (
        (bmu_row - neuron_row) ** 2
        + (bmu_col - neuron_col) ** 2
    )

    return math.exp(
        -d_sq / (2 * sigma ** 2)
    )
