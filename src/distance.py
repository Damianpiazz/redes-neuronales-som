"""
Modulo de funciones de distancia.

Proporciona metricas de similaridad entre vectores:
- Euclidiana: distancia en linea recta entre dos puntos
- Manhattan: suma de diferencias absolutas (taxi)
"""

import math


def euclidean(a, b):
    """
    Distancia euclidiana entre dos vectores.

    Calcula la raiz cuadrada de la suma de las diferencias
    cuadradas entre cada par de componentes.
    Se usa para encontrar el BMU (Best Matching Unit).
    """
    return math.sqrt(
        sum(
            (ai - bi) ** 2
            for ai, bi in zip(a, b)
        )
    )


def manhattan(a, b):
    """
    Distancia Manhattan (taxi) entre dos vectores.

    Suma de las diferencias absolutas entre cada par
    de componentes. Mas robusta que la euclidiana
    ante valores atipicos (outliers).
    """
    return sum(
        abs(ai - bi)
        for ai, bi in zip(a, b)
    )
