"""
Tests para la funcion de distancia euclidiana y la busqueda del BMU.
"""
from src.distance import euclidean
from src.neuron import Neuron


def test_euclidean_same_point():
    """Verifica que la distancia entre dos puntos identicos es cero."""

    a = [1.0, 2.0, 3.0]
    b = [1.0, 2.0, 3.0]

    result = euclidean(a, b)

    assert result == 0


def test_euclidean_basic():
    """Verifica la distancia euclidiana basica (triangulo 3-4-5)."""

    a = [0.0, 0.0]
    b = [3.0, 4.0]

    result = euclidean(a, b)

    assert result == 5.0


def test_bmu_finds_closest():
    """Verifica que el BMU se encuentra la neurona mas cercana al patron."""

    n1 = Neuron(3, 0, 0)
    n1.weights = [1.0, 1.0, 1.0]

    n2 = Neuron(3, 0, 1)
    n2.weights = [5.0, 5.0, 5.0]

    pattern = [1.2, 1.1, 1.3]

    d1 = euclidean(pattern, n1.weights)
    d2 = euclidean(pattern, n2.weights)

    assert d1 < d2
