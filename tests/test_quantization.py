"""
Tests para los metricas de calidad del SOM: error de cuantizacion y topografico.
"""
from src.som import SOM


def test_quantization_error_positive():
    """Verifica que el error de cuantizacion es no negativo."""

    data = [
        [0.0, 0.0],
        [1.0, 1.0],
        [0.5, 0.5],
    ]

    som = SOM(
        grid_rows=3,
        grid_cols=3,
        dimensions=2,
        epochs=50,
    )

    som.fit(data)

    qe = som.quantization_error(data)

    assert qe >= 0


def test_quantization_error_small_single():
    """Verifica que un SOM con una neurona y un patron tiene error casi cero."""

    data = [
        [0.0, 0.0],
    ]

    som = SOM(
        grid_rows=1,
        grid_cols=1,
        dimensions=2,
        epochs=100,
    )

    som.fit(data)

    qe = som.quantization_error(data)

    assert qe < 0.01


def test_topographic_error_range():
    """Verifica que el error topografico esta en el rango [0, 1]."""

    data = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ]

    som = SOM(
        grid_rows=3,
        grid_cols=3,
        dimensions=2,
        epochs=100,
    )

    som.fit(data)

    te = som.topographic_error(data)

    assert 0 <= te <= 1
