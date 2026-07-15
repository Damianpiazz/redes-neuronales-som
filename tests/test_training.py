from src.som import SOM


def test_grid_created():

    som = SOM(
        grid_rows=5,
        grid_cols=5,
        dimensions=3,
        epochs=10,
    )

    assert len(som.grid) == 5

    assert len(som.grid[0]) == 5


def test_training_reduces_error():

    data = [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.5, 0.0],
        [0.5, 0.0, 0.5],
    ]

    som = SOM(
        grid_rows=3,
        grid_cols=3,
        dimensions=3,
        learning_rate=0.5,
        epochs=100,
    )

    som.fit(data)

    qe = som.quantization_error(data)

    assert qe < 1.0


def test_bmu_returns_neuron():

    som = SOM(
        grid_rows=3,
        grid_cols=3,
        dimensions=2,
        epochs=10,
    )

    pattern = [0.5, 0.5]

    bmu = som._find_bmu(pattern)

    assert bmu is not None

    assert 0 <= bmu.row < 3

    assert 0 <= bmu.col < 3
