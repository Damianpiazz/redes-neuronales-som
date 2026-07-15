import random
import math

from src.neuron import Neuron
from src.distance import euclidean
from src.neighborhood import gaussian


class SOM:
    def __init__(
        self,
        grid_rows,
        grid_cols,
        dimensions,
        learning_rate=0.5,
        sigma=None,
        epochs=100,
    ):

        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.dimensions = dimensions
        self.initial_lr = learning_rate
        self.initial_sigma = (
            sigma
            if sigma is not None
            else max(grid_rows, grid_cols) / 2
        )
        self.epochs = epochs
        self.grid = self._init_grid()

    def _init_grid(self):

        grid = []

        for r in range(self.grid_rows):

            row = []

            for c in range(self.grid_cols):

                neuron = Neuron(
                    self.dimensions,
                    r,
                    c,
                )

                row.append(neuron)

            grid.append(row)

        return grid

    def _find_bmu(self, pattern):

        best_neuron = None
        best_dist = float('inf')

        for r in range(self.grid_rows):

            for c in range(self.grid_cols):

                neuron = self.grid[r][c]

                dist = euclidean(
                    pattern,
                    neuron.weights,
                )

                if dist < best_dist:

                    best_dist = dist
                    best_neuron = neuron

        return best_neuron

    def _update_weights(
        self,
        bmu,
        pattern,
        lr,
        sigma,
    ):

        for r in range(self.grid_rows):

            for c in range(self.grid_cols):

                neuron = self.grid[r][c]

                h = gaussian(
                    bmu.row,
                    bmu.col,
                    neuron.row,
                    neuron.col,
                    sigma,
                )

                for i in range(self.dimensions):

                    neuron.weights[i] += (
                        lr
                        * h
                        * (
                            pattern[i]
                            - neuron.weights[i]
                        )
                    )

    def fit(self, data):

        self.data = data

        for epoch in range(self.epochs):

            t = epoch

            decay = math.exp(-t / self.epochs)

            lr = self.initial_lr * decay

            sigma = (
                self.initial_sigma * decay
            )

            for pattern in data:

                bmu = self._find_bmu(pattern)

                self._update_weights(
                    bmu,
                    pattern,
                    lr,
                    sigma,
                )

        return self

    def quantization_error(self, data):

        total = 0

        for pattern in data:

            bmu = self._find_bmu(pattern)

            total += euclidean(
                pattern,
                bmu.weights,
            )

        return total / len(data)

    def topographic_error(self, data):

        errors = 0

        for pattern in data:

            distances = []

            for r in range(self.grid_rows):

                for c in range(self.grid_cols):

                    neuron = self.grid[r][c]

                    dist = euclidean(
                        pattern,
                        neuron.weights,
                    )

                    distances.append(
                        (dist, neuron)
                    )

            distances.sort(
                key=lambda x: x[0],
            )

            bmu1 = distances[0][1]
            bmu2 = distances[1][1]

            if bmu1.distance_to(bmu2) > 1:
                errors += 1

        return errors / len(data)

    def get_bmu_map(self, data):

        bmu_map = {}

        for r in range(self.grid_rows):

            for c in range(self.grid_cols):

                bmu_map[(r, c)] = 0

        for pattern in data:

            bmu = self._find_bmu(pattern)

            bmu_map[
                (bmu.row, bmu.col)
            ] += 1

        return bmu_map

    def get_umatrix(self):

        umatrix = []

        for r in range(self.grid_rows):

            row = []

            for c in range(self.grid_cols):

                neighbors = []

                for dr in [-1, 0, 1]:

                    for dc in [-1, 0, 1]:

                        if dr == 0 and dc == 0:
                            continue

                        nr = r + dr
                        nc = c + dc

                        if (
                            0 <= nr
                            and nr < self.grid_rows
                            and 0 <= nc
                            and nc < self.grid_cols
                        ):

                            dist = euclidean(
                                self.grid[r][c].weights,
                                self.grid[nr][nc].weights,
                            )

                            neighbors.append(dist)

                avg = (
                    sum(neighbors)
                    / len(neighbors)
                    if neighbors
                    else 0
                )

                row.append(avg)

            umatrix.append(row)

        return umatrix
