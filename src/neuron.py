import random


class Neuron:
    def __init__(self, dimensions, row, col):
        self.weights = [
            random.random()
            for _ in range(dimensions)
        ]
        self.row = row
        self.col = col

    def distance_to(self, other):
        return (
            abs(self.row - other.row)
            + abs(self.col - other.col)
        )

    def __repr__(self):
        return (
            f"Neuron({self.row}, {self.col})"
        )
