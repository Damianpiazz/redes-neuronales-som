"""
Modulo que define la clase Neuron.

Cada neurona del SOM tiene pesos aleatorios y una posicion en la grilla.
La distancia entre neuronas se calcula con la distancia Manhattan (taxi).
"""

import random


class Neuron:
    """Neurona individual del Self-Organizing Map."""

    def __init__(self, dimensions, row, col):
        """
        Inicializa una neurona con pesos aleatorios.

        Args:
            dimensions: cantidad de dimensiones del espacio de entrada
            row: fila en la grilla del SOM
            col: columna en la grona del SOM
        """
        # Pesos inicializados aleatoriamente en [0, 1)
        self.weights = [
            random.random()
            for _ in range(dimensions)
        ]
        self.row = row
        self.col = col

    def distance_to(self, other):
        """
        Calcula la distancia Manhattan (taxi) entre esta neurona y otra.

        Se usa para determinar si dos neuronas son vecinas en la grilla.
        Distancia 1 = vecinas adyacentes, >1 = no son vecinas directas.
        """
        return (
            abs(self.row - other.row)
            + abs(self.col - other.col)
        )

    def __repr__(self):
        """Representacion textual de la neurona (fila, columna)."""
        return (
            f"Neuron({self.row}, {self.col})"
        )
