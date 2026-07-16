"""
Modulo principal del Self-Organizing Map (SOM).

Implementa el algoritmo de Kohonen para entrenamiento no supervisado.
El SOM proyecta datos de alta dimensionalidad a una grilla 2D,
preservando la topologia de los datos de entrada.

Algoritmo por epoch:
1. Para cada patron de entrada, encontrar el BMU (Best Matching Unit)
2. Actualizar los pesos de todas las neuronas usando la funcion de vecindad
3. Decair el learning rate y sigma (radio de vecindad) exponencialmente
"""

import random
import math

from src.neuron import Neuron
from src.distance import euclidean
from src.neighborhood import gaussian


class SOM:
    """Self-Organizing Map de Kohonen."""

    def __init__(
        self,
        grid_rows,
        grid_cols,
        dimensions,
        learning_rate=0.5,
        sigma=None,
        epochs=100,
    ):
        """
        Inicializa el SOM con los parametros de configuracion.

        Args:
            grid_rows: cantidad de filas de la grilla
            grid_cols: cantidad de columnas de la grilla
            dimensions: dimensiones del espacio de entrada (atributos)
            learning_rate: tasa de aprendizaje inicial (default: 0.5)
            sigma: radio inicial de vecindad (default: max(filas, columnas)/2)
            epochs: cantidad de epocas de entrenamiento
        """
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.dimensions = dimensions
        self.initial_lr = learning_rate
        # Si no se especifica sigma, se calcula como la mitad del lado mayor
        self.initial_sigma = (
            sigma
            if sigma is not None
            else max(grid_rows, grid_cols) / 2
        )
        self.epochs = epochs
        # Inicializa la grilla de neuronas con pesos aleatorios
        self.grid = self._init_grid()

    def _init_grid(self):
        """
        Inicializa la grilla de neuronas con pesos aleatorios.

        Cada neurona se posiciona en (fila, columna) de la grilla
        y tiene un vector de pesos de 'dimensions' componentes.
        """
        grid = []

        for r in range(self.grid_rows):

            row = []

            for c in range(self.grid_cols):
                # Crea una neurona con pesos aleatorios en la posicion (r, c)
                neuron = Neuron(
                    self.dimensions,
                    r,
                    c,
                )

                row.append(neuron)

            grid.append(row)

        return grid

    def _find_bmu(self, pattern):
        """
        Encuentra la Best Matching Unit (BMU) para un patron dado.

        La BMU es la neurona cuyos pesos son mas cercanos al patron
        de entrada (menor distancia euclidiana). Este es el paso
        competitivo del algoritmo SOM.
        """
        best_neuron = None
        best_dist = float('inf')

        for r in range(self.grid_rows):

            for c in range(self.grid_cols):

                neuron = self.grid[r][c]

                # Calcula la distancia euclidiana entre el patron y los pesos
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
        """
        Actualiza los pesos de todas las neuronas de la grilla.

        Para cada neurona, calcula el factor de vecindad (h) usando
        la funcion gaussiana, y ajusta sus pesos segun:

            w(t+1) = w(t) + lr * h * (patron - w(t))

        - lr: learning rate (controla la magnitud del ajuste)
        - h: funcion de vecindad (mayor para neuronas cercanas al BMU)
        - (patron - w): direccion del ajuste (acercar pesos al patron)
        """
        for r in range(self.grid_rows):

            for c in range(self.grid_cols):

                neuron = self.grid[r][c]

                # Calcula la influencia del BMU sobre esta neurona
                h = gaussian(
                    bmu.row,
                    bmu.col,
                    neuron.row,
                    neuron.col,
                    sigma,
                )

                # Ajusta cada dimension del vector de pesos
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
        """
        Entrena el SOM con los datos de entrada.

        Para cada epoch:
        1. Calcula los valores de lr y sigma con decaimiento exponencial
        2. Para cada patron, encuentra el BMU y actualiza los pesos
        3. La vecindad y el learning rate se reducen con el tiempo,
           permitiendo convergencia gradual
        """
        self.data = data

        for epoch in range(self.epochs):

            t = epoch

            # Decaimiento exponencial: controla la rapidez de convergencia
            decay = math.exp(-t / self.epochs)

            # Learning rate y sigma se reducen exponencialmente
            lr = self.initial_lr * decay

            sigma = (
                self.initial_sigma * decay
            )

            for pattern in data:
                # Paso competitivo: encontrar la neurona ganadora
                bmu = self._find_bmu(pattern)

                # Paso cooperativo: actualizar pesos de toda la grilla
                self._update_weights(
                    bmu,
                    pattern,
                    lr,
                    sigma,
                )

        return self

    def quantization_error(self, data):
        """
        Calcula el error de cuantizacion.

        Promedio de las distancias euclidianas entre cada patron
        de entrada y su BMU. Indica que tan bien representan
        las neuronas a los datos originales.
        Valores bajos = buena representacion.
        """
        total = 0

        for pattern in data:

            bmu = self._find_bmu(pattern)

            total += euclidean(
                pattern,
                bmu.weights,
            )

        return total / len(data)

    def topographic_error(self, data):
        """
        Calcula el error topografico.

        Para cada patron, encuentra las dos neuronas mas cercanas (BMU1 y BMU2).
        Si no son vecinas en la grilla (distancia Manhattan > 1), cuenta como error.
        Indica si la SOM preserva la topologia de los datos.
        Valores bajos = buena preservacion topologica.
        """
        errors = 0

        for pattern in data:

            distances = []

            # Calcula distancia de cada neurona al patron
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

            # Ordena por distancia para encontrar las dos mas cercanas
            distances.sort(
                key=lambda x: x[0],
            )

            bmu1 = distances[0][1]
            bmu2 = distances[1][1]

            # Si las dos BMUs no son vecinas en la grilla, es un error topografico
            if bmu1.distance_to(bmu2) > 1:
                errors += 1

        return errors / len(data)

    def get_bmu_map(self, data):
        """
        Genera el mapa de hits (frecuencia de activacion).

        Cuenta cuantos patrones de entrada son mapeados a cada neurona
        de la grilla. Las neuronas con muchos hits son representativas
        de regiones densas del espacio de datos.
        """
        bmu_map = {}

        # Inicializa el mapa con ceros para todas las posiciones
        for r in range(self.grid_rows):

            for c in range(self.grid_cols):

                bmu_map[(r, c)] = 0

        # Cuenta los hits para cada patron
        for pattern in data:

            bmu = self._find_bmu(pattern)

            bmu_map[
                (bmu.row, bmu.col)
            ] += 1

        return bmu_map

    def get_umatrix(self):
        """
        Calcula la U-Matrix (Unified Distance Matrix).

        Para cada neurona, calcula la distancia promedio entre sus pesos
        y los de sus vecinas adyacentes (8 vecinos en la grilla).

        - Valores ALTOS = frontera entre clusters (neuronas muy diferentes)
        - Valores BAJOS =区域内 homogenea (neuronas similares, mismo cluster)

        Se usa para visualizar la estructura de clusters del SOM.
        """
        umatrix = []

        for r in range(self.grid_rows):

            row = []

            for c in range(self.grid_cols):

                neighbors = []

                # Recorre los 8 vecinos adyacentes (3x3 sin el centro)
                for dr in [-1, 0, 1]:

                    for dc in [-1, 0, 1]:

                        if dr == 0 and dc == 0:
                            continue

                        nr = r + dr
                        nc = c + dc

                        # Verifica que el vecino este dentro de la grilla
                        if (
                            0 <= nr
                            and nr < self.grid_rows
                            and 0 <= nc
                            and nc < self.grid_cols
                        ):

                            # Distancia euclidiana entre los pesos de las neuronas
                            dist = euclidean(
                                self.grid[r][c].weights,
                                self.grid[nr][nc].weights,
                            )

                            neighbors.append(dist)

                # Promedio de distancias a los vecinos
                avg = (
                    sum(neighbors)
                    / len(neighbors)
                    if neighbors
                    else 0
                )

                row.append(avg)

            umatrix.append(row)

        return umatrix
