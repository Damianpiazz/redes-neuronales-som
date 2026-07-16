"""
Modulo de visualizacion para el SOM.

Genera graficos para analizar los resultados del entrenamiento:
- U-Matrix: mapa de distancias entre neuronas vecinas
- Hit Map: frecuencia de activacion de cada neurona
- Cluster Map: dispersión de patrones coloreados por clase
- Component Planes: heatmap de los pesos por dimension
- Network Graph: grafo de la red SOM con aristas ponderadas

Tambien incluye funciones para imprimir representaciones ASCII en consola.
"""

import os

import matplotlib

# Usa el backend 'Agg' para generar graficos sin ventana (sin GUI)
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import networkx as nx

from src.distance import euclidean


def ensure_plots_dir():
    """Crea el directorio 'plots/' si no existe para guardar los graficos."""

    os.makedirs(
        'plots',
        exist_ok=True,
    )


def plot_umatrix(umatrix, prefix=''):
    """
    Genera un heatmap de la U-Matrix.

    La U-Matrix muestra las distancias promedio entre neuronas vecinas.
    Colores claros (amarillo) = neuronas similares (mismo cluster)
    Colores oscuros (rojo) = fronteras entre clusters
    """
    ensure_plots_dir()

    # Convierte la U-Matrix a array de numpy para seaborn
    data = np.array(umatrix)

    plt.figure(
        figsize=(8, 6),
    )

    # Heatmap con escala de colores YlOrRd (amarillo -> naranja -> rojo)
    sns.heatmap(
        data,
        cmap='YlOrRd',
        annot=False,
        square=True,
        linewidths=0.5,
        cbar_kws={
            'label': 'Distancia promedio',
        },
    )

    plt.title(
        'U-Matrix',
        fontsize=14,
        fontweight='bold',
    )

    plt.xlabel('Columna')
    plt.ylabel('Fila')

    plt.tight_layout()

    # Guarda el grafico como PNG en alta resolucion
    plt.savefig(
        f'plots/{prefix}umatrix.png',
        dpi=150,
    )

    plt.close()


def plot_hit_map(
    bmu_map,
    grid_rows,
    grid_cols,
    prefix='',
):
    """
    Genera un heatmap del mapa de hits (frecuencia de activacion).

    Muestra cuantos patrones de entrada fueron mapeados a cada neurona.
    Neuronas con muchos hits = regiones densas del espacio de datos.
    Neuronas con cero hits = regiones sin datos representativas.
    """
    ensure_plots_dir()

    # Crea una matriz de ceros con las dimensiones de la grilla
    matrix = np.zeros(
        (grid_rows, grid_cols),
    )

    # Llena la matriz con los conteos de hits
    for (r, c), hits in bmu_map.items():

        matrix[r][c] = hits

    plt.figure(
        figsize=(8, 6),
    )

    # Heatmap con escala de colores Azul (pocos hits) -> Oscuro (muchos hits)
    sns.heatmap(
        matrix,
        cmap='Blues',
        annot=True,
        fmt='.0f',
        square=True,
        linewidths=0.5,
        cbar_kws={
            'label': 'Observaciones',
        },
    )

    plt.title(
        'Hit Map',
        fontsize=14,
        fontweight='bold',
    )

    plt.xlabel('Columna')
    plt.ylabel('Fila')

    plt.tight_layout()

    plt.savefig(
        f'plots/{prefix}hit_map.png',
        dpi=150,
    )

    plt.close()


def plot_cluster_map(
    som,
    data,
    labels,
    prefix='',
):
    """
    Genera un mapa de clusters con scatter plot.

    Cada patron se proyecta a su BMU en la grilla y se colorea
    segun su clase (label). Permite visualizar como el SOM
    separa los diferentes grupos de datos.
    """
    ensure_plots_dir()

    # Obtiene las clases unicas y las ordena
    unique_labels = sorted(
        set(labels),
    )

    # Para cada patron, encuentra la posicion de su BMU en la grilla
    bmu_positions = []

    for pattern in data:

        bmu = som._find_bmu(pattern)

        bmu_positions.append(
            (bmu.row, bmu.col)
        )

    bmu_positions = np.array(
        bmu_positions,
    )

    plt.figure(
        figsize=(8, 6),
    )

    # Scatter plot: x=columna, y=fila, color=clase
    sns.scatterplot(
        x=bmu_positions[:, 1],
        y=bmu_positions[:, 0],
        hue=labels,
        palette='Set1',
        alpha=0.6,
        s=30,
        edgecolor='white',
        linewidth=0.3,
    )

    plt.title(
        'Cluster Map',
        fontsize=14,
        fontweight='bold',
    )

    plt.xlabel('Columna')
    plt.ylabel('Fila')

    # Configura los limites del eje para que coincidan con la grilla
    plt.xlim(-0.5, som.grid_cols - 0.5)
    plt.ylim(som.grid_rows - 0.5, -0.5)

    plt.legend(
        title='Clase',
        loc='upper right',
    )

    plt.tight_layout()

    plt.savefig(
        f'plots/{prefix}cluster_map.png',
        dpi=150,
    )

    plt.close()


def plot_component_planes(
    som,
    feature_names,
    prefix='',
):
    """
    Genera planos de componentes (Component Planes).

    Para cada dimension del espacio de entrada, crea un heatmap
    mostrando los valores de los pesos de esa dimension en toda la grilla.
    Permite ver como cada atributo se distribuye en la superficie del SOM.
    """
    ensure_plots_dir()

    dims = len(feature_names)

    # Calcula la disposition de subplots en grilla (maximo 3 columnas)
    cols = 3

    rows = (dims + cols - 1) // cols

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(
            cols * 4,
            rows * 3.5,
        ),
    )

    # Aplana el array de axes para iterar facilmente
    axes_flat = axes.flatten()

    for i in range(dims):
        # Extrae los pesos de la dimension i para toda la grilla
        matrix = np.zeros(
            (som.grid_rows, som.grid_cols),
        )

        for r in range(som.grid_rows):

            for c in range(som.grid_cols):

                matrix[r][c] = (
                    som.grid[r][c].weights[i]
                )

        # Heatmap con escala de colores viridis (morado -> verde -> amarillo)
        sns.heatmap(
            matrix,
            cmap='viridis',
            ax=axes_flat[i],
            square=True,
            linewidths=0.3,
            cbar_kws={
                'shrink': 0.8,
            },
        )

        # Titulo con el nombre del atributo
        axes_flat[i].set_title(
            feature_names[i],
            fontsize=10,
            fontweight='bold',
        )

    # Oculta los subplots vacios si hay menos atributos que espacios
    for j in range(dims, len(axes_flat)):

        axes_flat[j].set_visible(False)

    plt.suptitle(
        'Component Planes',
        fontsize=14,
        fontweight='bold',
        y=1.01,
    )

    plt.tight_layout()

    plt.savefig(
        f'plots/{prefix}component_planes.png',
        dpi=150,
        bbox_inches='tight',
    )

    plt.close()


def plot_network_graph(
    som,
    data,
    labels,
    prefix='',
):
    """
    Genera un grafo de la red SOM.

    Visualiza la grilla como un grafo donde:
    - Nodos = neuronas del SOM (coloreados por clase dominante)
    - Aristas = conexiones entre neuronas adyacentes
    - Grosor/color de aristas = distancia entre pesos (mas fino = mas similar)

    Permite ver la estructura topologica de la red y como
    los diferentes grupos de datos se distribuyen en la grilla.
    """
    ensure_plots_dir()

    # Crea un grafo grid 2D con NetworkX
    G = nx.grid_2d_graph(
        som.grid_rows,
        som.grid_cols,
    )

    # Calcula el mapa de hits para determinar que patron domina cada neurona
    bmu_map = som.get_bmu_map(data)

    # Mapea cada patron a su BMU para determinar la clase dominante
    bmu_lookup = {}

    for pattern in data:

        bmu = som._find_bmu(pattern)

        bmu_lookup.setdefault(
            (bmu.row, bmu.col), []
        ).append(pattern)

    unique_labels = sorted(
        set(labels),
    )

    # Paleta de colores para las clases
    cmap = plt.cm.Set1

    # Mapea cada label a un indice numerico
    label_idx = {
        label: i
        for i, label in enumerate(
            unique_labels
        )
    }

    # Determina el color de cada nodo (clase dominante o gris si sin datos)
    node_colors = []

    for node in G.nodes():

        r, c = node

        if (r, c) in bmu_lookup:

            patterns = bmu_lookup[(r, c)]

            # Encuentra la clase dominante entre los patrones mapeados a esta neurona
            idx = sum(
                label_idx.get(labels[i], 0)
                for i, p in enumerate(data)
                if any(
                    abs(p[j] - patterns[0][j]) < 0.001
                    for j in range(len(p))
                )
            ) % len(unique_labels)

            dominant = max(
                set(labels[i]
                    for i, p in enumerate(data)
                    if any(
                        abs(p[j] - patterns[0][j]) < 0.001
                        for j in range(len(p))
                    )),
                key=lambda x: sum(
                    1 for i, p in enumerate(data)
                    if labels[i] == x
                    and any(
                        abs(p[j] - patterns[0][j]) < 0.001
                        for j in range(len(p))
                    )
                ),
            )

            node_colors.append(
                cmap(
                    label_idx[dominant]
                    / max(len(unique_labels) - 1, 1)
                )
            )

        else:
            # Neurona sin datos: color gris transparente
            node_colors.append(
                (0.85, 0.85, 0.85, 0.5)
            )

    # Calcula los pesos de las aristas (distancia euclidiana entre pesos de neuronas)
    edge_weights = []

    for u, v in G.edges():

        r1, c1 = u
        r2, c2 = v

        w1 = som.grid[r1][c1].weights
        w2 = som.grid[r2][c2].weights

        dist = euclidean(w1, w2)

        edge_weights.append(dist)

    # Normaliza los pesos para el color y grosor de las aristas
    max_w = max(edge_weights) if edge_weights else 1

    # Color: de azul (similar) a rojo (diferente)
    edge_colors = [
        plt.cm.RdYlBu_r(w / max_w)
        for w in edge_weights
    ]

    # Grosor: mas fino = mas similar, mas grueso = mas diferente
    edge_widths = [
        0.5 + 2.0 * (w / max_w)
        for w in edge_weights
    ]

    fig, ax = plt.subplots(
        figsize=(10, 8),
    )

    # Posiciona los nodos en la grilla (columna=x, -fila=y para que arriba sea fila 0)
    pos = {
        node: (node[1], -node[0])
        for node in G.nodes()
    }

    # Dibuja las aristas con color y grosor segun la distancia
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=edge_colors,
        width=edge_widths,
        alpha=0.6,
        ax=ax,
    )

    # Dibuja los nodos con color segun la clase dominante
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=120,
        edgecolors='black',
        linewidths=0.5,
        ax=ax,
    )

    # Crea la leyenda con los colores de cada clase
    handles = [
        mpatches.Patch(
            color=cmap(
                label_idx[l]
                / max(len(unique_labels) - 1, 1)
            ),
            label=l,
        )
        for l in unique_labels
    ]

    ax.legend(
        handles=handles,
        title='Clase',
        loc='upper right',
        fontsize=9,
    )

    ax.set_title(
        'SOM Network Graph',
        fontsize=14,
        fontweight='bold',
    )

    ax.set_axis_off()

    plt.tight_layout()

    plt.savefig(
        f'plots/{prefix}network_graph.png',
        dpi=150,
    )

    plt.close()


def print_umatrix(umatrix, console):
    """
    Imprime la U-Matrix como arte ASCII en la consola.

    Usa una escala de simbolos: . : - = + * # % @
    Donde simbolos mas simples = valores bajos (neuronas similares)
    y simbolos mas densos = valores altos (fronteras entre clusters).
    """
    console.print("\nU-Matrix:")

    for row in umatrix:

        line = ""

        for val in row:
            # Escala el valor a un indice de simbolo (0-9)
            level = int(
                min(val * 10, 9)
            )

            # Paleta de simbolos de menor a mayor intensidad
            symbols = " .:-=+*#%@"

            idx = min(
                level,
                len(symbols) - 1,
            )

            line += (
                symbols[idx] + " "
            )

        console.print(line)


def print_hit_map(
    bmu_map,
    grid_rows,
    grid_cols,
    console,
):
    """
    Imprime el mapa de hits (frecuencia de activacion) en la consola.

    Muestra un numero junto a un simbolo para cada neurona:
    - '.' = sin hits (neurona no activada)
    - Simbolo + numero = cantidad de patrones mapeados
    """
    console.print("\nHit Map:")

    # Encuentra el maximo de hits para escalar los simbolos
    max_hits = max(
        bmu_map.values(),
    )

    for r in range(grid_rows):

        line = ""

        for c in range(grid_cols):

            hits = bmu_map[(r, c)]

            if hits == 0:
                line += " . "
            else:
                # Escala los hits a un indice de simbolo
                level = int(
                    hits / max_hits * 9
                )

                symbols = (
                    " .:-=+*#%@"
                )

                idx = min(
                    level,
                    len(symbols) - 1,
                )

                # Simbolo + numero de hits
                line += (
                    symbols[idx]
                    + f"{hits:2d}"
                )

        console.print(line)


def print_som_config(
    grid_rows,
    grid_cols,
    epochs,
    console,
):
    """
    Imprime la configuracion del SOM en la consola.

    Muestra las dimensiones de la grilla, total de neuronas
    y cantidad de epocas de entrenamiento.
    """
    total = grid_rows * grid_cols

    console.print("\nConfiguracion:")

    console.print(
        f"  Grilla: {grid_rows} x {grid_cols}"
    )

    console.print(
        f"  Total neuronas: {total}"
    )

    console.print(
        f"  Epocas: {epochs}"
    )
