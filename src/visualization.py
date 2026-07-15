import os

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import networkx as nx

from src.distance import euclidean


def ensure_plots_dir():

    os.makedirs(
        'plots',
        exist_ok=True,
    )


def plot_umatrix(umatrix, prefix=''):

    ensure_plots_dir()

    data = np.array(umatrix)

    plt.figure(
        figsize=(8, 6),
    )

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

    ensure_plots_dir()

    matrix = np.zeros(
        (grid_rows, grid_cols),
    )

    for (r, c), hits in bmu_map.items():

        matrix[r][c] = hits

    plt.figure(
        figsize=(8, 6),
    )

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

    ensure_plots_dir()

    unique_labels = sorted(
        set(labels),
    )

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

    ensure_plots_dir()

    dims = len(feature_names)

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

    axes_flat = axes.flatten()

    for i in range(dims):

        matrix = np.zeros(
            (som.grid_rows, som.grid_cols),
        )

        for r in range(som.grid_rows):

            for c in range(som.grid_cols):

                matrix[r][c] = (
                    som.grid[r][c].weights[i]
                )

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

        axes_flat[i].set_title(
            feature_names[i],
            fontsize=10,
            fontweight='bold',
        )

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

    ensure_plots_dir()

    G = nx.grid_2d_graph(
        som.grid_rows,
        som.grid_cols,
    )

    bmu_map = som.get_bmu_map(data)

    bmu_lookup = {}

    for pattern in data:

        bmu = som._find_bmu(pattern)

        bmu_lookup.setdefault(
            (bmu.row, bmu.col), []
        ).append(pattern)

    unique_labels = sorted(
        set(labels),
    )

    cmap = plt.cm.Set1

    label_idx = {
        label: i
        for i, label in enumerate(
            unique_labels
        )
    }

    node_colors = []

    for node in G.nodes():

        r, c = node

        if (r, c) in bmu_lookup:

            patterns = bmu_lookup[(r, c)]

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

            node_colors.append(
                (0.85, 0.85, 0.85, 0.5)
            )

    edge_weights = []

    for u, v in G.edges():

        r1, c1 = u
        r2, c2 = v

        w1 = som.grid[r1][c1].weights
        w2 = som.grid[r2][c2].weights

        dist = euclidean(w1, w2)

        edge_weights.append(dist)

    max_w = max(edge_weights) if edge_weights else 1

    edge_colors = [
        plt.cm.RdYlBu_r(w / max_w)
        for w in edge_weights
    ]

    edge_widths = [
        0.5 + 2.0 * (w / max_w)
        for w in edge_weights
    ]

    fig, ax = plt.subplots(
        figsize=(10, 8),
    )

    pos = {
        node: (node[1], -node[0])
        for node in G.nodes()
    }

    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=edge_colors,
        width=edge_widths,
        alpha=0.6,
        ax=ax,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=120,
        edgecolors='black',
        linewidths=0.5,
        ax=ax,
    )

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

    console.print("\nU-Matrix:")

    for row in umatrix:

        line = ""

        for val in row:

            level = int(
                min(val * 10, 9)
            )

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

    console.print("\nHit Map:")

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
