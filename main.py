"""
Punto de entrada principal del programa.

Carga datasets, entrena un SOM (Self-Organizing Map) y genera
visualizaciones de resultados (U-Matrix, Hit Map, clusteres, etc.).
"""

import sys

from rich.console import Console
from rich.table import Table

from src.som import SOM
from src.utils import (
    load_txt,
    load_xls,
    normalize,
)

from src.visualization import (
    print_umatrix,
    print_hit_map,
    print_som_config,
    plot_umatrix,
    plot_hit_map,
    plot_cluster_map,
    plot_component_planes,
    plot_network_graph,
)


console = Console()


# Diccionario de datasets disponibles con sus configuraciones.
# Cada entrada define: nombre, formato de carga, ruta, atributos,
# atributo de etiqueta, dimensiones de la grilla y épocas de entrenamiento.
DATASETS = {
    'breast': {
        'name': 'Breast Cancer Wisconsin',
        'loader': 'txt',
        'path': 'data/breast.txt',
        'numeric_attrs': [
            'clump',
            'ucellsize',
            'ucellshape',
            'mgadhesion',
            'sepics',
            'bnuclei',
            'bchromatin',
            'normnucl',
            'mitoses',
        ],
        'label_attr': 'class',
        'grid_rows': 10,
        'grid_cols': 10,
        'epochs': 200,
    },
    'cars': {
        'name': 'Cars (MPG Prediction)',
        'loader': 'xls',
        'path': 'data/cars.xls',
        'numeric_attrs': [
            'MPG',
            'Weight',
            'Drive_Ratio',
            'Horsepower',
            'Displacement',
            'Cylinders',
        ],
        'label_attr': 'Country',
        'grid_rows': 6,
        'grid_cols': 6,
        'epochs': 200,
    },
}


def header(text):
    """Imprime un encabezado formateado en la consola."""

    console.print(f"\n[bold]{text}[/bold]")
    console.print("-" * len(text))


def print_dataset_info(data, config):
    """Muestra una tabla con los atributos del dataset y su tipo."""

    total = len(data)

    header("Dataset")

    table = Table(
        show_header=True,
    )

    table.add_column("ATRIBUTO")
    table.add_column("TIPO", justify="right")

    numeric = config['numeric_attrs']

    for attr in data[0].keys():
        # Clasifica cada atributo: numerico, etiqueta o categorico
        tipo = (
            "Numerico"
            if attr in numeric
            else (
                "Etiqueta"
                if attr == config['label_attr']
                else "Categorico"
            )
        )

        table.add_row(attr, tipo)

    console.print(table)

    console.print(
        f"\nTotal de instancias: {total}"
    )


def print_results(
    qe,
    te,
    grid_rows,
    grid_cols,
    epochs,
):
    """Muestra las metricas de calidad del SOM entrenado."""

    header("Resultados")

    table = Table(
        show_header=True,
    )

    table.add_column("METRICA")
    table.add_column("VALOR", justify="right")

    # Error de cuantizacion: promedio de distancias entre cada patron y su BMU
    table.add_row(
        "Quantization Error",
        f"{qe:.6f}",
    )

    # Error topografico: proporcion de patrones cuyos dos BMUs no son vecinos
    table.add_row(
        "Topographic Error",
        f"{te:.6f}",
    )

    table.add_row(
        "Grilla",
        f"{grid_rows} x {grid_cols}",
    )

    table.add_row(
        "Epocas",
        str(epochs),
    )

    console.print(table)


def print_usage():
    """Muestra el uso correcto del programa y los datasets disponibles."""

    console.print(
        "Uso: python main.py [dataset]\n"
    )

    console.print(
        "Datasets disponibles:"
    )

    for key, cfg in DATASETS.items():

        console.print(
            f"  {key} - {cfg['name']}"
        )

    console.print(
        "\nSin argumento ejecuta todos."
    )


def run_dataset(key, config):
    """Ejecuta el pipeline completo para un dataset: carga, normaliza, entrena y visualiza."""

    header(config['name'].upper())

    if config['loader'] == 'txt':
        # Carga archivos de texto separados por tabulaciones
        raw_data = load_txt(
            config['path'],
        )

    else:
        # Carga archivos Excel (.xls)
        raw_data = load_xls(
            config['path'],
        )

    print_dataset_info(raw_data, config)

    numeric = config['numeric_attrs']

    # Normaliza los atributos numericos al rango [0, 1]
    data_norm, _, _ = normalize(
        raw_data,
        numeric,
    )

    # Convierte los datos normalizados a listas de floats para el SOM
    float_data = [
        [float(row[a]) for a in numeric]
        for row in data_norm
    ]

    grid_rows = config['grid_rows']
    grid_cols = config['grid_cols']
    epochs = config['epochs']

    print_som_config(
        grid_rows,
        grid_cols,
        epochs,
        console,
    )

    console.print(
        "\nEntrenando SOM..."
    )

    # Crea y entrena el SOM con los parametros configurados
    som = SOM(
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        dimensions=len(numeric),
        learning_rate=0.5,
        epochs=epochs,
    )

    # Entrena el SOM con los datos normalizados
    som.fit(float_data)

    # Calcula el error de cuantizacion (distancia promedio patron-BMU)
    qe = som.quantization_error(
        float_data,
    )

    # Calcula el error topografico (preservacion de la topologia)
    te = som.topographic_error(
        float_data,
    )

    print_results(
        qe,
        te,
        grid_rows,
        grid_cols,
        epochs,
    )

    # Calcula la U-Matrix (distancia promedio entre neuronas vecinas)
    umatrix = som.get_umatrix()

    # Imprime la U-Matrix como arte ASCII en consola
    print_umatrix(umatrix, console)

    # Calcula el mapa de hits (cantidad de patrones mapeados a cada neurona)
    bmu_map = som.get_bmu_map(
        float_data,
    )

    # Imprime el mapa de hits en consola
    print_hit_map(
        bmu_map,
        grid_rows,
        grid_cols,
        console,
    )

    # Extrae las etiquetas de clase de los datos originales
    labels = [
        str(row[config['label_attr']])
        for row in raw_data
    ]

    prefix = f"{key}_"

    console.print(
        "\nGenerando graficos en plots/..."
    )

    # Genera y guarda los graficos como imagenes PNG
    plot_umatrix(umatrix, prefix=prefix)

    plot_hit_map(
        bmu_map,
        grid_rows,
        grid_cols,
        prefix=prefix,
    )

    # Mapa de clusters: scatter plot de los BMUs coloreados por clase
    plot_cluster_map(
        som,
        float_data,
        labels,
        prefix=prefix,
    )

    # Planos de componentes: heatmap de los pesos de cada dimension
    plot_component_planes(
        som,
        numeric,
        prefix=prefix,
    )

    # Grafo de la red SOM con aristas ponderadas por distancia entre pesos
    plot_network_graph(
        som,
        float_data,
        labels,
        prefix=prefix,
    )

    return qe, te


if __name__ == '__main__':

    # Obtiene el argumento de linea de comandos (nombre del dataset)
    arg = (
        sys.argv[1]
        if len(sys.argv) > 1
        else None
    )

    if arg is None:
        # Sin argumento: ejecuta todos los datasets disponibles
        for key, cfg in DATASETS.items():

            run_dataset(key, cfg)

    elif arg in DATASETS:
        # Ejecuta solo el dataset especificado
        run_dataset(arg, DATASETS[arg])

    else:
        # Dataset no valido: muestra uso y termina
        print_usage()
        sys.exit(1)
