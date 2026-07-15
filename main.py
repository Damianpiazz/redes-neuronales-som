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

    console.print(f"\n[bold]{text}[/bold]")
    console.print("-" * len(text))


def print_dataset_info(data, config):

    total = len(data)

    header("Dataset")

    table = Table(
        show_header=True,
    )

    table.add_column("ATRIBUTO")
    table.add_column("TIPO", justify="right")

    numeric = config['numeric_attrs']

    for attr in data[0].keys():

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

    header("Resultados")

    table = Table(
        show_header=True,
    )

    table.add_column("METRICA")
    table.add_column("VALOR", justify="right")

    table.add_row(
        "Quantization Error",
        f"{qe:.6f}",
    )

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

    header(config['name'].upper())

    if config['loader'] == 'txt':

        raw_data = load_txt(
            config['path'],
        )

    else:

        raw_data = load_xls(
            config['path'],
        )

    print_dataset_info(raw_data, config)

    numeric = config['numeric_attrs']

    data_norm, _, _ = normalize(
        raw_data,
        numeric,
    )

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

    som = SOM(
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        dimensions=len(numeric),
        learning_rate=0.5,
        epochs=epochs,
    )

    som.fit(float_data)

    qe = som.quantization_error(
        float_data,
    )

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

    umatrix = som.get_umatrix()

    print_umatrix(umatrix, console)

    bmu_map = som.get_bmu_map(
        float_data,
    )

    print_hit_map(
        bmu_map,
        grid_rows,
        grid_cols,
        console,
    )

    labels = [
        str(row[config['label_attr']])
        for row in raw_data
    ]

    prefix = f"{key}_"

    console.print(
        "\nGenerando graficos en plots/..."
    )

    plot_umatrix(umatrix, prefix=prefix)

    plot_hit_map(
        bmu_map,
        grid_rows,
        grid_cols,
        prefix=prefix,
    )

    plot_cluster_map(
        som,
        float_data,
        labels,
        prefix=prefix,
    )

    plot_component_planes(
        som,
        numeric,
        prefix=prefix,
    )

    plot_network_graph(
        som,
        float_data,
        labels,
        prefix=prefix,
    )

    return qe, te


if __name__ == '__main__':

    arg = (
        sys.argv[1]
        if len(sys.argv) > 1
        else None
    )

    if arg is None:

        for key, cfg in DATASETS.items():

            run_dataset(key, cfg)

    elif arg in DATASETS:

        run_dataset(arg, DATASETS[arg])

    else:

        print_usage()
        sys.exit(1)
