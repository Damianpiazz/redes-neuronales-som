"""
Modulo de utilidades para carga y preprocesamiento de datos.

Funciones para:
- Cargar datos desde archivos CSV, TXT (tab-separated) y XLS
- Normalizar atributos numericos al rango [0, 1]
- Calcular matrices de correlacion entre atributos
"""

import csv

import xlrd


def load_csv(path):
    """
    Carga datos desde un archivo CSV usando DictReader.

    Cada fila se convierte en un diccionario donde las claves
    son los nombres de las columnas (header).
    """
    data = []

    with open(
        path,
        newline='',
        encoding='utf-8',
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            data.append(row)

    return data


def load_txt(path, separator='\t'):
    """
    Carga datos desde un archivo de texto separado por tabulaciones.

    La primera linea contiene los headers (nombres de columnas).
    Las lineas siguientes contienen los valores separados por el
    caracter de separacion (por defecto tabulacion).
    """
    data = []

    with open(
        path,
        newline='',
        encoding='utf-8',
    ) as file:

        lines = file.readlines()

        # La primera linea son los nombres de las columnas
        headers = (
            lines[0].strip().split(separator)
        )

        for line in lines[1:]:

            values = line.strip().split(separator)

            # Combina headers con valores en un diccionario
            row = dict(
                zip(headers, values)
            )

            data.append(row)

    return data


def load_xls(path):
    """
    Carga datos desde un archivo Excel (.xls) usando xlrd.

    Lee la primera hoja del workbook, usa la primera fila como headers
    y convierte cada fila restante en un diccionario.
    """
    # Abre el workbook Excel
    wb = xlrd.open_workbook(path)
    sh = wb.sheet_by_index(0)

    # Extrae los nombres de las columnas de la primera fila
    headers = [
        sh.cell_value(0, c)
        for c in range(sh.ncols)
    ]

    data = []

    # Itera sobre cada fila (desde la segunda)
    for r in range(1, sh.nrows):

        row = {}

        for c in range(sh.ncols):

            val = sh.cell_value(r, c)

            row[headers[c]] = val

        data.append(row)

    return data


def normalize(data, numeric_attrs):
    """
    Normaliza los atributos numericos al rango [0, 1].

    Usa la formula: (valor - min) / (max - min)
    Esto es esencial para el SOM porque las distancias
    se calculan en el espacio de atributos, y los atributos
    deben tener la misma escala para no dominar la distancia.

    Returns:
        Tupla de (datos_normalizados, minimos, maximos)
    """
    # Encuentra min y max de cada atributo numerico
    min_vals = {}
    max_vals = {}

    for attr in numeric_attrs:

        values = [
            float(row[attr])
            for row in data
        ]

        min_vals[attr] = min(values)
        max_vals[attr] = max(values)

    normalized = []

    for row in data:

        new_row = dict(row)

        for attr in numeric_attrs:

            val = float(new_row[attr])
            min_v = min_vals[attr]
            max_v = max_vals[attr]

            # Si todos los valores son iguales, normaliza a 0
            if max_v - min_v == 0:
                new_row[attr] = 0.0
            else:
                # Aplica la normalizacion min-max
                new_row[attr] = (
                    (val - min_v)
                    / (max_v - min_v)
                )

        normalized.append(new_row)

    return normalized, min_vals, max_vals


def denormalize(
    value,
    min_val,
    max_val,
):
    """
    Invierte la normalizacion para un valor individual.

    Aplica: valor_original = valor_normalizado * (max - min) + min
    Util para convertir valores del SOM de vuelta a la escala original.
    """
    return (
        value * (max_val - min_val)
        + min_val
    )


def correlation_matrix(data, numeric_attrs):
    """
    Calcula la matriz de correlacion de Pearson entre atributos numericos.

    La correlacion mide la relacion lineal entre dos variables:
    - +1: correlacion positiva perfecta
    -  0: sin relacion lineal
    - -1: correlacion negativa perfecta

    Se usa para identificar atributos redundantes que podrian
    eliminarse sin perder informacion.
    """
    n = len(numeric_attrs)

    m = len(data)

    # Calcula la media de cada atributo
    means = {}

    for attr in numeric_attrs:

        values = [
            float(row[attr])
            for row in data
        ]

        means[attr] = sum(values) / m

    # Calcula la desviacion estandar de cada atributo
    stds = {}

    for attr in numeric_attrs:

        values = [
            float(row[attr])
            for row in data
        ]

        # Varianza muestral (division por n-1)
        variance = sum(
            (v - means[attr]) ** 2
            for v in values
        ) / (m - 1)

        stds[attr] = variance ** 0.5

    # Construye la matriz de correlacion
    matrix = []

    for i in range(n):

        row = []

        for j in range(n):

            # Si la desviacion es cero, la correlacion es 0
            if stds[numeric_attrs[i]] == 0 or \
               stds[numeric_attrs[j]] == 0:

                row.append(0.0)

                continue

            # Calcula la covarianza entre los atributos i y j
            cov = sum(
                (
                    float(row[numeric_attrs[i]])
                    - means[numeric_attrs[i]]
                )
                * (
                    float(row[numeric_attrs[j]])
                    - means[numeric_attrs[j]]
                )
                for row in data
            ) / (m - 1)

            # Correlacion de Pearson = covarianza / (std_i * std_j)
            r = (
                cov
                / (
                    stds[numeric_attrs[i]]
                    * stds[numeric_attrs[j]]
                )
            )

            row.append(round(r, 4))

        matrix.append(row)

    return matrix


def find_correlated_pairs(
    matrix,
    numeric_attrs,
    threshold=0.8,
):
    """
    Encuentra pares de atributos altamente correlacionados.

    Recorre la matriz de correlacion y retorna los pares
    cuya correlacion absoluta supera el umbral (default: 0.8).
    Los pares se ordenan por correlacion absoluta descendente.

    Util para:
    - Detectar atributos redundantes
    - Seleccionar特征 relevantes
    - Reducir dimensionalidad
    """
    pairs = []

    n = len(numeric_attrs)

    for i in range(n):

        for j in range(i + 1, n):

            r = matrix[i][j]

            # Solo incluye pares con correlacion fuerte
            if abs(r) >= threshold:

                pairs.append(
                    (numeric_attrs[i],
                     numeric_attrs[j],
                     round(r, 4))
                )

    # Ordena por correlacion absoluta (mayor primero)
    pairs.sort(
        key=lambda x: abs(x[2]),
        reverse=True,
    )

    return pairs
