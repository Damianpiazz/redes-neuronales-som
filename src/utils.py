import csv

import xlrd


def load_csv(path):

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

    data = []

    with open(
        path,
        newline='',
        encoding='utf-8',
    ) as file:

        lines = file.readlines()

        headers = (
            lines[0].strip().split(separator)
        )

        for line in lines[1:]:

            values = line.strip().split(separator)

            row = dict(
                zip(headers, values)
            )

            data.append(row)

    return data


def load_xls(path):

    wb = xlrd.open_workbook(path)
    sh = wb.sheet_by_index(0)

    headers = [
        sh.cell_value(0, c)
        for c in range(sh.ncols)
    ]

    data = []

    for r in range(1, sh.nrows):

        row = {}

        for c in range(sh.ncols):

            val = sh.cell_value(r, c)

            row[headers[c]] = val

        data.append(row)

    return data


def normalize(data, numeric_attrs):

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

            if max_v - min_v == 0:
                new_row[attr] = 0.0
            else:
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

    return (
        value * (max_val - min_val)
        + min_val
    )


def correlation_matrix(data, numeric_attrs):

    n = len(numeric_attrs)

    m = len(data)

    means = {}

    for attr in numeric_attrs:

        values = [
            float(row[attr])
            for row in data
        ]

        means[attr] = sum(values) / m

    stds = {}

    for attr in numeric_attrs:

        values = [
            float(row[attr])
            for row in data
        ]

        variance = sum(
            (v - means[attr]) ** 2
            for v in values
        ) / (m - 1)

        stds[attr] = variance ** 0.5

    matrix = []

    for i in range(n):

        row = []

        for j in range(n):

            if stds[numeric_attrs[i]] == 0 or \
               stds[numeric_attrs[j]] == 0:

                row.append(0.0)

                continue

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

    pairs = []

    n = len(numeric_attrs)

    for i in range(n):

        for j in range(i + 1, n):

            r = matrix[i][j]

            if abs(r) >= threshold:

                pairs.append(
                    (numeric_attrs[i],
                     numeric_attrs[j],
                     round(r, 4))
                )

    pairs.sort(
        key=lambda x: abs(x[2]),
        reverse=True,
    )

    return pairs
