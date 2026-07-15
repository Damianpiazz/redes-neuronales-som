import math


def euclidean(a, b):

    return math.sqrt(
        sum(
            (ai - bi) ** 2
            for ai, bi in zip(a, b)
        )
    )


def manhattan(a, b):

    return sum(
        abs(ai - bi)
        for ai, bi in zip(a, b)
    )
