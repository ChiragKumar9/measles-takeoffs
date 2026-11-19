"""Summaries of a state's outbreaks for the state-level figures."""

import numpy as np


def weighted_mean_population(records):
    """
    The mean population of a state's outbreaks: `records` are (population,
    weight) pairs, one per outbreak, and the mean is weighted by the weights
    (it is the plain mean if the weights add up to 0). None when there is no
    outbreak, as then there is nothing to average.
    """
    if len(records) == 0:
        return None
    total_weight = sum(weight for _, weight in records)
    if total_weight > 0:
        return (
            sum(population * weight for population, weight in records)
            / total_weight
        )
    return np.mean([population for population, _ in records])
