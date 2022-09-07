"""
This module contains probability calculations that are
used in many places in Tyche.
"""
from typing import Union

import numpy as np


class TycheProbabilityException(Exception):
    """
    An exception type that is thrown when errors occur in
    the use of the probability methods in this module.
    """
    def __init__(self, message: str):
        self.message = "TycheProbabilityException: " + message


def random_probability(rng: np.random.Generator, shape: Union[int, tuple, None] = None):
    """
    Returns a random probability from the range [0, 1].
    """
    # We use nextafter so that we can include the
    # end-point (1) in the generated numbers.
    return rng.uniform(0, np.nextafter(1.0, 1), shape)


def uncertain_bayes_rule(prob_concept: float, prob_obs: float, prob_obs_given_concept: float, likelihood: float):
    """
    Bayes' rule for uncertain observations.

    Let A be a concept, and B be an observation with a phi=likelihood chance of being true.
    P(A|B) = phi * P(B|A) * P(A) / P(B) + (1 - phi) * (1 - P(B|A)) * P(A) / (1 - P(B))

    To avoid division by zero, the following special cases are also used:
    - If likelihood is 0, then P(A|B) = (1 - likelihood) * (1 - P(B|A)) * P(A) / (1 - P(B))
    - If likelihood is 1, then P(A|B) = likelihood * P(B|A) * P(A) / P(B)

    The rules for this were derived by applying bayes rule to the event that the observation occurred
    or was incorrectly observed ((event with likelihood) OR (NOT event with 1 - likelihood)).
    Therefore, a likelihood of 0 represents that the observation was observed to be false
    (this is equivalent to observing NOT observation).
    See: https://stats.stackexchange.com/questions/345200/applying-bayess-theorem-when-evidence-is-uncertain
    """
    if prob_obs <= 0 and likelihood > 0:
        raise TycheProbabilityException(
            "The observation is impossible, but the likelihood of the observation is not 0")
    if prob_obs >= 1 and likelihood < 1:
        raise TycheProbabilityException(
            "The observation is certain, but the likelihood of the observation is not 1")

    concept_given_obs_prob = 0
    if likelihood > 0:
        obs_correct_prob = prob_concept * prob_obs_given_concept / prob_obs
        concept_given_obs_prob += likelihood * obs_correct_prob
    if likelihood < 1:
        obs_incorrect_prob = prob_concept * (1 - prob_obs_given_concept) / (1 - prob_obs)
        concept_given_obs_prob += (1 - likelihood) * obs_incorrect_prob

    return concept_given_obs_prob
