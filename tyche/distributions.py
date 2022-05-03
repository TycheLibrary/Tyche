"""
This file contains the definitions of probability
distributions to be used with Tyche.
"""
from typing import Union

import numpy as np
from scipy import stats
from scipy.stats.distributions import rv_frozen
from numpy.typing import ArrayLike


ProbDistLike: type = Union[float, int, 'ProbabilityDistribution']


class TycheDistributionsException(Exception):
    """
    Class for detailing exceptions with distributions.
    """
    def __init__(self, message: str):
        self.message = "TycheDistributionsException: " + message


class ProbDist:
    """
    Allows the representation of a probability distribution in Tyche.
    """
    def __init__(self):
        pass

    def sample(self, rng: np.random.Generator, shape: Union[int, tuple, None] = None) -> ArrayLike:
        """ Samples size random values from this distribution. """
        raise NotImplementedError("sample is unimplemented for " + type(self).__name__)

    def cdf(self, x: ArrayLike) -> ArrayLike:
        """ Evaluates the cumulative density function at x. """
        raise NotImplementedError("cdf is unimplemented for " + type(self).__name__)

    def pdf(self, x: ArrayLike) -> ArrayLike:
        """ Evaluates the probability density function at x. """
        raise NotImplementedError("pdf is unimplemented for " + type(self).__name__)

    def inverse_cdf(self, prob: ArrayLike) -> ArrayLike:
        """ Calculates the x-value that would produce the given probability using the cdf. """
        raise NotImplementedError("inverse_cdf is unimplemented for " + type(self).__name__)


class ContinuousProbDist(ProbDist):
    """
    Allows the representation of continuous probability distributions.
    """
    def __init__(self):
        super().__init__()

    def sample(self, rng: np.random.Generator, shape: Union[int, tuple, None] = None) -> float:
        """
        Samples a random value from this distribution.
        """
        # We use nextafter so that we can include the end-point 1 in the generated numbers.
        prob = rng.uniform(0, np.nextafter(1.0, 1), shape)
        return self.inverse_cdf(prob)

    def __add__(self, other: ProbDistLike) -> ProbDistLike:
        """
        Adds the other scalar or probability distribution to this probability distribution.
        """
        # Scalar additions such as UniformDist(0, 1) < 0.1
        if np.isscalar(other):
            other = float(other)
            return LinearTransformedContinuousProbDist(self, other, 1)

        # Distribution additions such as UniformDist(0, 1) + UniformDist(0.5, 1)
        raise NotImplementedError("Addition of distributions is not yet implemented")

    def __radd__(self, other: ProbDistLike) -> ProbDistLike:
        return self + other  # Add is commutative

    def __mul__(self, other: ProbDistLike) -> ProbDistLike:
        """
        Adds the other scalar or probability distribution to this probability distribution.
        """
        # Scalar additions such as UniformDist(0, 1) < 0.1
        if np.isscalar(other):
            other = float(other)
            return LinearTransformedContinuousProbDist(self, 0, other)

        # Distribution additions such as UniformDist(0, 1) + UniformDist(0.5, 1)
        raise NotImplementedError("Addition of distributions is not yet implemented")

    def __rmul__(self, other: ProbDistLike) -> ProbDistLike:
        return self * other  # Mul is commutative

    def __truediv__(self, other: ProbDistLike) -> ProbDistLike:
        """
        Adds the other scalar or probability distribution to this probability distribution.
        """
        return self * (1.0 / other)

    def __rtruediv__(self, other: ProbDistLike) -> ProbDistLike:
        raise NotImplementedError("Division by probability distributions is not yet implemented")

    def __sub__(self, other: ProbDistLike) -> ProbDistLike:
        """
        Subtracts the other scalar or probability distribution from this probability distribution.
        """
        return self + (-other)

    def __rsub__(self, other: ProbDistLike) -> ProbDistLike:
        return other + (-self)

    def __neg__(self) -> ProbDistLike:
        """
        Negates the values of this probability distribution.
        """
        return self * -1

    def __lt__(self, other: ProbDistLike) -> ArrayLike:
        """
        Returns the probability that a value sampled from this distribution
        is less than a value sampled from other.
        """
        # Scalar comparisons such as UniformDist(0, 1) < 0.1
        if np.isscalar(other):
            other = float(other)
            return self.cdf(other)

        # Distribution comparisons such as UniformDist(0, 1) < UniformDist(0.5, 1)
        return self - other < 0

    def __le__(self, other: ProbDistLike) -> ArrayLike:
        """
        Returns the probability that a value sampled from this distribution
        is less than or equal to a value sampled from other.
        As this is continuous, this is the same as less than.
        """
        return self < other

    def __gt__(self, other: ProbDistLike) -> ArrayLike:
        """
        Returns the probability that a value sampled from this distribution
        is less than or equal to a value sampled from other.
        As this is continuous, this is the same as less than.
        """
        return 1.0 - (self <= other)

    def __ge__(self, other: ProbDistLike) -> ArrayLike:
        """
        Returns the probability that a value sampled from this distribution
        is less than or equal to a value sampled from other.
        As this is continuous, this is the same as less than.
        """
        return self > other


class LinearTransformedContinuousProbDist(ContinuousProbDist):
    """ Applies a linear transformation to a ContinuousProbDist. """
    def __init__(self, dist: ContinuousProbDist, shift: float, scale: float):
        super().__init__()
        self.dist = dist
        self.shift = shift
        self.scale = scale
        self.inverse_shift = -shift
        self.inverse_scale = 1.0 / scale

    def __str__(self):
        if self.shift == 0:
            return "({} * {})".format(self.scale, str(self.dist))
        if self.scale == 1:
            return "({} + {})".format(self.shift, str(self.dist))

        return "({} + {} * {})".format(
            self.shift, self.scale, str(self.dist)
        )

    def __repr__(self):
        return "LinearTransformation(shift={}, scale={}, dist={})".format(
            self.shift, self.scale, repr(self.dist)
        )

    def _transform(self, x: ArrayLike) -> ArrayLike:
        """ Applies the shift and scale of this transformation to x. """
        return self.shift + self.scale * x

    def _inverse_transform(self, x: ArrayLike) -> ArrayLike:
        """ Applies the shift and scale of this transformation to x. """
        return self.inverse_shift + self.inverse_scale * x

    def sample(self, rng: np.random.Generator, shape: int = None) -> ArrayLike:
        return self._transform(self.dist.sample(rng, shape))

    def cdf(self, x: ArrayLike) -> ArrayLike:
        values = self.dist.cdf(self._inverse_transform(x))
        return values if self.scale >= 0 else 1 - values

    def pdf(self, x: ArrayLike) -> ArrayLike:
        return self.dist.pdf(self._inverse_transform(x))

    def inverse_cdf(self, prob: ArrayLike) -> ArrayLike:
        prob = prob if self.scale >= 0 else 1 - prob
        return self._transform(self.dist.inverse_cdf(prob))


class SciPyStatsContinuousProbDist(ContinuousProbDist):
    """
    Represents a continuous probability distribution that uses a scipy.stats
    generator under-the-hood.
    """
    def __init__(self, generator: rv_frozen):
        super().__init__()
        self.generator = generator

    def cdf(self, x: ArrayLike) -> ArrayLike:
        """ Evaluates the cumulative density function at x. """
        return self.generator.cdf(x)

    def pdf(self, x: ArrayLike) -> ArrayLike:
        """ Evaluates the probability density function at x. """
        return self.generator.pdf(x)

    def inverse_cdf(self, prob: ArrayLike) -> ArrayLike:
        """ Calculates the x-value that would produce the given probability using the cdf. """
        return self.generator.ppf(prob)


class UniformDist(SciPyStatsContinuousProbDist):
    """
    A continuous uniform probability distribution.
    """
    def __init__(self, minimum: float, maximum: float):
        super().__init__(UniformDist._gen_dist(minimum, maximum))
        self.minimum = minimum
        self.maximum = maximum

    @staticmethod
    def _gen_dist(minimum: float, maximum: float) -> rv_frozen:
        if maximum < minimum:
            raise TycheDistributionsException(
                "UniformDist maximum must be >= to minimum. {} < {}".format(maximum, minimum)
            )
        return stats.uniform(loc=minimum, scale=maximum - minimum)

    def __str__(self):
        return "Uniform({}, {})".format(self.minimum, self.maximum)

    def __repr__(self):
        return "UniformDist(min={}, max={})".format(self.minimum, self.maximum)
