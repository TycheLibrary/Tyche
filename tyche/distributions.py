"""
This file contains the definitions of probability
distributions to be used with Tyche.
"""
from typing import Union

import numpy as np
from scipy import stats
from numpy.typing import ArrayLike


# TODO :
#   1) Create a TruncatedContinuousProbDist that can truncate arbitrary distributions.
#   2) Test the distribution's own shift/scale against LinearTransformedContinuousProbDist directly


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


class ContinuousProbDist(ProbDist):
    """
    Allows the representation of continuous probability distributions.
    """
    def __init__(self):
        super().__init__()

    def _shift(self, shift: float) -> 'ContinuousProbDist':
        return LinearTransformedContinuousProbDist(self, shift, 1)

    def _scale(self, scale: float) -> 'ContinuousProbDist':
        return LinearTransformedContinuousProbDist(self, 0, scale)

    def sample(self, rng: np.random.Generator, shape: Union[int, tuple, None] = None) -> float:
        """
        Samples a random value from this distribution.
        """
        # We use nextafter so that we can include the end-point 1 in the generated numbers.
        prob = rng.uniform(0, np.nextafter(1.0, 1), shape)
        return self.inverse_cdf(prob)

    def cdf(self, x: ArrayLike) -> ArrayLike:
        """ Evaluates the cumulative density function at x. """
        raise NotImplementedError("cdf is unimplemented for " + type(self).__name__)

    def pdf(self, x: ArrayLike) -> ArrayLike:
        """ Evaluates the probability density function at x. """
        raise NotImplementedError("pdf is unimplemented for " + type(self).__name__)

    def inverse_cdf(self, prob: ArrayLike) -> ArrayLike:
        """
        Calculates the x-value that would produce the given probability using
        the cumulative density function of this probability distribution.
        """
        raise NotImplementedError("inverse_cdf is unimplemented for " + type(self).__name__)

    def __add__(self, other: ProbDistLike) -> 'ContinuousProbDist':
        """
        Adds the other scalar or probability distribution to this probability distribution.
        """
        # Scalar additions such as UniformDist(0, 1) < 0.1
        if np.isscalar(other):
            return self._shift(float(other))

        # Distribution additions such as UniformDist(0, 1) + UniformDist(0.5, 1)
        """
        I would love for this to 'just work', but unfortunately this requires the
        convolution of the PDFs of the two distributions. These convolutions
        are known for fixed pairs of distributions. However, they require an infinite
        integral for the general case (for the convolution). Therefore, this cannot be
        implemented generally for any pair of distributions (or at least, I don't know
        how to do that). However, implementing this for common pairs of distributions may
        be worth it if this functionality would be helpful.

        Alternatively, monte-carlo methods can be used to estimate the addition of two
        probability distributions, but it is not exact. Therefore, I think it should be
        more explicitly chosen than overloading +.
        """
        raise NotImplementedError("Addition of distributions is not yet implemented")

    def __radd__(self, other: ProbDistLike) -> 'ContinuousProbDist':
        return self + other  # Add is commutative

    def __mul__(self, other: ProbDistLike) -> 'ContinuousProbDist':
        """
        Adds the other scalar or probability distribution to this probability distribution.
        """
        # Scalar additions such as UniformDist(0, 1) < 0.1
        if np.isscalar(other):
            return self._scale(float(other))

        # Distribution multiplications such as UniformDist(0, 1) * UniformDist(0.5, 1)
        raise NotImplementedError("Addition of distributions is not yet implemented")

    def __rmul__(self, other: ProbDistLike) -> 'ContinuousProbDist':
        return self * other  # Mul is commutative

    def __truediv__(self, other: ProbDistLike) -> 'ContinuousProbDist':
        """
        Adds the other scalar or probability distribution to this probability distribution.
        """
        return self * (1.0 / other)

    def __rtruediv__(self, other: ProbDistLike) -> 'ContinuousProbDist':
        raise NotImplementedError("Division by probability distributions is not yet implemented")

    def __sub__(self, other: ProbDistLike) -> 'ContinuousProbDist':
        """
        Subtracts the other scalar or probability distribution from this probability distribution.
        """
        return self + (-other)

    def __rsub__(self, other: ProbDistLike) -> 'ContinuousProbDist':
        return other + (-self)

    def __neg__(self) -> 'ContinuousProbDist':
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
        self.inverse_shift = -shift / scale
        self.inverse_scale = 1.0 / scale

    def _shift(self, shift: float) -> 'ContinuousProbDist':
        return LinearTransformedContinuousProbDist(self.dist, self.shift + shift, self.scale)

    def _scale(self, scale: float) -> 'ContinuousProbDist':
        return LinearTransformedContinuousProbDist(self.dist, self.shift * scale, self.scale * scale)

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


class UniformDist(ContinuousProbDist):
    """
    A uniform probability distribution.
    """
    def __init__(self, minimum: float, maximum: float):
        super().__init__()
        if maximum <= minimum:
            raise TycheDistributionsException("UniformDist maximum must be > than minimum. {} <= {}".format(
                maximum, minimum
            ))

        self.minimum = minimum
        self.maximum = maximum

    def cdf(self, x: ArrayLike) -> ArrayLike:
        return stats.uniform.cdf(x, loc=self.minimum, scale=self.maximum - self.minimum)

    def pdf(self, x: ArrayLike) -> ArrayLike:
        return stats.uniform.pdf(x, loc=self.minimum, scale=self.maximum - self.minimum)

    def inverse_cdf(self, prob: ArrayLike) -> ArrayLike:
        return stats.uniform.ppf(prob, loc=self.minimum, scale=self.maximum - self.minimum)

    def _shift(self, shift: float) -> 'ContinuousProbDist':
        return UniformDist(self.minimum + shift, self.maximum + shift)

    def _scale(self, scale: float) -> 'ContinuousProbDist':
        bound1 = self.minimum * scale
        bound2 = self.maximum * scale
        return UniformDist(min(bound1, bound2), max(bound1, bound2))

    def __str__(self):
        return "Uniform({}, {})".format(self.minimum, self.maximum)

    def __repr__(self):
        return "UniformDist(min={}, max={})".format(self.minimum, self.maximum)


class NormalDist(ContinuousProbDist):
    """
    A normal probability distribution.
    """
    def __init__(self, mean: float, std_dev: float):
        super().__init__()
        if std_dev <= 0:
            raise TycheDistributionsException("NormalDist std_dev must be > than 0. {} <= 0".format(std_dev))

        self.mean = mean
        self.std_dev = std_dev

    def cdf(self, x: ArrayLike) -> ArrayLike:
        return stats.norm.cdf(x, loc=self.mean, scale=self.std_dev)

    def pdf(self, x: ArrayLike) -> ArrayLike:
        return stats.norm.pdf(x, loc=self.mean, scale=self.std_dev)

    def inverse_cdf(self, prob: ArrayLike) -> ArrayLike:
        return stats.norm.ppf(prob, loc=self.mean, scale=self.std_dev)

    def _shift(self, shift: float) -> 'ContinuousProbDist':
        return NormalDist(self.mean + shift, self.std_dev)

    def _scale(self, scale: float) -> 'ContinuousProbDist':
        return NormalDist(
            self.mean * scale,
            abs(self.std_dev * scale)  # Normal distributions are symmetrical
        )

    def __str__(self):
        return "Normal({}, {})".format(self.mean, self.std_dev)

    def __repr__(self):
        return "NormalDist(mean={}, std_dev={})".format(self.mean, self.std_dev)


class TruncatedNormalDist(ContinuousProbDist):
    """
    A truncated normal probability distribution.
    """
    def __init__(self, mean: float, std_dev: float, minimum: float, maximum: float):
        super().__init__()
        if maximum <= minimum:
            raise TycheDistributionsException("TruncatedNormalDist maximum must be > than minimum. {} <= {}".format(
                maximum, minimum
            ))
        if std_dev <= 0:
            raise TycheDistributionsException("TruncatedNormalDist std_dev must be > than 0. {} <= 0".format(std_dev))

        self.mean = mean
        self.std_dev = std_dev
        self.minimum = minimum
        self.maximum = maximum

    @property
    def min_z_index(self):
        """ The z-index of the minimum of this truncated normal distribution. """
        return (self.minimum - self.mean) / self.std_dev

    @property
    def max_z_index(self):
        """ The z-index of the maximum of this truncated normal distribution. """
        return (self.maximum - self.mean) / self.std_dev

    def cdf(self, x: ArrayLike) -> ArrayLike:
        return stats.truncnorm.cdf(x, self.min_z_index, self.max_z_index, loc=self.mean, scale=self.std_dev)

    def pdf(self, x: ArrayLike) -> ArrayLike:
        return stats.truncnorm.pdf(x, self.min_z_index, self.max_z_index, loc=self.mean, scale=self.std_dev)

    def inverse_cdf(self, prob: ArrayLike) -> ArrayLike:
        return stats.truncnorm.ppf(prob, self.min_z_index, self.max_z_index, loc=self.mean, scale=self.std_dev)

    def _shift(self, shift: float) -> 'ContinuousProbDist':
        return TruncatedNormalDist(
            self.mean + shift,
            self.std_dev,
            self.minimum + shift,
            self.maximum + shift
        )

    def _scale(self, scale: float) -> 'ContinuousProbDist':
        bound1 = self.minimum * scale
        bound2 = self.maximum * scale
        return TruncatedNormalDist(
            self.mean * scale,
            abs(self.std_dev * scale),  # Normal distributions are symmetrical
            min(bound1, bound2),
            max(bound1, bound2)
        )

    def __str__(self):
        return "TruncatedNormal({}, {}, [{}, {}])".format(self.mean, self.std_dev, self.minimum, self.maximum)

    def __repr__(self):
        return "TruncatedNormalDist(mean={}, std_dev={}, min={}, max={})".format(
            self.mean, self.std_dev, self.minimum, self.maximum
        )
