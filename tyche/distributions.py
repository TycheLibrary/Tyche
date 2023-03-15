"""
This module contains several classes of probability distributions
that can be easily manipulated (e.g., shift, scale, or add to
distributions to get new distributions).

Example::

    person_height_cm = NormalDist(mean=180, std_dev=5)
    heel_height_cm = NormalDist(mean=5, std_dev=3)

    total_height_cm = person_height_cm + heel_height_cm
    total_height_ft = total_height_cm * 0.0328084

    hoop_height_ft = 10
    distance_to_hoop_ft = hoop_height_ft - total_height_ft

    # Result: distance_to_hoop_ft = Normal(mean=3.930, std_dev=0.191)

The following probability distributions are currently available:

- `UniformDist <#tyche.distributions.UniformDist>`_: A uniform probability distribution.
- `NormalDist <#tyche.distributions.NormalDist>`_: A normal probability distribution.

This aims to allow the more flexible and intuitive use of probability
distributions, by reducing the amount of manual work that is required
to manipulate and use them.
"""
from __future__ import annotations

import math
import typing
from typing import Union, Optional

import numpy as np
from scipy import stats
from numpy.typing import ArrayLike

from tyche.probability import random_probability

ProbDistLike: type = Union[float, int, 'ProbDist']


class TycheDistributionsException(Exception):
    """
    An exception type that is thrown when errors occur in
    the construction or use of probability distributions.
    """
    def __init__(self, message: str):
        """
        Parameters:
            message: A message describing the reason that this
              exception was raised.
        """
        self.message = "TycheDistributionsException: " + message


class ProbDist:
    """
    A probability distribution over a space of numeric values.
    """
    def __init__(self):
        pass

    def sample(self, rng: np.random.Generator, shape: Union[int, tuple, None] = None) -> ArrayLike:
        """
        Samples random values from this distribution.

        Parameters:
            rng: A NumPy random number generator that is used to generate random samples
              from this distribution.
            shape: Configures the format of the returned samples from this probability
              distribution. By default, a single sampled value is returned. If an int
              is provided, then a NumPy array of `shape` values will be returned. If a
              tuple is provided, then a NumPy array of sampled values will be returned
              with the tuple used as the array's shape.
        """
        raise NotImplementedError("sample is unimplemented for " + type(self).__name__)


class ContinuousProbDist(ProbDist):
    """
    A probability distribution over a continuous space of numeric values.
    """
    def __init__(self):
        super().__init__()

    def truncate(self, minimum: float, maximum: float) -> 'ContinuousProbDist':
        """
        Returns a new distribution that represents the truncation of
        this distribution to the range [minimum, maximum].

        Parameters:
            minimum: A lower limit on the numeric values in this distribution
              after truncation.
            maximum: An upper limit on the numeric values in this distribution
              after truncation.
        """
        return TruncatedContinuousProbDist(self, minimum, maximum)

    def _shift(self, shift: float) -> 'ContinuousProbDist':
        """
        Returns a new distribution that represents a shift of this
        distribution by `shift`.

        Parameters:
            shift: The numerical amount to shift the values in this
              distribution by.
        """
        return LinearTransformContinuousProbDist(self, shift, 1)

    def _scale(self, scale: float) -> 'ContinuousProbDist':
        """
        Returns a new distribution that represents a scale of this
        distribution by `scale`.

        Parameters:
            scale: The numerical amount to multiply the values in this
              distribution by.
        """
        return LinearTransformContinuousProbDist(self, 0, scale)

    def sample(self, rng: np.random.Generator, shape: Union[int, tuple, None] = None) -> ArrayLike:
        return self.inverse_cdf(random_probability(rng, shape))

    def cdf(self, x: ArrayLike) -> ArrayLike:
        """
        Evaluates the cumulative density function of this distribution
        at `x`. The resulting value should fall in the range [0, 1],
        although numerical error can cause slightly lower or higher
        values to be returned.

        Parameters:
            x: The point or points at which to evaluate the cumulative density
              function of this distribution. If a numerical value is provided,
              a numerical value is returned. If a NumPy array of values is
              provided, then a NumPy array of values will be returned that
              matches the shape of `x`.
        """
        raise NotImplementedError("cdf is unimplemented for " + type(self).__name__)

    def mean(self) -> float:
        """
        Evaluates to the mean value of this distribution.
        """
        return self.inverse_cdf(0.5)

    def variance(self) -> float:
        """
        Evaluates to the variance of this distribution.
        """
        raise NotImplementedError("variance is unimplemented for " + type(self).__name__)

    def std_dev(self) -> float:
        """
        Evaluates to the standard deviation of this distribution.
        """
        return math.sqrt(self.variance())

    def pdf(self, x: ArrayLike) -> ArrayLike:
        """
        Evaluates the probability density function of this distribution
        at `x`. The resulting value should fall in the range [0, 1],
        although numerical error can cause slightly lower or higher
        values to be returned.

        Parameters:
            x: The point or points at which to evaluate the probability density
              function of this distribution. If a numerical value is provided,
              a numerical value is returned. If a NumPy array of values is
              provided, then a NumPy array of values will be returned that
              matches the shape of `x`.
        """
        raise NotImplementedError("pdf is unimplemented for " + type(self).__name__)

    def inverse_cdf(self, prob: ArrayLike) -> ArrayLike:
        """
        Evaluates the inverse cumulative density function of this distribution
        at `prob`. This involves calculating the value in this distribution
        that would produce the given probability, `prob`, if it was used to
        evaluate the cumulative density function of this distribution.

        Parameters:
            prob: The point or points at which to evaluate the inverse cumulative
              density function of this distribution. If a numerical value is
              provided, a numerical value is returned. If a NumPy array of values
              is provided, then a NumPy array of values will be returned that
              matches the shape of `prob`.
        """
        raise NotImplementedError("inverse_cdf is unimplemented for " + type(self).__name__)

    def _try_add_to_distribution(self, other: ContinuousProbDist) -> Optional['ContinuousProbDist']:
        """
        Attempts to add this continuous probability distribution to the other continuous
        probability distribution. If this is not supported, then None should be returned.
        """
        return None

    def __add__(self, other: ProbDistLike) -> 'ContinuousProbDist':
        """
        Adds the other scalar or probability distribution to this probability distribution.
        Sampling values from the resulting continuous probability distribution is equivalent
        to sampling values from this and the other probability distributions, and then
        adding them.
        """
        # Scalar additions such as UniformDist(0, 1) + 0.1
        if np.isscalar(other):
            return self._shift(float(other))

        if not isinstance(other, ContinuousProbDist):
            raise NotImplementedError(
                f"Unable to add values of type {type(other)} to continuous probability distributions")

        result = self._try_add_to_distribution(other)
        if result is not None:
            return result

        result = other._try_add_to_distribution(self)
        if result is not None:
            return result

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
        raise NotImplementedError(
            f"Addition of distributions of type {type(self)} and {type(other)} is not yet supported")

    def __radd__(self, other: ProbDistLike) -> 'ContinuousProbDist':
        return self.__add__(other)  # Add is commutative

    def __mul__(self, other: ProbDistLike) -> 'ContinuousProbDist':
        """
        Multiplies the other scalar or probability distribution by this probability distribution.
        """
        # Scalar multiplications such as UniformDist(0, 1) * 0.1
        if np.isscalar(other):
            return self._scale(float(other))

        # Distribution multiplications such as UniformDist(0, 1) * UniformDist(0.5, 1)
        raise NotImplementedError("Multiplication of distributions is not yet implemented")

    def __rmul__(self, other: ProbDistLike) -> 'ContinuousProbDist':
        return self.__mul__(other)  # Mul is commutative

    def __truediv__(self, other: ProbDistLike) -> 'ContinuousProbDist':
        """
        Adds the other scalar or probability distribution to this probability distribution.
        """
        return self * (1.0 / other)

    def __rtruediv__(self, other: ProbDistLike) -> 'ContinuousProbDist':
        raise TycheDistributionsException("Division by probability distributions is not yet implemented")

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
        is greater than a value sampled from other.
        """
        return 1.0 - (self <= other)

    def __ge__(self, other: ProbDistLike) -> ArrayLike:
        """
        Returns the probability that a value sampled from this distribution
        is greater than or equal to a value sampled from other.
        As this is continuous, this is the same as greater than.
        """
        return self > other


class LinearTransformContinuousProbDist(ContinuousProbDist):
    """
    Applies a linear transformation (scale + shift) to a ContinuousProbDist.
    The scaling of values is performed before the shift of values.
    """
    def __init__(self, dist: ContinuousProbDist, shift: float, scale: float):
        """
        Parameters:
            dist: A delegate continuous probability distribution to scale and shift.
            shift: A shift to be applied to the delegate distribution, `dist`, after
              the scale is applied.
            scale: A scale to be applied to the delegate distribution, `dist`, before
              the shift is applied.
        """
        super().__init__()
        if scale == 0:
            raise TycheDistributionsException("The scale must be non-zero")

        self._dist = dist
        self._linear_shift = shift
        self._linear_scale = scale

    def _shift(self, shift: float) -> 'ContinuousProbDist':
        return LinearTransformContinuousProbDist(self._dist, self._linear_shift + shift, self._linear_scale)

    def _scale(self, scale: float) -> 'ContinuousProbDist':
        return LinearTransformContinuousProbDist(self._dist, self._linear_shift * scale, self._linear_scale * scale)

    def _transform(self, x: ArrayLike) -> ArrayLike:
        """ Applies the shift and scale of this transformation to x. """
        return self._linear_shift + self._linear_scale * x

    def _inverse_transform(self, y: ArrayLike) -> ArrayLike:
        """ Applies the inverse shift and scale of this transformation to y. """
        return y / self._linear_scale  - self._linear_shift / self._linear_scale

    def sample(self, rng: np.random.Generator, shape: int = None) -> ArrayLike:
        return self._transform(self._dist.sample(rng, shape))

    def cdf(self, x: ArrayLike) -> ArrayLike:
        values = self._dist.cdf(self._inverse_transform(x))
        return values if self._linear_scale >= 0 else 1 - values

    def variance(self) -> float:
        return self._dist.variance() * self._linear_scale**2

    def pdf(self, x: ArrayLike) -> ArrayLike:
        return self._dist.pdf(self._inverse_transform(x)) / abs(self._linear_scale)

    def inverse_cdf(self, prob: ArrayLike) -> ArrayLike:
        prob = prob if self._linear_scale >= 0 else 1 - prob
        return self._transform(self._dist.inverse_cdf(prob))

    def __str__(self):
        if self._linear_shift == 0:
            return f"({self._linear_scale} * {str(self._dist)})"
        if self._linear_scale == 1:
            return f"({self._linear_shift} + {str(self._dist)})"
        if self._linear_scale < 0:
            return f"({self._linear_shift} - {abs(self._linear_scale)} * {str(self._dist)})"
        
        return f"({self._linear_shift} + {self._linear_scale} * {str(self._dist)})"

    def __repr__(self):
        return (
            f"LinearTransformDist(shift={self._linear_shift}, "
            f"scale={self._linear_scale}, dist={repr(self._dist)})"
        )


class TruncatedContinuousProbDist(ContinuousProbDist):
    """
    Applies a truncation to a ContinuousProbDist. All values below `minimum`
    and above `maximum` are discarded.
    """
    def __init__(self, dist: ContinuousProbDist, minimum: float, maximum: float):
        """
        Parameters:
            dist: A delegate continuous probability distribution to truncate.
            minimum: The lower bound to use to truncate the delegate distribution.
            maximum: The upper bound to use to truncate the delegate distribution.
        """
        super().__init__()
        if maximum <= minimum:
            raise TycheDistributionsException(
                f"TruncatedNormalDist maximum must be > than minimum. {maximum} <= {minimum}"
            )

        self._dist = dist
        self._minimum = minimum
        self._maximum = maximum
        self._lower_cdf = dist.cdf(minimum)
        self._upper_cdf = dist.cdf(maximum)
        if self._lower_cdf >= self._upper_cdf:
            raise TycheDistributionsException(
                f"The truncation of [{minimum}, {maximum}] applied to {dist} would result "
                f"in an all-zero probability distribution"
            )

        self._inverse_transform_cdf_mul = self._upper_cdf - self._lower_cdf
        self._transform_cdf_mul = 1.0 / (self._upper_cdf - self._lower_cdf)

    def truncate(self, minimum: float, maximum: float) -> 'ContinuousProbDist':
        if minimum >= self._minimum and maximum <= self._maximum:
            return self

        return TruncatedContinuousProbDist(
            self._dist,
            min(self._minimum, minimum),
            min(self._maximum, maximum)
        )

    def cdf(self, x: ArrayLike) -> ArrayLike:
        values = (self._dist.cdf(x) - self._lower_cdf) * self._transform_cdf_mul
        return np.clip(values, 0, 1)

    def pdf(self, x: ArrayLike) -> ArrayLike:
        if np.isscalar(x) and (x < self._minimum or x > self._maximum):
            return 0

        result = self._dist.pdf(x) * self._transform_cdf_mul
        if not np.isscalar(x):
            result[np.where((x < self._minimum) | (x > self._maximum))] = 0

        return result

    def inverse_cdf(self, prob: ArrayLike) -> ArrayLike:
        return self._dist.inverse_cdf(prob * self._inverse_transform_cdf_mul + self._lower_cdf)

    def __str__(self):
        return f"Truncated([{self._minimum}, {self._maximum}], {self._dist})"

    def __repr__(self):
        return f"TruncatedDist(min={self._minimum}, max={self._maximum}, dist={repr(self._dist)})"


class UniformDist(ContinuousProbDist):
    """
    A uniform probability distribution over the range [minimum, maximum].
    All values within this range have equal probability of being sampled.
    """
    def __init__(self, minimum: float, maximum: float):
        """
        Parameters:
            minimum: The lower bound on values that could be sampled from this
              uniform probability distribution.
            maximum: The upper bound on values that could be sampled from this
              uniform probability distribution.
        """
        super().__init__()
        if maximum <= minimum:
            raise TycheDistributionsException(
                f"UniformDist maximum must be > than minimum. {maximum} <= {minimum}"
            )

        self._minimum = minimum
        self._maximum = maximum

    def _shift(self, shift: float) -> 'ContinuousProbDist':
        return UniformDist(self._minimum + shift, self._maximum + shift)

    def _scale(self, scale: float) -> 'ContinuousProbDist':
        if scale == 0:
            raise TycheDistributionsException("The scale must be non-zero")

        bound1 = self._minimum * scale
        bound2 = self._maximum * scale
        return UniformDist(min(bound1, bound2), max(bound1, bound2))

    def cdf(self, x: ArrayLike) -> ArrayLike:
        return stats.uniform.cdf(x, loc=self._minimum, scale=self._maximum - self._minimum)

    def variance(self) -> float:
        return (self._maximum - self._minimum)**2 / 12

    def pdf(self, x: ArrayLike) -> ArrayLike:
        return stats.uniform.pdf(x, loc=self._minimum, scale=self._maximum - self._minimum)

    def inverse_cdf(self, prob: ArrayLike) -> ArrayLike:
        return stats.uniform.ppf(prob, loc=self._minimum, scale=self._maximum - self._minimum)

    def __str__(self):
        return f"Uniform({self._minimum:.3f} to {self._maximum:.3f})"

    def __repr__(self):
        return f"UniformDist(min={self._minimum}, max={self._maximum})"


class NormalDist(ContinuousProbDist):
    """
    A normal probability distribution centered around `mean` with a
    standard deviation of `std_dev`.
    """
    def __init__(self, mean: float, std_dev: float):
        """
        Parameters:
            mean: The mean of this normal distribution.
            std_dev: The standard deviation of this normal distribution.
        """
        super().__init__()
        if std_dev <= 0:
            raise TycheDistributionsException(
                f"NormalDist std_dev must be > than 0. {std_dev} <= 0"
            )

        self._mean = mean
        self._std_dev = std_dev

    def _shift(self, shift: float) -> 'ContinuousProbDist':
        return NormalDist(self._mean + shift, self._std_dev)

    def _scale(self, scale: float) -> 'ContinuousProbDist':
        if scale == 0:
            raise TycheDistributionsException("The scale must be non-zero")

        return NormalDist(
            self._mean * scale,
            abs(self._std_dev * scale)  # Normal distributions are symmetrical
        )

    def _try_add_to_distribution(self, other: ContinuousProbDist) -> Optional[ContinuousProbDist]:
        # Allow addition of normal distributions to themselves.
        if type(other) == NormalDist:
            other_norm = typing.cast(NormalDist, other)
            new_mean = self._mean + other_norm._mean
            new_std_dev = (self._std_dev**2 + other_norm._std_dev**2)**0.5
            return NormalDist(new_mean, new_std_dev)

        return None

    def cdf(self, x: ArrayLike) -> ArrayLike:
        return stats.norm.cdf(x, loc=self._mean, scale=self._std_dev)

    def variance(self) -> float:
        return self._std_dev**2

    def std_dev(self) -> float:
        return self._std_dev

    def pdf(self, x: ArrayLike) -> ArrayLike:
        return stats.norm.pdf(x, loc=self._mean, scale=self._std_dev)

    def inverse_cdf(self, prob: ArrayLike) -> ArrayLike:
        return stats.norm.ppf(prob, loc=self._mean, scale=self._std_dev)

    def __str__(self):
        return f"Normal(mean={self._mean:.3f}, std_dev={self._std_dev:.3f})"

    def __repr__(self):
        return f"NormalDist(mean={self._mean}, std_dev={self._std_dev})"
