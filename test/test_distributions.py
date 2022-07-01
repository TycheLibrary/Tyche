import sys
import unittest

from tyche.distributions import *


class TestDistributions(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_uniform(self):
        """
        Tests UniformDist.
        """
        rng = np.random.default_rng()
        self._test_dist_min_max(rng, UniformDist(0, 1), 0, 1)
        self._test_dist_min_max(rng, UniformDist(-1, 0), -1, 0)
        self._test_dist_min_max(rng, UniformDist(-1, 1), -1, 1)
        self._test_dist_min_max(rng, UniformDist(-5, 5), -5, 5)
        self._test_dist_min_max(rng, UniformDist(5, 15), 5, 15)
        self._test_dist_min_max(rng, UniformDist(-15, -5), -15, -5)

    def test_normal(self):
        """
        Tests NormalDist.
        """
        dist = NormalDist(0, 1)
        self.assertAlmostEqual(0.5, dist.cdf(0))
        for x in np.linspace(1, 10, num=25):
            self.assertAlmostEqual(dist.cdf(-x), 1.0 - dist.cdf(x))

    def test_truncate_normal(self):
        """
        Tests truncating NormalDist.
        """
        rng = np.random.default_rng()
        a = TruncateContinuousProbDist(NormalDist(0, 1), 0, 1)
        b = TruncateContinuousProbDist(NormalDist(-1, 3), -1, 1)
        c = TruncateContinuousProbDist(NormalDist(-1, 6), -3, 1)
        self._test_dist_min_max(rng, a, 0, 1)
        self._test_dist_min_max(rng, b, -1, 1)
        self._test_dist_min_max(rng, c, -3, 1)

    def test_linear_transform(self):
        """
        Tests linear transformations to distributions (e.g. shift and scale).
        """
        rng = np.random.default_rng()
        for low, high in [(0, 1), (2, 4), (-8, 0), (-5, 5)]:
            test_distributions = [
                UniformDist(low, high),
                NormalDist(0, 1).truncate(low, high)
            ]
            for dist in test_distributions:
                # Test Addition.
                self._test_dist_min_max(rng, dist + 1, low + 1, high + 1)
                self._test_dist_min_max(rng, 1 + dist, low + 1, high + 1)
                self._test_dist_min_max(rng, dist + 2.5, low + 2.5, high + 2.5)
                self._test_dist_min_max(rng, 2.5 + dist, low + 2.5, high + 2.5)
                self._test_equivalent(dist + 1, LinearTransformContinuousProbDist(dist, 1, 1))
                self._test_equivalent(1 + dist, LinearTransformContinuousProbDist(dist, 1, 1))
                self._test_equivalent(dist + 2.5, LinearTransformContinuousProbDist(dist, 2.5, 1))
                self._test_equivalent(2.5 + dist, LinearTransformContinuousProbDist(dist, 2.5, 1))

                # Test Subtraction.
                self._test_dist_min_max(rng, dist - 1, low - 1, high - 1)
                self._test_dist_min_max(rng, 1 - dist, 1 - high, 1 - low)
                self._test_dist_min_max(rng, dist - 2.5, low - 2.5, high - 2.5)
                self._test_dist_min_max(rng, 2.5 - dist, 2.5 - high, 2.5 - low)
                self._test_equivalent(dist - 1, LinearTransformContinuousProbDist(dist, -1, 1))
                self._test_equivalent(1 - dist, LinearTransformContinuousProbDist(dist, 1, -1))
                self._test_equivalent(dist - 2.5, LinearTransformContinuousProbDist(dist, -2.5, 1))
                self._test_equivalent(2.5 - dist, LinearTransformContinuousProbDist(dist, 2.5, -1))

                # Test Multiplication.
                self._test_dist_min_max(rng, dist * 2, low * 2, high * 2)
                self._test_dist_min_max(rng, 2 * dist, low * 2, high * 2)
                self._test_dist_min_max(rng, dist * -1, high * -1, low * -1)
                self._test_dist_min_max(rng, -1 * dist, high * -1, low * -1)
                self._test_equivalent(dist * 2, LinearTransformContinuousProbDist(dist, 0, 2))
                self._test_equivalent(2 * dist, LinearTransformContinuousProbDist(dist, 0, 2))
                self._test_equivalent(dist * -1, LinearTransformContinuousProbDist(dist, 0, -1))
                self._test_equivalent(-1 * dist, LinearTransformContinuousProbDist(dist, 0, -1))

                # Test Division.
                self._test_dist_min_max(rng, dist / 2, low / 2, high / 2)
                self._test_dist_min_max(rng, dist / -1, high / -1, low / -1)
                self._test_equivalent(dist / 2, LinearTransformContinuousProbDist(dist, 0, 0.5))
                self._test_equivalent(dist / -1, LinearTransformContinuousProbDist(dist, 0, -1))

                # Test Addition and Multiplication.
                self._test_dist_min_max(rng, 2 * (dist + 1), 2 * (low + 1), 2 * (high + 1))
                self._test_dist_min_max(rng, (1 + dist) * 2, 2 * (low + 1), 2 * (high + 1))
                self._test_dist_min_max(rng, 0.5 * (dist + 2.5), 0.5 * (low + 2.5), 0.5 * (high + 2.5))
                self._test_dist_min_max(rng, (2.5 + dist) * 0.5, 0.5 * (low + 2.5), 0.5 * (high + 2.5))
                self._test_equivalent(2 * (dist + 1), LinearTransformContinuousProbDist(dist, 2, 2))
                self._test_equivalent((1 + dist) * 2, LinearTransformContinuousProbDist(dist, 2, 2))
                self._test_equivalent(0.5 * (dist + 2.5), LinearTransformContinuousProbDist(dist, 1.25, 0.5))
                self._test_equivalent((2.5 + dist) * 0.5, LinearTransformContinuousProbDist(dist, 1.25, 0.5))

    def _test_equivalent(self, dist1: ContinuousProbDist, dist2: ContinuousProbDist):
        """
        Tests that the CDF and PDF functions of dist1 and dist2 give the same values.
        """
        minimum = dist1.inverse_cdf(0)
        maximum = dist1.inverse_cdf(1)
        self.assertAlmostEqual(minimum, dist2.inverse_cdf(0))
        self.assertAlmostEqual(maximum, dist2.inverse_cdf(1))

        values = np.linspace(minimum, maximum, num=100)
        pdf1 = dist1.pdf(values)
        pdf2 = dist2.pdf(values)
        if not np.allclose(pdf1, pdf2):
            print("_test_equivalent failure:\npdf1 = {}\npdf2 = {}".format(pdf1, pdf2), file=sys.stderr)
            self.fail("PDF mismatch for {} and {}".format(dist1, dist2))

        cdf1 = dist1.cdf(values)
        cdf2 = dist2.cdf(values)
        if not np.allclose(cdf1, cdf2):
            print("_test_equivalent failure:\ncdf1 = {}\ncdf2 = {}".format(cdf1, cdf2), file=sys.stderr)
            self.fail("CDF mismatch for {} and {}".format(dist1, dist2))

        self.assertTrue(np.allclose(dist1.pdf(values), dist2.pdf(values)))
        self.assertTrue(np.allclose(dist1.cdf(values), dist2.cdf(values)))

        self.assertAlmostEqual(0, dist1.cdf(minimum - 1))
        self.assertAlmostEqual(0, dist2.cdf(minimum - 1))
        self.assertAlmostEqual(1, dist1.cdf(maximum + 1))
        self.assertAlmostEqual(1, dist2.cdf(maximum + 1))

    def _test_dist_min_max(self, rng: np.random.Generator, dist: ContinuousProbDist, minimum: float, maximum: float):
        """
        Tests that sampled values fall between minimum and maximum.
        This makes a lot of assumptions about the distribution.
        """
        name = "{} evaluated from {:.1f} to {:.1f}".format(str(dist), minimum, maximum)

        # Check that the cdf is 0 below minimum, and 1 above maximum.
        self.assertAlmostEqual(0, dist.cdf(minimum), msg=name)
        self.assertAlmostEqual(0, dist.cdf(minimum - 0.1), msg=name)
        self.assertAlmostEqual(0, dist.cdf(minimum - 1), msg=name)
        self.assertAlmostEqual(1, dist.cdf(maximum), msg=name)
        self.assertAlmostEqual(1, dist.cdf(maximum + 0.1), msg=name)
        self.assertAlmostEqual(1, dist.cdf(maximum + 1), msg=name)

        # Check the probability comparisons between dist and scalars.
        self.assertAlmostEqual(0, dist < minimum, msg=name)
        self.assertAlmostEqual(0, dist < minimum - 0.1, msg=name)
        self.assertAlmostEqual(0, dist < minimum - 1, msg=name)
        self.assertAlmostEqual(1, dist >= minimum, msg=name)
        self.assertAlmostEqual(1, dist >= minimum - 0.1, msg=name)
        self.assertAlmostEqual(1, dist >= minimum - 1, msg=name)

        self.assertAlmostEqual(0, dist <= minimum, msg=name)
        self.assertAlmostEqual(0, dist <= minimum - 0.1, msg=name)
        self.assertAlmostEqual(0, dist <= minimum - 1, msg=name)
        self.assertAlmostEqual(1, dist > minimum, msg=name)
        self.assertAlmostEqual(1, dist > minimum - 0.1, msg=name)
        self.assertAlmostEqual(1, dist > minimum - 1, msg=name)

        self.assertAlmostEqual(0, dist > maximum, msg=name)
        self.assertAlmostEqual(0, dist > maximum + 0.1, msg=name)
        self.assertAlmostEqual(0, dist > maximum + 1, msg=name)
        self.assertAlmostEqual(1, dist <= maximum, msg=name)
        self.assertAlmostEqual(1, dist <= maximum + 0.1, msg=name)
        self.assertAlmostEqual(1, dist <= maximum + 1, msg=name)

        self.assertAlmostEqual(0, dist >= maximum, msg=name)
        self.assertAlmostEqual(0, dist >= maximum + 0.1, msg=name)
        self.assertAlmostEqual(0, dist >= maximum + 1, msg=name)
        self.assertAlmostEqual(1, dist < maximum, msg=name)
        self.assertAlmostEqual(1, dist < maximum + 0.1, msg=name)
        self.assertAlmostEqual(1, dist < maximum + 1, msg=name)

        # Check that inverse_cdf returns minimum for 0, and maximum for 1.
        self.assertAlmostEqual(minimum, dist.inverse_cdf(0), msg=name)
        self.assertAlmostEqual(maximum, dist.inverse_cdf(1), msg=name)

        # Check that the pdf is zero outside [minimum, maximum], and non-zero within.
        self.assertAlmostEqual(0, dist.pdf(minimum - 0.1), msg=name)
        self.assertAlmostEqual(0, dist.pdf(maximum + 0.1), msg=name)
        self.assertGreater(dist.pdf(minimum + 0.1), 0, msg=name)
        self.assertGreater(dist.pdf((minimum + maximum) / 2), 0, msg=name)
        self.assertGreater(dist.pdf(maximum - 0.1), 0, msg=name)

        # Check that sampling single values at a time all fall within the expected range.
        for i in range(100):
            self.assertTrue(minimum <= dist.sample(rng) <= maximum, msg=name)

        # Check that sampling many values at once all fall within the expected range.
        samples = dist.sample(rng, shape=(100,))
        self.assertTrue(np.all(samples >= minimum) and np.all(samples <= maximum), msg=name)
