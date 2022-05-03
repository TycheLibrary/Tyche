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

        dist = UniformDist(0, 1)
        self._test_dist_min_max(rng, dist, 0, 1)

    def test_transform(self):
        """
        Tests transformations to distributions (e.g. shift and scale).
        """
        rng = np.random.default_rng()
        for low, high in [(0, 1), (2, 4), (-8, -7), (-5, 5)]:
            dist = UniformDist(low, high)

            # Test Addition.
            self._test_dist_min_max(rng, dist + 1, low + 1, high + 1)
            self._test_dist_min_max(rng, 1 + dist, low + 1, high + 1)
            self._test_dist_min_max(rng, dist + 2.5, low + 2.5, high + 2.5)
            self._test_dist_min_max(rng, 2.5 + dist, low + 2.5, high + 2.5)

            # Test Subtraction.
            self._test_dist_min_max(rng, dist - 1, low - 1, high - 1)
            self._test_dist_min_max(rng, 1 - dist, 1 - high, 1 - low)
            self._test_dist_min_max(rng, dist - 2.5, low - 2.5, high - 2.5)
            self._test_dist_min_max(rng, 2.5 - dist, 2.5 - high, 2.5 - low)

            # Test Multiplication.
            self._test_dist_min_max(rng, dist * 2, low * 2, high * 2)
            self._test_dist_min_max(rng, 2 * dist, low * 2, high * 2)
            self._test_dist_min_max(rng, dist * -1, high * -1, low * -1)
            self._test_dist_min_max(rng, -1 * dist, high * -1, low * -1)

            # Test Division.
            self._test_dist_min_max(rng, dist / 2, low / 2, high / 2)
            self._test_dist_min_max(rng, dist / -1, high / -1, low / -1)

            # Test Addition and Multiplication.
            self._test_dist_min_max(rng, 2 * (dist + 1), 2 * (low + 1), 2 * (high + 1))
            self._test_dist_min_max(rng, (1 + dist) * 2, 2 * (low + 1), 2 * (high + 1))
            self._test_dist_min_max(rng, 0.5 * (dist + 2.5), 0.5 * (low + 2.5), 0.5 * (high + 2.5))
            self._test_dist_min_max(rng, (2.5 + dist) * 0.5, 0.5 * (low + 2.5), 0.5 * (high + 2.5))

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
