import unittest
from tyche.language import *
from tyche.individuals import *


class TestSubIndividual(Individual):
    _y: float
    z: TycheConcept

    def __init__(self, y: float, z: float):
        super().__init__()
        self._y = y
        self.z = z

    @concept
    def y(self) -> float:
        return self._y


class TestIndividual(Individual):
    x: TycheConcept
    sub1: TestSubIndividual
    sub2: TestSubIndividual
    sub1Ratio: float = 0.4

    def __init__(self, x: float):
        super().__init__()
        self.x = x
        self.sub1 = TestSubIndividual(0.5, 0.5)
        self.sub2 = TestSubIndividual(0.2, 0.8)

    @role
    def sub(self) -> RoleDistribution:
        dist = RoleDistribution()
        dist.add(self.sub1, self.sub1Ratio)
        dist.add(self.sub2, 1 - self.sub1Ratio)
        return dist


class TestIndividuals(unittest.TestCase):
    def setUp(self):
        self.x = Atom('my_X')
        self.y = Atom('my_Y')
        self.z = Atom('my_Z')
        self.r = 'my_R'

    def tearDown(self):
        pass

    def test_individuals(self):
        """
        Tests the evaluation of the probability of formulas.
        """
        self.assertEqual({"x"}, Individual.get_concept_names(TestIndividual))
        self.assertEqual({"sub"}, Individual.get_role_names(TestIndividual))

        self.assertEqual({"y", "z"}, Individual.get_concept_names(TestSubIndividual))
        self.assertEqual(set(), Individual.get_role_names(TestSubIndividual))

        individual = TestIndividual(0.5)

        x = Atom("x")
        y = Atom("y")
        z = Atom("z")

        self.assertEqual(1, always.eval(individual))
        self.assertEqual(0, never.eval(individual))
        self.assertAlmostEqual(0.5, x.eval(individual))
        self.assertAlmostEqual(0.4 * 0.5 + 0.6 * 0.2, Expectation("sub", y).eval(individual))
        self.assertAlmostEqual(0.4 * 0.5 + 0.6 * 0.8, Expectation("sub", z).eval(individual))
        self.assertAlmostEqual(
            0.4 * (0.5 * 0.5) + 0.6 * (0.2 * 0.8),
            Expectation("sub", y & z).eval(individual)
        )

        # Test the mutability of the model.
        individual.x = 0.4
        individual.sub1Ratio = 0.1
        individual.sub2.z = 1

        self.assertAlmostEqual(0.4, x.eval(individual))
        self.assertAlmostEqual(0.1 * 0.5 + 0.9 * 0.2, Expectation("sub", y).eval(individual))
        self.assertAlmostEqual(0.1 * 0.5 + 0.9 * 1, Expectation("sub", z).eval(individual))
        self.assertAlmostEqual(
            0.1 * (0.5 * 0.5) + 0.9 * (0.2 * 1),
            Expectation("sub", y & z).eval(individual)
        )
