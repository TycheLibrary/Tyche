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
    role_var: TycheRole
    sub1: TestSubIndividual
    sub2: TestSubIndividual
    sub1Ratio: float = 0.4

    def __init__(self, x: float):
        super().__init__()
        self.x = x
        self.sub1 = TestSubIndividual(0.5, 0.5)
        self.sub2 = TestSubIndividual(0.2, 0.8)
        self.role_var = RoleDistribution()
        self.role_var.add(self.sub1)

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

    def test_expectation(self):
        """
        Tests the marginal expectation operator.
        """
        individual = TestIndividual(0.5)
        individual.sub1.z = 0
        individual.sub2.z = 1

        individual.role_var.clear()

        # Vacuously true when the role is empty.
        self.assertAlmostEqual(1, Expectation("role_var", "z").eval(individual))
        individual.role_var.add(None, 1)
        self.assertAlmostEqual(1, Expectation("role_var", "z").eval(individual))

        # Expectations should apply an implicit 'given non-None'
        individual.role_var.add(individual.sub1, 1)
        self.assertAlmostEqual(0, Expectation("role_var", "z").eval(individual))
        individual.role_var.add(individual.sub2, 1)
        self.assertAlmostEqual(0.5, Expectation("role_var", "z").eval(individual))

        # Individuals should override their own weight when added twice.
        individual.role_var.add(individual.sub2, 3)
        self.assertAlmostEqual(0.75, Expectation("role_var", "z").eval(individual))

        # Removing the None-individual should not affect the expectation.
        individual.role_var.remove(None)
        self.assertAlmostEqual(0.75, Expectation("role_var", "z").eval(individual))

        # Removing the individuals should affect the expectation.
        individual.role_var.remove(individual.sub1)
        self.assertAlmostEqual(1, Expectation("role_var", "z").eval(individual))
        individual.role_var.add(individual.sub1, 1)
        individual.role_var.remove(individual.sub2)
        self.assertAlmostEqual(0, Expectation("role_var", "z").eval(individual))
        individual.role_var.remove(individual.sub1)
        self.assertAlmostEqual(1, Expectation("role_var", "z").eval(individual))

    def test_individuals(self):
        """
        Tests the evaluation of the probability of formulas.
        """
        self.assertEqual({"x"}, Individual.get_concept_names(TestIndividual))
        self.assertEqual({"sub", "role_var"}, Individual.get_role_names(TestIndividual))

        self.assertEqual({"y", "z"}, Individual.get_concept_names(TestSubIndividual))
        self.assertEqual(set(), Individual.get_role_names(TestSubIndividual))

        # Create an individual to test with.
        individual = TestIndividual(0.5)

        x = Atom("x")
        y = Atom("y")
        z = Atom("z")
        sub = Role("sub")

        # Test evaluating some concepts for the individual.
        self.assertEqual(1, always.eval(individual))
        self.assertEqual(0, never.eval(individual))
        self.assertAlmostEqual(0.5, x.eval(individual))
        self.assertAlmostEqual(0.4 * 0.5 + 0.6 * 0.2, Expectation("sub", "y").eval(individual))
        self.assertAlmostEqual(0.4 * 0.5 + 0.6 * 0.8, Expectation("sub", "z").eval(individual))
        self.assertAlmostEqual(0.4 * 0.5 + 0.6 * 0.2, Expectation(sub, y).eval(individual))
        self.assertAlmostEqual(0.4 * 0.5 + 0.6 * 0.8, Expectation(sub, z).eval(individual))
        self.assertAlmostEqual(0.4 * 0.5 + 0.6 * 0.2, Expectation("sub", y).eval(individual))
        self.assertAlmostEqual(0.4 * 0.5 + 0.6 * 0.8, Expectation(sub, "z").eval(individual))
        self.assertAlmostEqual(
            0.4 * (0.5 * 0.5) + 0.6 * (0.2 * 0.8),
            Expectation("sub", y & z).eval(individual)
        )

        # Test using a role variable.
        sub_role_concept = Expectation("role_var", y | z)
        self.assertAlmostEqual(1 - (1 - 0.5) * (1 - 0.5), sub_role_concept.eval(individual))

        # Test modifying a role variable.
        individual.role_var.add(individual.sub2)
        self.assertAlmostEqual(
            0.5 * (1 - (1 - 0.5) * (1 - 0.5)) + 0.5 * (1 - (1 - 0.2) * (1 - 0.8)),
            sub_role_concept.eval(individual)
        )

        # Test the mutability of the model.
        individual.x = 0.4
        individual.sub1Ratio = 0.1
        individual.sub2.z = 1

        self.assertAlmostEqual(0.4, x.eval(individual))
        self.assertAlmostEqual(0.1 * 0.5 + 0.9 * 0.2, Expectation("sub", "y").eval(individual))
        self.assertAlmostEqual(0.1 * 0.5 + 0.9 * 1, Expectation("sub", "z").eval(individual))
        self.assertAlmostEqual(0.1 * 0.5 + 0.9 * 0.2, Expectation(sub, y).eval(individual))
        self.assertAlmostEqual(0.1 * 0.5 + 0.9 * 1, Expectation(sub, z).eval(individual))
        self.assertAlmostEqual(0.1 * 0.5 + 0.9 * 0.2, Expectation("sub", y).eval(individual))
        self.assertAlmostEqual(0.1 * 0.5 + 0.9 * 1, Expectation(sub, "z").eval(individual))
        self.assertAlmostEqual(
            0.1 * (0.5 * 0.5) + 0.9 * (0.2 * 1),
            Expectation("sub", y & z).eval(individual)
        )
