import unittest
from tyche.language import *
from tyche.individuals import *


class TestSubIndividual(Individual):
    _y: float
    z: TycheConceptField

    def __init__(self, y: float, z: float):
        super().__init__()
        self._y = y
        self.z = z

    @concept()
    def y(self) -> float:
        return self._y


class TestIndividual(Individual):
    x: TycheConceptField
    role_var: TycheRoleField
    independent_role_var: TycheRoleField
    sub1: TestSubIndividual
    sub2: TestSubIndividual
    sub1Ratio: float = 0.4

    def __init__(self, x: float):
        super().__init__()
        self.x = x
        self.sub1 = TestSubIndividual(0.5, 0.5)
        self.sub2 = TestSubIndividual(0.2, 0.8)
        self.role_var = ExclusiveRoleDist()
        self.role_var.add(self.sub1)
        self.independent_role_var = IndependentRoleDist()
        self.independent_role_var.add(self.sub1, 0.7)
        self.independent_role_var.add(self.sub2, 0.7)

    @TycheRoleDecorator
    def sub(self) -> ExclusiveRoleDist:
        dist = ExclusiveRoleDist()
        dist.add(self.sub1, self.sub1Ratio)
        dist.add(self.sub2, 1 - self.sub1Ratio)
        return dist


class TestIndividuals(unittest.TestCase):
    def setUp(self):
        self.x = Concept('my_X')
        self.y = Concept('my_Y')
        self.z = Concept('my_Z')
        self.r = 'my_R'

    def tearDown(self):
        pass

    def test_expectation_exclusive(self):
        """
        Tests the marginal expectation operator on exclusive role distributions.
        """
        individual = TestIndividual(0.5)
        individual.sub1.z = 0
        individual.sub2.z = 1

        individual.role_var.clear()
        flip = Constant("flip", 0.5)

        # Vacuously true when the role is empty.
        self.assertAlmostEqual(1, individual.eval(Expectation("role_var", "z")))
        self.assertAlmostEqual(1, individual.eval(Expectation("role_var", "z", flip)))
        individual.role_var.add(None, 1)
        self.assertAlmostEqual(1, individual.eval(Expectation("role_var", "z")))
        self.assertAlmostEqual(1, individual.eval(Expectation("role_var", "z", flip)))

        # Expectations should apply an implicit 'given non-None'
        individual.role_var.add(individual.sub1, 1)
        self.assertAlmostEqual(0, individual.eval(Expectation("role_var", "z")))
        self.assertAlmostEqual(0, individual.eval(Expectation("role_var", "z", flip)))
        individual.role_var.add(individual.sub2, 1)
        self.assertAlmostEqual(0.5, individual.eval(Expectation("role_var", "z")))
        self.assertAlmostEqual(0.5, individual.eval(Expectation("role_var", "z", flip)))

        # While the weights of all the individuals in the role are equal,
        # E(z | z) should give sum(z^2) / sum(z), as E(z | z) = E(z and z) / E(z)
        try:
            z_given_z = Expectation("role_var", "z", "z")
            individual.sub1.z = 0.2
            individual.sub2.z = 0.8
            self.assertAlmostEqual(0.2**2 + 0.8**2, individual.eval(z_given_z))
            individual.sub1.z = 0.5
            individual.sub2.z = 0.7
            self.assertAlmostEqual((0.5**2 + 0.7**2) / (0.5 + 0.7), individual.eval(z_given_z))
        finally:
            individual.sub1.z = 0
            individual.sub2.z = 1

        # Individuals should override their own weight when added twice.
        individual.role_var.add(individual.sub2, 3)
        self.assertAlmostEqual(0.75, individual.eval(Expectation("role_var", "z")))

        # Removing the None-individual should not affect the expectation.
        individual.role_var.remove(None)
        self.assertAlmostEqual(0.75, individual.eval(Expectation("role_var", "z")))

        # Removing the individuals should affect the expectation.
        individual.role_var.remove(individual.sub1)
        self.assertAlmostEqual(1, individual.eval(Expectation("role_var", "z")))
        individual.role_var.add(individual.sub1, 1)
        individual.role_var.remove(individual.sub2)
        self.assertAlmostEqual(0, individual.eval(Expectation("role_var", "z")))
        individual.role_var.remove(individual.sub1)
        self.assertAlmostEqual(1, individual.eval(Expectation("role_var", "z")))

    def test_expectation_independent(self):
        """
        Tests the marginal expectation operator on independent role distributions.
        """
        individual = TestIndividual(0.5)
        individual.sub1.z = 0
        individual.sub2.z = 1

        individual.independent_role_var.clear()
        flip = Constant("flip", 0.5)

        # Vacuously true when the role is empty.
        self.assertAlmostEqual(1, individual.eval(Expectation("independent_role_var", "z")))
        self.assertAlmostEqual(1, individual.eval(Expectation("independent_role_var", "z", flip)))

        # Simple cases where the probability of sub1 and sub2 are 100%.
        individual.independent_role_var.add(individual.sub1, 1)
        self.assertAlmostEqual(0, individual.eval(Expectation("independent_role_var", "z")))
        self.assertAlmostEqual(0, individual.eval(Expectation("independent_role_var", "z", flip)))
        individual.independent_role_var.add(individual.sub2, 1)
        self.assertAlmostEqual(1, individual.eval(Expectation("independent_role_var", "z")))
        self.assertAlmostEqual(0.5 / (1 - 0.5**2), individual.eval(Expectation("independent_role_var", "z", flip)))

        # More complex test cases.
        try:
            z_given_z = Expectation("independent_role_var", "z", "z")
            individual.sub1.z = 0.2
            individual.sub2.z = 0.8
            self.assertAlmostEqual(
                (1 - (1 - 0.2**2) * (1 - 0.8**2)) / (1 - (1 - 0.2) * (1 - 0.8)),
                individual.eval(z_given_z)
            )
            individual.sub1.z = 0.5
            individual.sub2.z = 0.7
            self.assertAlmostEqual(
                (1 - (1 - 0.5**2) * (1 - 0.7**2)) / (1 - (1 - 0.5) * (1 - 0.7)),
                individual.eval(z_given_z)
            )
        finally:
            individual.sub1.z = 0
            individual.sub2.z = 1

        # Individuals should override their own probability when added twice.
        individual.independent_role_var.add(individual.sub2, 0.75)
        self.assertAlmostEqual(0.75, individual.eval(Expectation("independent_role_var", "z")))

        # Another, more complex, test-case.
        try:
            individual.sub1.z = 0.6
            individual.sub1._y = 0.85
            individual.sub2.z = 0.4
            individual.sub2._y = 0.75
            individual.independent_role_var.add(individual.sub1, 0.25)
            individual.independent_role_var.add(individual.sub2, 0.75)
            self.assertAlmostEqual(
                (1 - (1 - 0.25 * 0.6 * 0.85) * (1 - 0.75 * 0.4 * 0.75)) /
                (1 - (1 - 0.25 * 0.85) * (1 - 0.75 * 0.75)),
                individual.eval(Expectation("independent_role_var", "z", "y"))
            )
        finally:
            individual.sub1.z = 0
            individual.sub1._y = 0.5
            individual.sub2.z = 1
            individual.sub2._y = 0.2
            individual.independent_role_var.add(individual.sub1, 0.25)
            individual.independent_role_var.add(individual.sub2, 0.75)

        # Removing the individuals should affect the expectation.
        individual.independent_role_var.remove(individual.sub1)
        self.assertAlmostEqual(1, individual.eval(Expectation("independent_role_var", "z")))
        individual.independent_role_var.add(individual.sub1, 1)
        individual.independent_role_var.remove(individual.sub2)
        self.assertAlmostEqual(0, individual.eval(Expectation("independent_role_var", "z")))
        individual.independent_role_var.remove(individual.sub1)
        self.assertAlmostEqual(1, individual.eval(Expectation("independent_role_var", "z")))

    def test_exists(self):
        """
        Tests the role existence operator.
        """
        individual = TestIndividual(0.5)
        individual.sub1.z = 0
        individual.sub2.z = 1

        individual.role_var.clear()

        # The role does not exist when it is empty.
        self.assertAlmostEqual(0, individual.eval(Exists("role_var")))

        # The role does not exist if it only contains the None-individual.
        individual.role_var.add(None, 1)
        self.assertAlmostEqual(0, individual.eval(Exists("role_var")))

        # When individuals are added, the exists value should change.
        individual.role_var.add(individual.sub1, 1)
        self.assertAlmostEqual(0.5, individual.eval(Exists("role_var")))
        individual.role_var.add(individual.sub2, 1)
        self.assertAlmostEqual(2.0/3.0, individual.eval(Exists("role_var")))

        # Individuals should override their own weight when added twice.
        individual.role_var.add(individual.sub2, 3)
        self.assertAlmostEqual(0.8, individual.eval(Exists("role_var")))

        # Removing the None-individual should affect the exists value.
        individual.role_var.remove(None)
        self.assertAlmostEqual(1, individual.eval(Exists("role_var")))

        # Removing the individuals should affect the exists value.
        individual.role_var.add(None, 1)
        individual.role_var.remove(individual.sub1)
        self.assertAlmostEqual(0.75, individual.eval(Exists("role_var")))
        individual.role_var.add(individual.sub1, 1)
        individual.role_var.remove(individual.sub2)
        self.assertAlmostEqual(0.5, individual.eval(Exists("role_var")))
        individual.role_var.remove(individual.sub1)
        self.assertAlmostEqual(0, individual.eval(Exists("role_var")))

    def test_individuals(self):
        """
        Tests the evaluation of the probability of formulas.
        """
        self.assertEqual({"x"}, Individual.get_concept_names(TestIndividual))
        self.assertEqual({"sub", "role_var", "independent_role_var"}, Individual.get_role_names(TestIndividual))

        self.assertEqual({"y", "z"}, Individual.get_concept_names(TestSubIndividual))
        self.assertEqual(set(), Individual.get_role_names(TestSubIndividual))

        # Create an individual to test with.
        individual = TestIndividual(0.5)

        x = Concept("x")
        y = Concept("y")
        z = Concept("z")
        sub = Role("sub")

        # Test evaluating some concepts for the individual.
        self.assertEqual(1, individual.eval(ALWAYS))
        self.assertEqual(0, individual.eval(NEVER))
        self.assertAlmostEqual(0.5, individual.eval(x))
        self.assertAlmostEqual(0.4 * 0.5 + 0.6 * 0.2, individual.eval(Expectation("sub", "y")))
        self.assertAlmostEqual(0.4 * 0.5 + 0.6 * 0.8, individual.eval(Expectation("sub", "z")))
        self.assertAlmostEqual(0.4 * 0.5 + 0.6 * 0.2, individual.eval(Expectation(sub, y)))
        self.assertAlmostEqual(0.4 * 0.5 + 0.6 * 0.8, individual.eval(Expectation(sub, z)))
        self.assertAlmostEqual(0.4 * 0.5 + 0.6 * 0.2, individual.eval(Expectation("sub", y)))
        self.assertAlmostEqual(0.4 * 0.5 + 0.6 * 0.8, individual.eval(Expectation(sub, "z")))
        self.assertAlmostEqual(
            0.4 * (0.5 * 0.5) + 0.6 * (0.2 * 0.8),
            individual.eval(Expectation("sub", y & z))
        )

        # Test using a role variable.
        sub_role_concept = Expectation("role_var", y | z)
        self.assertAlmostEqual(1 - (1 - 0.5) * (1 - 0.5), individual.eval(sub_role_concept))

        # Test modifying a role variable.
        individual.role_var.add(individual.sub2)
        self.assertAlmostEqual(
            0.5 * (1 - (1 - 0.5) * (1 - 0.5)) + 0.5 * (1 - (1 - 0.2) * (1 - 0.8)),
            individual.eval(sub_role_concept)
        )

        # Test the mutability of the model.
        individual.x = 0.4
        individual.sub1Ratio = 0.1
        individual.sub2.z = 1

        self.assertAlmostEqual(0.4, individual.eval(x))
        self.assertAlmostEqual(0.1 * 0.5 + 0.9 * 0.2, individual.eval(Expectation("sub", "y")))
        self.assertAlmostEqual(0.1 * 0.5 + 0.9 * 1.0, individual.eval(Expectation("sub", "z")))
        self.assertAlmostEqual(0.1 * 0.5 + 0.9 * 0.2, individual.eval(Expectation(sub, y)))
        self.assertAlmostEqual(0.1 * 0.5 + 0.9 * 1.0, individual.eval(Expectation(sub, z)))
        self.assertAlmostEqual(0.1 * 0.5 + 0.9 * 0.2, individual.eval(Expectation("sub", y)))
        self.assertAlmostEqual(0.1 * 0.5 + 0.9 * 1.0, individual.eval(Expectation(sub, "z")))
        self.assertAlmostEqual(
            0.1 * (0.5 * 0.5) + 0.9 * (0.2 * 1),
            individual.eval(Expectation("sub", y & z))
        )
