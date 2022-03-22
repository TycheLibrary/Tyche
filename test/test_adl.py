import unittest
from tyche import language as adl


class TestADL(unittest.TestCase):
    def setUp(self):
        self.x = adl.Atom('my_X')
        self.y = adl.Atom('my_Y')
        self.z = adl.Atom('my_Z')
        self.r = adl.Role('my_R')

    def tearDown(self):
        pass

    def test_equals(self):
        # Constants
        self.assertEqual(adl.always, adl.always)
        self.assertNotEqual(adl.always, adl.never)
        self.assertNotEqual(adl.always, self.y)

        # Atoms
        self.assertEqual(self.x, self.x)
        self.assertNotEqual(self.x, self.y)

        # Conditionals
        cond1 = adl.Conditional(self.x, self.y, self.z)
        cond2 = adl.Conditional(self.x, adl.always, adl.never)
        cond3 = adl.Conditional(self.x, self.y, self.z)
        self.assertEqual(cond1, cond3)
        self.assertNotEqual(cond1, cond2)
        self.assertNotEqual(cond1, adl.never)

        # Marginal Expectations
        marg1 = adl.Expectation(self.r, self.x)
        marg2 = adl.Expectation(self.r, self.y)
        marg3 = adl.Expectation(self.r, self.x)
        self.assertEqual(marg1, marg3)
        self.assertNotEqual(marg1, marg2)
        self.assertNotEqual(marg1, cond1)


if __name__ == '__main__':
    unittest.main()
