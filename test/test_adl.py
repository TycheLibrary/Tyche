import unittest
from tyche import adl

class TestADL(unittest.TestCase):

    def setUp(self):
        self.x = adl.atom('my_X')
        self.y = adl.atom('my_Y')
        self.z = adl.atom('my_Z')
        self.r = adl.Role('my_R')

    def tearDown(self):
        pass

    def test_equals(self):
        #Constants
        self.assertEqual(adl.yes,adl.yes)
        self.assertNotEqual(adl.yes,adl.no)
        self.assertNotEqual(adl.yes,self.y)
        #Atoms
        self.assertEqual(self.x,self.x)
        self.assertNotEqual(self.x,self.y)
        #Conditionals
        cond1 = adl.conditional(self.x,self.y,self.z)
        cond2 = adl.conditional(self.x,adl.yes,adl.no)
        cond3 = adl.conditional(self.x,self.y,self.z)
        self.assertEqual(cond1,cond3)
        self.assertNotEqual(cond1,cond2)
        self.assertNotEqual(cond1,adl.no)
        #Marginals
        marg1 = adl.marginal(self.r,self.x,self.y)
        marg2 = adl.marginal(self.r,self.y,self.x)
        marg3 = adl.marginal(self.r,self.x,self.y)
        self.assertEqual(marg1,marg3)
        self.assertNotEqual(marg1,marg2)
        self.assertNotEqual(marg1,cond1)

if __name__ == '__main__':
    unittest.main()
