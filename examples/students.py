"""
Example python class for prototyping usage of the package
This example is for tracking student performance (grades)
and likely project supervisor.
"""
from tyche.individuals import *
from tyche.distributions import *
from tyche.language import *


class Person(Individual):
    is_male: TycheConceptField

    def __init__(self, name=None, gender=None, age=None):
        super().__init__()
        # self.name = StringField(name)
        # self.gender = StringDist('male','female','other')
        self.age = UniformDist(0, 120)
        self.height_cm = NormalDist(170.8, 7).truncate(10, 272)
        self.is_male = False

    @property
    def height_ft(self):
        """ Returns the distribution of height in feet. """
        return self.height_cm * 0.0328084

    @concept
    def adult(self) -> float:
        return self.age >= 18

    @concept
    def tall(self) -> float:
        """ We consider anyone over 6 feet to be tall. """
        return self.height_ft > 6


class Student(Person):
    supervisor: TycheRoleField

    def __init__(self, name=None, gender=None, age=None, gpa=50):
        super().__init__()
        self.gpa = NormalDist(60, 10).truncate(0, 100)
        self.supervisor = ExclusiveRoleDist()

    @concept
    def student(self) -> float:
        return 1.0

    @concept
    def passed(self):
        return self.gpa > 50

    # @concept('score_above<cut_off>')
    # def _scoreabove(self, cut_off):
    #     return self.gpa.sample(lambda x : x>cut_off)

    # @axiom('WWCC')#axioms return tuples of equivalent formula, or single formulas assumed to be equivalent to yes.
    # def _WWCC(self):
    #     '''juvenile students can only be supervised by people with a WWCC'''
    #     return (Conditional(Concept('adult'), Yes, Expect('supervisor',Concept('wwcc'))), Yes)


class Supervisor(Person):
    def __init__(self, wwcc=0.0):
        super().__init__()
        # self.students = SetDistribution(Student.cls)#somethig like this, hack otgether manually
        self.age = self.age.truncate(21, 120)
        self._wwcc = wwcc

    @concept
    def wwcc(self):
        """ Whether the supervisor has a working with childrens check. """
        return self._wwcc


clare = Student()
natasha = Supervisor(0.7)
tim = Supervisor(0.85)
clare.supervisor.add(natasha, 2)
clare.supervisor.add(tim, 0.5)
clare.supervisor.add(None, 1)

adult = Atom('adult')

# adult.lower_bound is an abbreviation for
#     Conditional(condition=adult, if_yes=Yes, if_no= Concept('X')) where X is a new String
# adult.upper_bound is an abbreviation for
#     Conditional(condition=adult, if_yes=Concept('X'), if_no=No) where X is a new String

supervisor_WWCC = Exists("supervisor") & Expectation("supervisor", "wwcc")
tall_adult = adult & Atom("tall")
supervisor_tall_adult = Exists("supervisor") & Expectation("supervisor", tall_adult)
supervisor = Role('supervisor')  # supervisor is a role object wih functions is, is_not, is_given

print()
print("P(clare is male) = {:.3f}".format(Atom("is_male").eval(clare)))
print("P(clare is an adult) = {:.3f}".format(adult.eval(clare)))
print("P(clare passed) = {:.3f}".format(Atom("passed").eval(clare)))
print("P(clare is tall) = {:.3f}".format(Atom("tall").eval(clare)))
print()
print("clare is a tall adult = {}".format(tall_adult))
print("P(clare is a tall adult) = {:.3f}".format(tall_adult.eval(clare)))
print()
print("clare's supervisor is a tall adult = {}".format(supervisor_tall_adult))
print("P(clare's supervisor is a tall adult) = {:.3f}".format(supervisor_tall_adult.eval(clare)))
print()
print("clare's supervisor has a WWCC = {}".format(supervisor_WWCC))
print("P(clare's supervisor has a WWCC) = {:.3f}".format(supervisor_WWCC.eval(clare)))
print()

# clare.supervisor #is a distribution of individuals
# clare.prob(adult.when(supervisor.is_not(concept('wwcc'))).otherwise(Yes))   #(a?b:c) = b.when(a).otherwise(c)

# student_book = Theory(Student)
# student_book.asserts(clare, adult, 0.8) # use present tense imperitive language for formulas and theories?
# student_book.consistent() #True/False

# bb = BeliefBase(Person, Student, Supervisor)
# bb.add(clare)
# bb.add(natasha)
# bb.prob(clare, adult.when(supervisor.is_not(concept('wwcc')))) #returns probability
# hd = Concept('score_above<80>')
# bb.observes(clare, hd) #we observe clare has an hd, and update our prior assumption on gpa based on this
# bb.prob(clare, hd)
# bb.updates(clare, hd) #present tense imperative??
