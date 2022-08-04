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

    def __init__(self, name: str = None):
        super().__init__(name=name)
        self.age = UniformDist(0, 120)
        self.height_cm = NormalDist(170.8, 7)
        self.tall_cutoff = 6 * 30.48  # 6 feet in cm
        self.is_male = False

    @concept
    def adult(self) -> float:
        return self.age >= 18

    @concept(symbol="tall")
    def is_tall(self) -> float:
        return self.height_cm > self.tall_cutoff

    @is_tall.learning_func(StatisticalConceptLearningStrategy(initial_value_weight=1))
    def is_tall(self, prob: float):
        # Move the mean of the height distribution to a point where is_tall would return prob.
        # This relies on the tall cutoff point and the standard deviation of height being fixed.
        current_mean = self.height_cm.mean()
        std_dev = self.height_cm.std_dev()
        new_z_score = (self.height_cm.inverse_cdf(prob) - current_mean) / std_dev
        new_mean = self.tall_cutoff + new_z_score * std_dev
        self.height_cm = NormalDist(new_mean, std_dev)


class Student(Person):
    supervisor: TycheRoleField

    def __init__(self, name: str = None, gender=None, age=None, gpa=50):
        super().__init__(name=name)
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
    def __init__(self, name: str, wwcc: float = 0.0):
        super().__init__(name=name)
        # self.students = SetDistribution(Student.cls)#somethig like this, hack otgether manually
        self.age = self.age.truncate(21, 120)
        self._wwcc = wwcc

    @concept
    def student(self) -> float:
        return 0.0

    @concept(symbol="wwcc")
    def has_working_with_children_check(self):
        """ Whether the supervisor has a working with children check. """
        return self._wwcc


if __name__ == "__main__":
    print()
    print(Individual.describe(Person))
    print(Individual.describe(Student))
    print(Individual.describe(Supervisor))

    clare = Student("Clare")
    natasha = Supervisor("Natasha", 0.7)
    tim = Supervisor("Tim", 0.85)
    clare.supervisor.add(natasha, 2)
    clare.supervisor.add(tim, 0.5)
    clare.supervisor.add(None, 1)

    print()
    print("Model:")
    print(f"  {clare}")
    print(f"  {natasha}")
    print(f"  {tim}")
    print()

    # adult.lower_bound is an abbreviation for
    #     Conditional(condition=adult, if_yes=Yes, if_no= Concept('X')) where X is a new String
    # adult.upper_bound is an abbreviation for
    #     Conditional(condition=adult, if_yes=Concept('X'), if_no=No) where X is a new String

    supervisor_WWCC = Exists("supervisor") & Expectation("supervisor", "wwcc")
    tall_and_adult = Concept("adult") & Concept("tall")
    tall_or_adult = Concept("adult") | Concept("tall")
    supervisor_tall_adult = Exists("supervisor") & Expectation("supervisor", tall_and_adult)
    supervisor = Role('supervisor')  # supervisor is a role object wih functions is, is_not, is_given

    print()
    print("P(clare is male) = {:.3f}".format(clare.eval("is_male")))
    print("P(clare is an adult) = {:.3f}".format(clare.eval("adult")))
    print("P(clare passed) = {:.3f}".format(clare.eval("passed")))
    print("P(clare is tall) = {:.3f}".format(clare.eval("tall")))
    print()
    print("clare is a tall adult = {}".format(tall_and_adult))
    print("P(clare is a tall adult) = {:.3f}".format(clare.eval(tall_and_adult)))
    print(f".. Clare's current height mean = {clare.height_cm.mean():.1f} cm, P(tall) = {clare.eval('tall'):.3f}")
    print(f".. Observe that Clare is a tall adult, {tall_and_adult}")
    clare.observe(tall_and_adult)
    print(f".. Clare's new height mean = {clare.height_cm.mean():.1f} cm, P(tall) = {clare.eval('tall'):.3f}")
    print(f".. Observe that Clare is not a tall adult, {tall_and_adult.complement()}")
    clare.observe(tall_and_adult.complement())
    print(f".. Clare's new height mean = {clare.height_cm.mean():.1f} cm, P(tall) = {clare.eval('tall'):.3f}")
    print("P(clare is a tall adult) = {:.3f}".format(clare.eval(tall_and_adult)))
    print()
    print("clare's supervisor is a tall adult = {}".format(supervisor_tall_adult))
    print("P(clare's supervisor is a tall adult) = {:.3f}".format(clare.eval(supervisor_tall_adult)))
    print()
    print("clare's supervisor has a WWCC = {}".format(supervisor_WWCC))
    print("P(clare's supervisor has a WWCC) = {:.3f}".format(clare.eval(supervisor_WWCC)))
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
