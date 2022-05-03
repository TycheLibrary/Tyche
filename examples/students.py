"""
Example python class for prototyping usage of the package
This example is for tracking student performance (grades)
and likely project supervisor.
"""
import tyche.language as language
import tyche.model as model
import tyche.logic as logic
from language.adl import * #....etc

class Person(Individual):

    def __init__(self, name=None, gender=None, age=None):
        self.name = StringField(name)
        self.gender = StringDist('male','female','other')
        self.age = UniformDist(0,120)

    @concept('adult')#adult added to concept listener
    def _adult(self):#use underscore by convention?
        return self.age.greater_than(17)

class Student(Person):

    def __init__(self, name=None, gender=None, age=None, gpa=50):
        super(self)
        self.gpa = NormalDist(50,10)
        self.supervisor = IndividualDist(supervisors) #initialise thes edistributions of dictionaries

    @concept('student')
    def _student(self):
        return 1.0

    @concept('pass')#concept functions return probabilities
    def _pass(self):
        return self.gpa.sample(lambda x : x>50)

    @concept('score_above<cut_off>')
    def _scoreabove(self, cut_off):
        return self.gpa.sample(lambda x : x>cut_off)

    def set_supervisor(supervisors, weight=1.0):
        self.supervisor.update(supervisors, weight)

    @role('supervisor')#supervisor functions return individual distributions
    def _supervisor(self):
        if self.supervisors is None:
            #have some link to possible supervisor sets
            return {Null: 1.0} #return the null individual
        else return self.supervisors

    @axiom('WWCC')#axioms return tuples of equivalent formula, or single formulas assumed to be equivalent to yes.
    def _WWCC(self):
        '''juvenile students can only be supervised by people with a WWCC'''
        return (Conditional(Concept('adult'), Yes, Expect('supervisor',Concept('wwcc'))), Yes)


class Supervisor(Person):

    def __init__(self, wwcc=0.0):
        #similar cases
        self.students = SetDistribution(Student.cls)#somethig like this, hack otgether manually
        self.wwcc = wwcc

    @concept('wwcc')#has a working with childrens check
    def _wwcc(self):
        return wwcc

clare = Student()
natasha = Supervisor(0.7)
adult = Concept('adult')# is a concept and a formula
#adult.lower_bound is an abbreviation for Conditional(condition=adult, if_yes=Yes, if_no= Concept('X')) where X is a new String
#adult.upper_bound is an abbreviation for Conditional(condition=adult, if_yes=Concept('X'), if_no=No) where X is a new String
supervisor_WWCC = Expect('supervisor', Concept('wwcc'))
supervisor = Role('supervisor') #supervisor is a role object wih functions is, is_not, is_given
clare.prob(adult) #is a probability
clare.supervisor #is a distribution of individuals
clare.prob(adult.when(supervisor.is_not(concept('wwcc'))).otherwise(Yes))   #(a?b:c) = b.when(a).otherwise(c)

student_book = Theory(Student)
student_book.asserts(clare, adult, 0.8) # use present tense imperitive language for formulas and theories?
student_book.consistent() #True/False

bb = BeliefBase(Person, Student, Supervisor)
bb.add(clare)
bb.add(natasha)
bb.prob(clare, adult.when(supervisor.is_not(concept('wwcc')))) #returns probability
hd = Concept('score_above<80>')
bb.observes(clare, hd) #we observe clare has an hd, and update our prior assumption on gpa based on this
bb.prob(clare, hd)
bb.updates(clare, hd) #present tense imperative??
