"""
This file contains the examples from the 'Tyche: A library for
probabilistic reasoning and belief modelling in Python' paper.
"""
from tyche.distributions import *
from tyche.individuals import *
from tyche.language import *


class Person(Individual):
    positive: TycheConceptField
    conversed_with: TycheRoleField

    def __init__(self, positive: float, height_cm: NormalDist):
        super().__init__()
        self.positive = positive
        self.conversed_with = ExclusiveRoleDist()
        self.height_cm = height_cm

    @concept(symbol='tall')
    def is_tall(self):
        return self.height_cm > 180


class Student(Person):
    def __init__(self, good_grades: float):
        super().__init__(0.33, NormalDist(175, 6.5))
        self._good_grades = good_grades

    @concept()
    def good_grades(self):
        return self._good_grades

    @good_grades.learning_func(DirectLearningStrategy(0.5))
    def set_good_grades(self, good_grades: float):
        self._good_grades = good_grades
