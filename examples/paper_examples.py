"""
This file contains the examples from the "Tyche: A library for
probabilistic reasoning and belief modelling in Python" paper.
"""
from tyche.distributions import *
from tyche.individuals import *
from tyche.language import *


class Person(Individual):
    positive: TycheConceptField
    conversed_with: TycheRoleField

    def __init__(self, name: str, positive: float, height_cm: NormalDist):
        super().__init__(name)
        self.positive = positive
        self.conversed_with = ExclusiveRoleDist()
        self.height_cm = height_cm

    @concept(symbol='tall')
    def is_tall(self):
        return self.height_cm > 180


class Student(Person):
    def __init__(self, name: str, good_grades: float):
        super().__init__(name, 0.33, NormalDist(175, 6.5))
        self._good_grades = good_grades

    @concept()
    def good_grades(self):
        return self._good_grades

    @good_grades.learning_func(DirectConceptLearningStrategy())
    def set_good_grades(self, good_grades: float):
        self._good_grades = good_grades


# The examples here use the models from the paper
# for some example operations.
if __name__ == "__main__":
    # Print out how Tyche interpreted our model.
    print(Individual.describe(Person))
    print(Individual.describe(Student))
    print()

    # Construct a good student.
    good_student = Student("Good", 0.8)
    print(f"Example Good Student = {good_student}")

    # Query the chance of them getting good grades, or being tall and positive.
    query_sentence = Concept("good_grades") | (Concept("tall") & Concept("positive"))
    print(f"- P {query_sentence} = {good_student.eval(query_sentence):.3f}")
    print()

    # Construct a couple bad students.
    # The first is very likely to be tall.
    bad_student1 = Student("Bad1", 0.2)
    bad_student1.height_cm = NormalDist(190, 3.5)
    print("Bad Students:")
    print(f"1) {bad_student1}")

    # The second is always positive.
    bad_student2 = Student("Bad2", 0.25)
    bad_student2.positive = 1.0
    print(f"2) {bad_student2}")
    print()

    # The bad students are more likely to converse with
    # one another, instead of the good student.
    bad_student1.conversed_with.add(good_student, 0.1)
    bad_student2.conversed_with.add(good_student, 0.1)
    bad_student1.conversed_with.add(bad_student2, 0.9)
    bad_student2.conversed_with.add(bad_student1, 0.9)

    # The good student is more likely to converse with no one.
    good_student.conversed_with.add(bad_student1, 0.1)
    good_student.conversed_with.add(bad_student2, 0.1)
    good_student.conversed_with.add(None, 0.8)

    # Query the chance that the students talked with
    # another student who got good grades.
    query_sentence = Exists("conversed_with") & Expectation("conversed_with", "good_grades")
    print(f"P ({query_sentence})")
    print(f"- Bad Student 1 = {bad_student1.eval(query_sentence):.3f}")
    print(f"- Bad Student 2 = {bad_student2.eval(query_sentence):.3f}")
    print(f"- Good Student = {good_student.eval(query_sentence):.3f}")
    print()

    # Observe that the good student conversed with someone who got
    # good grades. They only could have conversed with the students
    # that we thought got bad grades. Therefore, it is more likely
    # that those students get good grades more often.
    observe_sentence = Expectation("conversed_with", "good_grades")
    print("Good Student:")
    print(f"- Observe {observe_sentence}")
    good_student.observe(observe_sentence)
    print()
    print("New Students:")
    print(f"- Bad Student 1 = {bad_student1}")
    print(f"- Bad Student 2 = {bad_student2}")
    print(f"- Good Student = {good_student}")
    print()


