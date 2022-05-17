"""
This file contains examples from the paper "Aleatoric Description Logic
for Probabilistic Reasoning" by Tim French and Thomas Smoker.
"""
from tyche.individuals import Individual, TycheConceptField, IdentityIndividual
from tyche.language import Atom, Concept


class VirusTransmissionIndividual(Individual):
    """
    An individual from the virus transmission scenario.
    """
    name: str
    has_virus: TycheConceptField
    has_fever: TycheConceptField

    def __init__(self, name: str, has_virus: float, has_fever: float):
        super().__init__(name)
        self.name = name
        self.has_virus = has_virus
        self.has_fever = has_fever


class VirusTransmissionScenario:
    """
    The model for the virus transmission scenario.
    """
    def __init__(self):
        self.h0 = VirusTransmissionIndividual("Hector_0", 0.0, 0.1)
        self.h1 = VirusTransmissionIndividual("Hector_1", 1.0, 0.6)
        self.h = IdentityIndividual(name="Hector", entries={self.h0: 0.1, self.h1: 0.9})

        self.i0 = VirusTransmissionIndividual("Igor_0", 0.0, 0.3)
        self.i1 = VirusTransmissionIndividual("Igor_1", 1.0, 0.8)
        self.i = IdentityIndividual(name="Igor", entries=[self.i0, self.i1])

        self.j0 = VirusTransmissionIndividual("Julia_0", 0.0, 0.2)
        self.j1 = VirusTransmissionIndividual("Julia_1", 1.0, 0.9)
        self.j = IdentityIndividual(name="Julia")
        self.j.add(self.j0, 0.3)
        self.j.add(self.j1, 0.7)

        self.individuals = [self.h, self.i, self.j]

    def evaluate_for_individuals(self, concept: Concept):
        """ Creates a map from individual names  """
        return {individual.name: concept.eval(individual) for individual in self.individuals}

    def __str__(self):
        """ Prints the current state of the model to the console. """
        return "\n".join([str(individual) for individual in self.individuals])


if __name__ == "__main__":
    model = VirusTransmissionScenario()
    print(": Original model")
    print(model)

    print()
    has_fever = Atom("has_fever")
    has_virus = Atom("has_virus")
    has_virus_or_no_fever = has_fever.complement() | has_virus

    def prob_dict_to_str(prob_dict: dict[str, float]):
        key_values = ", ".join([f"{name}: {prob:.3f}" for name, prob in prob_dict.items()])
        return f"{{{key_values}}}"

    print(": Probabilities")
    print(f"P({has_fever}) = {prob_dict_to_str(model.evaluate_for_individuals(has_fever))}")
    print(f"P({has_virus}) = {prob_dict_to_str(model.evaluate_for_individuals(has_virus))}")
    print(f"P({has_virus_or_no_fever}) = {prob_dict_to_str(model.evaluate_for_individuals(has_virus_or_no_fever))}")

    for concept in [has_fever, has_virus, has_virus_or_no_fever]:
        print()
        print(f": Reset model, and observe {concept}")
        model = VirusTransmissionScenario()
        model.h.observe(concept)
        model.i.observe(concept)
        model.j.observe(concept)
        print(model)

    print()
