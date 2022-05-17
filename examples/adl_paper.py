"""
This file contains examples from the paper "Aleatoric Description Logic
for Probabilistic Reasoning" by Tim French and Thomas Smoker.
"""
from tyche.individuals import Individual, TycheConceptField, TycheRoleField
from tyche.language import WeightedRoleDistribution, Atom, Concept, Expectation


class VirusTransmissionIndividual(Individual):
    """
    An individual from the virus transmission scenario.
    """
    name: str
    has_virus: TycheConceptField
    has_fever: TycheConceptField

    def __init__(self, name: str, has_virus: float, has_fever: float):
        super().__init__()
        self.name = name
        self.has_virus = has_virus
        self.has_fever = has_fever

    def __str__(self):
        return f"{self.name} (V={self.has_virus}, F={self.has_fever})"


class VirusTransmissionIdentityIndividual(Individual):
    """
    Represents the id role over an individual. In the future,
    this will be able to be represented implicitly, although
    for now it must be explicit.
    """
    name: str
    id: TycheRoleField

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.id = WeightedRoleDistribution()

    def __str__(self):
        individuals = ", ".join([f"{100 * prob:.1f}%: {ctx}" for ctx, prob in self.id])
        return f"{self.name} {{{individuals}}}"


class VirusTransmissionScenario:
    """
    The model for the virus transmission scenario.
    """
    def __init__(self):
        self.h0 = VirusTransmissionIndividual("Hector_0", 0.0, 0.1)
        self.h1 = VirusTransmissionIndividual("Hector_1", 1.0, 0.6)
        self.h = VirusTransmissionIdentityIndividual("Hector")
        self.h.id.add(self.h0, 0.1)
        self.h.id.add(self.h1, 0.9)

        self.i0 = VirusTransmissionIndividual("Igor_0", 0.0, 0.3)
        self.i1 = VirusTransmissionIndividual("Igor_1", 1.0, 0.8)
        self.i = VirusTransmissionIdentityIndividual("Igor")
        self.i.id.add(self.i0, 0.5)
        self.i.id.add(self.i1, 0.5)

        self.j0 = VirusTransmissionIndividual("Julia_0", 0.0, 0.2)
        self.j1 = VirusTransmissionIndividual("Julia_1", 1.0, 0.9)
        self.j = VirusTransmissionIdentityIndividual("Julia")
        self.j.id.add(self.j0, 0.3)
        self.j.id.add(self.j1, 0.7)

        self.individuals = [self.h, self.i, self.j]

    def evaluate_for_individuals(self, concept: Concept):
        """ Creates a map from individual names  """
        results = {}
        expectation = Expectation("id", concept)
        for id_individual in self.individuals:
            results[id_individual.name] = expectation.eval(id_individual)
        return results

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
    print(": Probabilities")
    print(f"P({has_fever}) = {model.evaluate_for_individuals(has_fever)}")
    print(f"P({has_virus}) = {model.evaluate_for_individuals(has_virus)}")
    print(f"P({has_virus_or_no_fever}) = {model.evaluate_for_individuals(has_virus_or_no_fever)}")

    print()
    print(f": Observe {has_fever}")
    model.h.id.apply_bayes_rule(has_fever)
    model.i.id.apply_bayes_rule(has_fever)
    model.j.id.apply_bayes_rule(has_fever)
    print(model)

    print()
    print(f": Observe {has_virus}")
    model.h.id.apply_bayes_rule(has_virus)
    model.i.id.apply_bayes_rule(has_virus)
    model.j.id.apply_bayes_rule(has_virus)
    print(model)

    print()
    model = VirusTransmissionScenario()
    print(f": Reset model, and observe {has_virus_or_no_fever}")
    model.h.id.apply_bayes_rule(has_virus_or_no_fever)
    model.i.id.apply_bayes_rule(has_virus_or_no_fever)
    model.j.id.apply_bayes_rule(has_virus_or_no_fever)
    print(model)
