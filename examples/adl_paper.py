"""
This file contains examples from the paper "Aleatoric Description Logic
for Probabilistic Reasoning" by Tim French and Thomas Smoker.
"""
from tyche.individuals import Individual, TycheConcept, TycheRole
from tyche.language import WeightedRoleDistribution, Atom


class VirusTransmissionIndividual(Individual):
    """
    An individual from the virus transmission scenario.
    """
    name: str
    has_virus: TycheConcept
    has_fever: TycheConcept

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
    id: TycheRole

    def __init__(self):
        super().__init__()
        self.id = WeightedRoleDistribution()

    def __str__(self):
        individuals = "\n - ".join([f"{100 * prob:.1f}% for {ctx}" for ctx, prob in self.id])
        return f"id:\n - {individuals}"


class VirusTransmissionScenario:
    """
    The model for the virus transmission scenario.
    """
    def __init__(self):
        self.h0 = VirusTransmissionIndividual("Hector_0", 0.0, 0.1)
        self.h1 = VirusTransmissionIndividual("Hector_1", 1.0, 0.6)
        self.h = VirusTransmissionIdentityIndividual()
        self.h.id.add(self.h0, 0.1)
        self.h.id.add(self.h1, 0.9)

        self.i0 = VirusTransmissionIndividual("Igor_0", 0.0, 0.3)
        self.i1 = VirusTransmissionIndividual("Igor_1", 1.0, 0.8)
        self.i = VirusTransmissionIdentityIndividual()
        self.i.id.add(self.i0, 0.5)
        self.i.id.add(self.i1, 0.5)

        self.j0 = VirusTransmissionIndividual("Julia_0", 0.0, 0.2)
        self.j1 = VirusTransmissionIndividual("Julia_1", 1.0, 0.9)
        self.j = VirusTransmissionIdentityIndividual()
        self.j.id.add(self.j0, 0.3)
        self.j.id.add(self.j1, 0.7)

    def __str__(self):
        """ Prints the current state of the model to the console. """
        return "\n".join([str(individual) for individual in [self.h, self.i, self.j]])


if __name__ == "__main__":
    model = VirusTransmissionScenario()
    print(model)

    print()
    print("Observe Hector, Igor, and Julia with a fever,")
    has_fever = Atom("has_fever")
    print(f" .. Observing {has_fever} at Hector, Igor, and Julia")
    model.h.id.apply_bayes_rule(has_fever)
    model.i.id.apply_bayes_rule(has_fever)
    model.j.id.apply_bayes_rule(has_fever)
    print(model)

    print()
    model = VirusTransmissionScenario()
    print("Reset model, and observe Hector, Igor, and Julia not having a fever or having the virus,")
    has_virus = Atom("has_virus")
    has_virus_or_no_fever = has_fever.complement() | has_virus
    print(f" .. Observing {has_virus_or_no_fever} at Hector, Igor, and Julia")
    model.h.id.apply_bayes_rule(has_virus_or_no_fever)
    model.i.id.apply_bayes_rule(has_virus_or_no_fever)
    model.j.id.apply_bayes_rule(has_virus_or_no_fever)
    print(model)
