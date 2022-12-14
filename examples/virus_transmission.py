"""
This file contains the virus transmission example from the paper
"Aleatoric Description Logic for Probabilistic Reasoning" by
Tim French and Thomas Smoker.
"""
from tyche.individuals import Individual, TycheConceptField, IdentityIndividual, role, \
    BayesRuleLearningStrategy
from tyche.language import Concept, ADLNode, ExclusiveRoleDist, Expectation
from tyche.probability import TycheProbabilityException


class VirusTransmissionIndividual(Individual):
    """
    An individual from the virus transmission scenario.
    """
    name: str
    has_virus: TycheConceptField
    has_fever: TycheConceptField

    def __init__(self, name: str, has_virus: float, has_fever: float, contact: ExclusiveRoleDist):
        super().__init__(name)
        self.name = name
        self.has_virus = has_virus
        self.has_fever = has_fever
        self._contact = contact

    @role()
    def contact(self):
        return self._contact

    @contact.learning_func(BayesRuleLearningStrategy())
    def set_contact(self, dist: ExclusiveRoleDist):
        self._contact = dist


class VirusTransmissionScenario:
    """
    The model for the virus transmission scenario.
    """
    def __init__(self):
        h_contact = ExclusiveRoleDist()
        self.h0 = VirusTransmissionIndividual("Hector_0", 0.0, 0.1, h_contact)
        self.h1 = VirusTransmissionIndividual("Hector_1", 1.0, 0.6, h_contact)
        self.hector = IdentityIndividual(name="Hector", entries={self.h0: 0.9, self.h1: 0.1})

        i_contact = ExclusiveRoleDist()
        self.i0 = VirusTransmissionIndividual("Igor_0", 0.0, 0.3, i_contact)
        self.i1 = VirusTransmissionIndividual("Igor_1", 1.0, 0.8, i_contact)
        self.igor = IdentityIndividual(ExclusiveRoleDist([self.i0, self.i1]), name="Igor")

        j_contact = ExclusiveRoleDist()
        self.j0 = VirusTransmissionIndividual("Julia_0", 0.0, 0.2, j_contact)
        self.j1 = VirusTransmissionIndividual("Julia_1", 1.0, 0.9, j_contact)
        self.julia = IdentityIndividual(name="Julia")
        self.julia.add(self.j0, 0.3)
        self.julia.add(self.j1, 0.7)

        h_contact.add(self.igor, 0.3)
        h_contact.add(self.julia, 0.7)

        i_contact.add(self.hector, 0.4)
        i_contact.add(self.julia, 0.6)

        j_contact.add(self.hector, 0.4)
        j_contact.add(self.igor, 0.6)

        self.individuals = [self.hector, self.igor, self.julia]

    def evaluate_for_individuals(self, concept: ADLNode):
        """ Creates a map from individual names  """
        return {individual.name: individual.eval(concept) for individual in self.individuals}

    def __str__(self):
        return self.to_str()

    def to_str(self, *, detail_lvl: int = 1, indent_lvl: int = 0):
        return "\n".join([
            ctx.to_str(detail_lvl=detail_lvl, indent_lvl=indent_lvl) for ctx in self.individuals
        ])


if __name__ == "__main__":
    model = VirusTransmissionScenario()
    print(": Original model")
    print(model.to_str(detail_lvl=3, indent_lvl=1))

    print()
    has_fever = Concept("has_fever")
    has_virus = Concept("has_virus")
    has_virus_or_no_fever = has_fever.complement() | has_virus

    def prob_dict_to_str(prob_dict: dict[str, float]):
        key_values = ", ".join([f"{name}: {prob:.3f}" for name, prob in prob_dict.items()])
        return f"{{{key_values}}}"

    print(": Basic Probabilities")
    print(f"E({has_fever}) = {prob_dict_to_str(model.evaluate_for_individuals(has_fever))}")
    print(f"E({has_virus}) = {prob_dict_to_str(model.evaluate_for_individuals(has_virus))}")
    print(f"E({has_virus_or_no_fever}) = {prob_dict_to_str(model.evaluate_for_individuals(has_virus_or_no_fever))}")

    print()
    print(": Example 1 - Chance that Hector was newly exposed to the virus after contact with someone with a fever")
    no_fever = has_virus.complement()
    contact_with_virus = Expectation("contact", has_virus, has_fever)
    exposed = no_fever & contact_with_virus
    print(f"E_hector({no_fever}) = {model.hector.eval(no_fever):.3f}")
    print(f"E_hector({contact_with_virus}) = {model.hector.eval(contact_with_virus):.3f}")
    print(f"E_hector({exposed}) = {model.hector.eval(exposed):.3f}")

    for concept in [has_fever, has_virus, has_virus_or_no_fever]:
        print()
        print(f": Reset model, and observe {concept}")
        model = VirusTransmissionScenario()
        model.hector.observe(concept)
        model.igor.observe(concept)
        model.julia.observe(concept)
        print(model)

    print()
    print(f": Reset model, and observe {exposed} at Hector")
    model = VirusTransmissionScenario()
    model.hector.observe(exposed)
    print(model.to_str(detail_lvl=3, indent_lvl=1))

    print()
    print(f": Observe {exposed} at Igor")
    model.igor.observe(exposed)
    print(model.to_str(detail_lvl=3, indent_lvl=1))

    try:
        print()
        print(f": Observe {exposed} at Julia")
        model.julia.observe(exposed)
        print(model.to_str(detail_lvl=3, indent_lvl=1))
    except TycheProbabilityException as e:
        if "impossible" in str(e):
            print(
                "- Observation at Julia failed as expected, as it would be impossible after "
                "Hector and Igor were found to definitely not have the virus"
            )
        else:
            raise e

    print()
    not_exposed = exposed.complement()
    print(f": Reset model, and observe {not_exposed} at Hector")
    model = VirusTransmissionScenario()
    model.hector.observe(not_exposed)
    print(model.to_str(detail_lvl=3, indent_lvl=1))
