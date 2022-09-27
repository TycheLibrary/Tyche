"""
This file contains a base-class for the Tyche and ProbLog
models, to allow them to be tested using the same code.
"""
import numpy as np


class Message:
    """
    A message sent between people.
    """
    def __init__(self, uses_emoji: bool, capitalises_first_word: bool, is_positive: bool):
        self.uses_emoji = uses_emoji
        self.capitalises_first_word = capitalises_first_word
        self.is_positive = is_positive


class ModelPerson:
    """
    A person that has a set of preferences when writing messages.
    """
    def __init__(
            self, name: str,
            uses_emoji: float = 0.5,
            capitalises_first_word: float = 0.5,
            is_positive: float = 0.5):

        self.name = name
        self.conversed_with: dict[str, float] = {}
        self.uses_emoji = uses_emoji
        self.capitalises_first_word = capitalises_first_word
        self.is_positive = is_positive

    def copy(self) -> 'ModelPerson':
        person = ModelPerson(self.name, self.uses_emoji, self.capitalises_first_word, self.is_positive)
        for key, value in self.conversed_with.items():
            person.conversed_with[key] = value
        return person

    def sample_message(self, rng: np.random.Generator) -> Message:
        """ Randomly samples this individual to a Person with known properties. """
        uses_emoji = rng.uniform(0, 1) < self.uses_emoji
        capitalises_first_word = rng.uniform(0, 1) < self.capitalises_first_word
        is_positive = rng.uniform(0, 1) < self.is_positive
        return Message(uses_emoji, capitalises_first_word, is_positive)

    def sample_conversed_with(self, rng: np.random.Generator) -> str:
        """
        Selects a random individual from this role based upon their weights.
        """
        person_weights = [(name, weight) for name, weight in self.conversed_with.items()]
        total_weight = 0
        for _, weight in person_weights:
            total_weight += weight

        target_cumulative_weight = rng.uniform(0, total_weight)
        cumulative_weight = 0
        for name, weight in person_weights:
            cumulative_weight += weight
            if cumulative_weight >= target_cumulative_weight:
                return name

        return person_weights[-1][0]


class Model:
    """
    A set of people that have preferences when writing messages.
    """
    def __init__(
            self,
            bob: ModelPerson,
            alice: ModelPerson,
            jeff: ModelPerson):

        self.bob = bob
        self.alice = alice
        self.jeff = jeff
        self.all = [bob, alice, jeff]
        self.by_name = {p.name: p for p in self.all}

    def copy(self) -> 'Model':
        return Model(self.bob.copy(), self.alice.copy(), self.jeff.copy())


class AnonymousMessagesImplementation:
    """
    A base-class to be implemented by the different
    implementations for the anonymous messages example.
    """
    def __init__(self, name: str):
        self.name = name

    def set_model(self, model: Model):
        """ Sets the current values of the model. """
        raise NotImplementedError()

    def get_model(self) -> Model:
        """ Gets the current values of the model. """
        raise NotImplementedError()

    def query_author_probabilities(self, recipient: str, messages: list[Message]) -> dict[str, float]:
        """ Infers who the author of the set of messages is. """
        raise NotImplementedError()

    def apply_received_messages_observation(self, recipient: str, messages: list[Message]):
        """ Marks a set of messages set as received by the given person. """
        raise NotImplementedError()
