"""
An example use of Tyche for knowledge extraction from a set of messages,
where the source that sent each message is unknown, but the recipient
is known. The properties of the messages can be used to learn the
different tendencies of the possible sources of the message, without
explicitly knowing which sources sent any of the messages. For example,
we can learn that Bob uses emojis often, without knowing what messages
Bob sent. This is achieved through the use of the observation learning
mechanisms of the individuals module of Tyche.

The Jupyter notebook walkthrough.ipynb in this directory
also provides similar functionality, but with additional description
and analysis. It is also simpler than this example, which may be easier
to follow. This example is used to test extending the example from
the notebook with role learning.
"""
import functools

from examples.anonymous_messages.base_model import AnonymousMessagesImplementation, Message, Model, ModelPerson
from tyche.individuals import *
from tyche.language import *


# We want to decay earlier observations faster than later observations.
# This is done to improve convergence as observations are made,
# while allowing the model to change more during early learning.
decaying_concept_learning_strat = StatisticalConceptLearningStrategy(
    decay_rate=0.95, decay_rate_for_decay_rate=0.95
)
decaying_role_learning_strat = StatisticalRoleLearningStrategy(
    decay_rate=0.85, decay_rate_for_decay_rate=0.9995
)


class Person(Individual):
    """
    An example person that has a set of preferences when writing messages.
    """
    def __init__(
            self, name: str,
            uses_emoji: TycheConceptValue = 0.5,
            capitalises_first_word: TycheConceptValue = 0.5,
            is_positive: TycheConceptValue = 0.5):

        super().__init__(name)
        self._conversed_with = ExclusiveRoleDist()
        self._uses_emoji = uses_emoji
        self._capitalises_first_word = capitalises_first_word
        self._is_positive = is_positive

    @role()
    def conversed_with(self):
        return self._conversed_with

    @conversed_with.learning_func(decaying_role_learning_strat)
    def set_conversed_with(self, dist: ExclusiveRoleDist):
        self._conversed_with = dist

    @concept()
    def uses_emoji(self):
        return self._uses_emoji

    @uses_emoji.learning_func(decaying_concept_learning_strat)
    def set_uses_emoji(self, prob: float):
        self._uses_emoji = prob

    @concept()
    def capitalises_first_word(self):
        return self._capitalises_first_word

    @capitalises_first_word.learning_func(decaying_concept_learning_strat)
    def set_capitalises_first_word(self, prob: float):
        self._capitalises_first_word = prob

    @concept()
    def is_positive(self):
        return self._is_positive

    @is_positive.learning_func(decaying_concept_learning_strat)
    def set_is_positive(self, prob: float):
        self._is_positive = prob


class TycheImplementation(AnonymousMessagesImplementation):
    """
    A model for the testing framework to interact with for this
    implementation using Tyche.
    """

    def __init__(self):
        super().__init__("Tyche")
        self.alice: Optional[Person] = None
        self.bob: Optional[Person] = None
        self.jeff: Optional[Person] = None
        self.all: list[Person] = []
        self.by_name: dict[str, Person] = {}

        self.c_uses_emoji = Concept("uses_emoji")
        self.c_capitalises = Concept("capitalises_first_word")
        self.c_is_positive = Concept("is_positive")
        self.r_conversed_with = Role("conversed_with")

    @staticmethod
    def _person_from_model(model: ModelPerson) -> Person:
        return Person(model.name, model.uses_emoji, model.capitalises_first_word, model.is_positive)

    def set_model(self, model: Model):
        self.alice = TycheImplementation._person_from_model(model.alice)
        self.bob = TycheImplementation._person_from_model(model.bob)
        self.jeff = TycheImplementation._person_from_model(model.jeff)
        self.all: list[Person] = [self.alice, self.bob, self.jeff]
        self.by_name: dict[str, Person] = {m.name: m for m in self.all}

        # Add conversed with relationships.
        for model_person in model.all:
            person = self.by_name[model_person.name]
            for conversation_partner_name, weight in model_person.conversed_with.items():
                conversation_partner = self.by_name[conversation_partner_name]
                if weight > 0:
                    person.conversed_with().add(conversation_partner, weight)

    @staticmethod
    def _model_from_person(person: Person) -> ModelPerson:
        model = ModelPerson(person.name, person.uses_emoji(), person.capitalises_first_word(), person.is_positive())
        for partner, prob in person.conversed_with():
            model.conversed_with[partner.name] = prob

        return model

    def get_model(self) -> Model:
        alice = TycheImplementation._model_from_person(self.alice)
        bob = TycheImplementation._model_from_person(self.bob)
        jeff = TycheImplementation._model_from_person(self.jeff)
        return Model(alice, bob, jeff)

    def build_messages_sentence(self, messages: list[Message]) -> ADLNode:
        """ Builds a logical sentence from the set of messages. """
        # Sample the properties of random messages.
        message_sentences = []
        for m in messages:
            o_uses_emoji = self.c_uses_emoji if m.uses_emoji else ~self.c_uses_emoji
            o_capitalises = self.c_capitalises if m.capitalises_first_word else ~self.c_capitalises
            o_is_positive = self.c_is_positive if m.is_positive else ~self.c_is_positive
            message_sentences.append(o_uses_emoji & o_capitalises & o_is_positive)

        # Combine the messages into one observation.
        return functools.reduce(lambda a, b: a & b, message_sentences)

    def query_author_probabilities(self, recipient_name: str, messages: list[Message]) -> dict[str, float]:
        recipient = self.by_name[recipient_name]
        messages_sentence = self.build_messages_sentence(messages)

        # Evaluate the probability of each possible recipient.
        results = {}
        for author, prob in recipient.conversed_with():
            results[author.name] = author.eval(messages_sentence) * prob

        return results

    def build_received_messages_observation(self, messages: list[Message]) -> ADLNode:
        """ Builds an observation that represents receiving the given set of messages. """
        return Expectation(self.r_conversed_with, self.build_messages_sentence(messages))

    def apply_received_messages_observation(self, recipient_name: str, messages: list[Message]):
        recipient = self.by_name[recipient_name]
        recipient.observe(self.build_received_messages_observation(messages))
