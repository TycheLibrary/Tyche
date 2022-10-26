"""
This contains an alternative implementation of the anonymous
messages example using ProbLog. This is created to serve as
a comparison to the Tyche implementation.
"""
from typing import Optional, Any

from problog import get_evaluatable
from problog.logic import Term, Constant
from problog.program import PrologString, SimpleProgram
from problog.learning import lfi

from examples.anonymous_messages.base_model import AnonymousMessagesImplementation, Model, Message, ModelPerson


class ProbLogImplementation(AnonymousMessagesImplementation):
    """
    A model for the testing framework to interact with for this
    implementation using Tyche.
    """

    def __init__(self):
        super().__init__("ProbLog")
        self.model: Optional[Model] = None
        self.learning_program_statements: list[Any] = []
        self.learning_evidence = []
        self.people_constants: dict[str, Constant] = {}
        self._next_message_number = 1
        self._next_obs_number = 1
        self._next_conv_number = 1

    def set_model(self, model: Model):
        self.model = model
        self.learning_program_statements = []
        self._next_message_number = 1
        self._next_obs_number = 1
        self._next_conv_number = 1

        # Add each person from the model to the learning program.
        t = Term("t")
        for index, person in enumerate(model.all):
            person_constant = Constant(index)
            self.people_constants[person.name] = person_constant

            self.learning_program_statements.append(
                Term("uses_emoji")(person_constant, None, p=t(Constant(person.uses_emoji))))
            self.learning_program_statements.append(
                Term("capitalises_first_word")(person_constant, None, p=t(Constant(person.capitalises_first_word))))
            self.learning_program_statements.append(
                Term("is_positive")(person_constant, None, p=t(Constant(person.is_positive))))

        # Add the relationships between people.
        for person in model.all:
            person_constant = self.people_constants[person.name]
            total_weight = 0
            for weight in person.conversed_with.values():
                total_weight += weight

            for other_name, weight in person.conversed_with.items():
                other_constant = self.people_constants[other_name]
                self.learning_program_statements.append(
                    Term("conversed_with")(person_constant, other_constant, None, p=t(Constant(weight / total_weight))))

    def get_model(self, *, debug_program_output_file: Optional[str] = None) -> Model:
        # Create the learning program.
        learning_program = SimpleProgram()
        for statement in self.learning_program_statements:
            learning_program += statement

        # I'm not sure why converting this to Prolog and back fixes an issue with grounding.
        learning_program = PrologString(learning_program.to_prolog())

        # Learn!
        score, weights, atoms, iteration, lfi_problem = lfi.run_lfi(
            learning_program, self.learning_evidence, max_iter=5)

        # Build a model from the results, relying on the fact that weights is sorted in program order.
        new_model = self.model.copy()
        weight_index = 0
        for person in new_model.all:
            person.uses_emoji = weights[weight_index]
            weight_index += 1
            person.capitalises_first_word = weights[weight_index]
            weight_index += 1
            person.is_positive = weights[weight_index]
            weight_index += 1

        for old_person in self.model.all:
            new_person = new_model.by_name[old_person.name]
            for other_name, _ in old_person.conversed_with.items():
                new_person.conversed_with[other_name] = weights[weight_index]
                weight_index += 1

        if debug_program_output_file is not None:
            with open(debug_program_output_file, "w") as f:
                f.write(lfi_problem.get_model())

        return new_model

    def construct_messages_term(
            self, messages: list[Message],
            additional_message_property_args: Optional[list[Any]] = None
    ) -> Term:
        """
        Constructs a ProbLog term representing the observation of the given messages.
        """
        if additional_message_property_args is None:
            additional_message_property_args = []

        term: Optional[Term] = None

        for message in messages:
            message_no = Constant(self._next_message_number)
            self._next_message_number += 1
            t_uses_emoji = Term("uses_emoji")(*additional_message_property_args, message_no)
            t_capitalises = Term("capitalises_first_word")(*additional_message_property_args, message_no)
            t_is_positive = Term("is_positive")(*additional_message_property_args, message_no)

            uses_emoji = (t_uses_emoji if message.uses_emoji else ~t_uses_emoji)
            if term is None:
                term = uses_emoji
            else:
                term &= uses_emoji

            term &= (t_capitalises if message.capitalises_first_word else ~t_capitalises)
            term &= (t_is_positive if message.is_positive else ~t_is_positive)

        if term is None:
            raise Exception("No messages supplied!")
        return term

    def _eval_messages_prob(self, person: ModelPerson, messages: list[Message]) -> float:
        """
        Evaluates the probability of a set of messages being authored by the given person.

        Evaluates ProbLog programs following the following Pseudo-code:
          {uses_emoji_prob} :: uses_emoji(_).
          {capitalises_first_word_prob} :: capitalises_first_word(_).
          {is_positive_prob} :: is_positive(_).

          wrote_messages :-
              [\\+]uses_emoji(1), [\\+]capitalises_first_word(1), [\\+]is_positive(1)
              [, [\\+]uses_emoji(2), [\\+]capitalises_first_word(2), [\\+]is_positive(2)]
              [, ...more messages].
          query(wrote_messages)
        """

        p = SimpleProgram()
        p += Term("uses_emoji", p=Constant(person.uses_emoji))(None)
        p += Term("capitalises_first_word", p=Constant(person.capitalises_first_word))(None)
        p += Term("is_positive", p=Constant(person.is_positive))(None)

        wrote_messages, query = Term("wrote_messages"), Term("query")
        messages_term = self.construct_messages_term(messages)
        p += (wrote_messages << messages_term)
        p += query(wrote_messages)

        results = get_evaluatable().create_from(p).evaluate()
        return results[wrote_messages]

    def query_author_probabilities(self, recipient_name: str, messages: list[Message]) -> dict[str, float]:
        recipient = self.model.by_name[recipient_name]

        # Evaluate the relative weighting of each possible recipient.
        results = {}
        for author_name, weight in recipient.conversed_with.items():
            author = self.model.by_name[author_name]
            results[author_name] = self._eval_messages_prob(author, messages) * weight

        return results

    def apply_received_messages_observation(self, recipient_name: str, messages: list[Message]):
        recipient = self.model.by_name[recipient_name]
        recipient_constant = self.people_constants[recipient_name]

        conversed_with_terms = {}
        for other_name in recipient.conversed_with.keys():
            other_constant = self.people_constants[other_name]
            conversed_with_terms[other_name] = Term("conversed_with")(
                recipient_constant, other_constant, Constant(self._next_conv_number))
            self._next_conv_number += 1

        obs = Term(f"obs{self._next_obs_number}")
        self._next_obs_number += 1
        for other_name in recipient.conversed_with.keys():
            other_constant = self.people_constants[other_name]
            messages_term = self.construct_messages_term(messages, [other_constant])

            received_term = conversed_with_terms[other_name]
            for name in recipient.conversed_with.keys():
                if name != other_name:
                    received_term &= ~conversed_with_terms[name]

            self.learning_program_statements.append((obs << (received_term & messages_term)))

        self.learning_evidence.append([(obs, True)])
