"""
The classes for representing and reasoning about aleatoric
description logic (ADL) formulas.

ADL is designed to support both mathematical notion and a formal english notion.
"""
import random
from typing import Final, cast, Optional, Union, Tuple, NewType, Callable

from tyche.probability import uncertain_bayes_rule
from tyche.reference import SymbolReference, BakedSymbolReference
from tyche.string_utils import format_dict


class TycheLanguageException(Exception):
    """
    Class for detailing language exceptions.
    """
    def __init__(self, message: str):
        self.message = "TycheLanguageException: " + message


RoleDistributionEntries: type = Union[
    None,
    list[Union['TycheContext', Tuple['TycheContext', float]]],
    dict['TycheContext', float]
]


class ExclusiveRoleDist:
    """
    Represents a probability distribution of contexts for a role,
    where either zero or one related context can possibly be
    active at once when observed. The zero in this case refers
    to the possibility of selecting the None-individual. However,
    an individual is always active if you include the None-individual.
    The items in the probability distribution use weights to
    represent their relative likelihood of being selected.
    """
    def __init__(self, entries: RoleDistributionEntries = None):
        self._entries: list[tuple[Optional['TycheContext'], float]] = []
        if entries is not None:
            if isinstance(entries, list):
                for entry in entries:
                    if isinstance(entry, TycheContext):
                        self.add(entry)
                        continue
                    if isinstance(entry, tuple):
                        ctx, weight = entry
                        if isinstance(ctx, TycheContext) and isinstance(weight, float):
                            self.add(ctx, weight)
                            continue

                    raise TycheLanguageException(f"Illegal entry in entries list: {entry}")
            elif isinstance(entries, dict):
                for ctx, weight in entries.items():
                    if not isinstance(ctx, TycheContext) or not isinstance(weight, float):
                        raise TycheLanguageException(f"Illegal entry in entries dict: {(ctx, weight)}")
                    self.add(ctx, weight)
            else:
                raise TycheLanguageException(f"Illegal entries value: {entries}")

    def is_empty(self) -> bool:
        """
        Returns whether no individuals (including None) have been added to this role.
        If None was explicitly added, then this will return False.
        """
        return len(self._entries) == 0

    @property
    def total_weight(self):
        """ The sum of the weights of all entries in this role distribution. """
        total = 0
        for _, weight in self._entries:
            total += weight
        return total

    def clear(self):
        """ Removes all individuals from this distribution. """
        self._entries = []

    def _index_of(self, individual: Optional['TycheContext']):
        for index, (ctx, _) in enumerate(self._entries):
            if ctx is individual:
                return index
        return None

    def contains(self, individual: Optional['TycheContext']):
        """
        Returns whether this role distribution contains the given individual.
        """
        return self._index_of(individual) is not None

    def add(self, individual: Optional['TycheContext'], weight: float = 1):
        """
        Add an individual to this distribution with the given weighting.
        The weightings of individuals are relative to one another. If
        an individual already exists in the distribution, then its
        weighting will be _replaced_.

        If no weight is supplied, then the default of 1 will be used.
        """
        if weight <= 0:
            raise TycheLanguageException("Value weights must be positive, not {}".format(weight))

        entry = (individual, weight)

        existing_index = self._index_of(individual)
        if existing_index is not None:
            self._entries[existing_index] = entry
        else:
            self._entries.append(entry)

    def add_combining_weights(self, individual: Optional['TycheContext'], weight: float):
        """
        Add an individual to this distribution with the given weight.
        The weightings of individuals are relative to one another. If
        an individual already exists in the distribution, then its
        weighting will be increased by the given weight.
        """
        if weight <= 0:
            raise TycheLanguageException("Value weights must be positive, not {}".format(weight))

        existing_index = self._index_of(individual)
        if existing_index is not None:
            _, existing_weight = self._entries[existing_index]
            self._entries[existing_index] = (individual, existing_weight + weight)
        else:
            self._entries.append((individual, weight))

    def remove(self, individual: Optional['TycheContext']):
        """ Removes the given individual from this distribution. """
        existing_index = self._index_of(individual)
        if existing_index is None:
            return

        del self._entries[existing_index]

    def contexts(self):
        """ Yields all the contexts within this role, including None if it is present. """
        for ctx, _ in self._entries:
            yield ctx

    def apply_bayes_rule(
            self, observation: 'Concept', likelihood: float = 1, learning_rate: float = 1) -> 'ExclusiveRoleDist':
        """
        Applies Bayes' rule to update the probabilities of the individuals
        mapped within this role based upon an uncertain observation. This
        cannot learn anything about the None individual. Returns an updated
        role distribution.

        The likelihood works by applying bayes rule to the event that the observation occurred
        or was incorrectly observed ((event with likelihood) OR (NOT event with 1 - likelihood)).
        Therefore, a likelihood of 0 represents that the observation was observed to be false
        (this is equivalent to observing NOT observation).
        See: https://stats.stackexchange.com/questions/345200/applying-bayess-theorem-when-evidence-is-uncertain

        The learning_rate works by acting as a weight for the result of Bayes' rule, and the
        original weight. A learning_rate of 1 represents to just use the result of Bayes' rule.
        A learning_rate of 0 represents that the weight of each individual in the role should
        remain unchanged.
        """
        if likelihood < 0 or likelihood > 1:
            raise TycheLanguageException(
                f"The likelihood should fall between 0 and 1 inclusive. It was {likelihood}")
        if learning_rate < 0 or learning_rate > 1:
            raise TycheLanguageException(
                f"The learning_rate should fall between 0 and 1 inclusive. It was {learning_rate}")

        # If the learning rate is 0, then nothing will be learned.
        if learning_rate == 0:
            return self

        # If there are no entries, then we can't learn anything.
        if self.total_weight == 0:
            return self

        role_belief: float = Expectation.evaluate_for_role(self, observation, ALWAYS)

        learned_entries: list[tuple[Optional['TycheContext'], float]] = []
        total_weight = self.total_weight
        for context, weight in self._entries:
            if context is None:
                # Can't learn the null-individual
                learned_entries.append((context, weight))
                continue

            belief = context.eval(observation)

            curr_prob = weight / total_weight
            new_prob = uncertain_bayes_rule(curr_prob, role_belief, belief, likelihood)
            new_weight = new_prob * total_weight

            learned_weight = learning_rate * new_weight + (1 - learning_rate) * weight
            if learned_weight > 0:
                learned_entries.append((context, learned_weight))

        return ExclusiveRoleDist(learned_entries)

    def __len__(self):
        """ Returns the number of entries in this role, including the None role if it is present. """
        return max(1, len(self._entries))

    def __iter__(self):
        """
        Yields tuples of TycheContext objects or None, and their associated
        probabilities (i.e. Tuple[Optional[TycheContext], float]).
        The sum of all returned probabilities should sum to 1,
        although there may be some deviance from this due to
        floating point error.
        """
        total_weight = self.total_weight
        if total_weight == 0:
            # If there are no entries, then yield the None entry with probability 1.
            yield None, 1
            return

        for context, weight in self._entries:
            yield context, weight / total_weight

    def sample(self) -> Optional['TycheContext']:
        """ Selects a random individual from this role based upon their weights. """
        target_cumulative_weight = random.uniform(0, self.total_weight)
        cumulative_weight = 0
        for context, weight in self._entries:
            cumulative_weight += weight
            if cumulative_weight >= target_cumulative_weight:
                return context

        return None

    def __str__(self):
        return self.to_str()

    def to_str(self, *, detail_lvl: int = 1, indent_lvl: int = 0):
        if self.is_empty():
            return f"{{<empty {type(self).__name__}>}}"

        sub_detail_lvl = detail_lvl - 1
        sub_indent_lvl = 0 if indent_lvl == 0 else indent_lvl + 1

        def format_prob(prob: int):
            return f"{100 * prob:.1f}%"

        def format_ctx(ctx: TycheContext):
            if ctx is None:
                return "<None context>"
            return ctx.to_str(detail_lvl=sub_detail_lvl, indent_lvl=sub_indent_lvl)

        key_values = [(prob, ctx) for ctx, prob in self]
        return format_dict(key_values, key_format_fn=format_prob, val_format_fn=format_ctx, indent_lvl=indent_lvl)


# This is used to allow passing names (e.g. "x", "y", etc...) directly
# to functions that require a concept or role. These names will then
# automatically be converted to an Atom or Role object.
CompatibleWithConcept: type = NewType("CompatibleWithConcept", Union['Concept', str])
CompatibleWithRole: type = NewType("CompatibleWithRole", Union['Role', str])


class TycheContext:
    """
    Provides context for relating variables in formulas
    (e.g. a, b, c, etc...) to their related objects.
    Each individual may supply their own context for
    their variables and roles.
    """
    def eval(self, concept: 'CompatibleWithConcept') -> float:
        """
        Evaluates the given concept to a probability of
        it being true if sampled within this context.
        """
        raise NotImplementedError("eval is unimplemented for " + type(self).__name__)

    def eval_role(self, role: 'CompatibleWithRole') -> ExclusiveRoleDist:
        """
        Evaluates the given role to a distribution of possible
        other contexts if it were sampled within this context.
        """
        raise NotImplementedError("eval_role is unimplemented for " + type(self).__name__)

    def observe(self, observation: 'Concept', likelihood: float = 1, learning_rate: float = 1):
        """
        Attempts to update the beliefs of this individual based upon
        an observation of the given concept.

        The optional likelihood parameter provides a degree of certainty
        about the observation. By default, the observation is assumed to
        be reliable.
        """
        raise NotImplementedError("observe is unimplemented for " + type(self).__name__)

    def get_concept(self, symbol: str) -> float:
        """
        Gets the probability of the atom with the given symbol
        being true, without modification by the context.
        """
        raise NotImplementedError("eval_concept is unimplemented for " + type(self).__name__)

    def get_role(self, symbol: str) -> ExclusiveRoleDist:
        """
        Gets the role distribution of the role with the
        given symbol, without modification by the context.
        """
        raise NotImplementedError("eval_role is unimplemented for " + type(self).__name__)

    def get_concept_reference(self, symbol: str) -> BakedSymbolReference[float]:
        """
        Gets a mutable reference to the probability of the atom with the given symbol
        being true. This reference can be used to get and set the value of the atom.
        """
        raise NotImplementedError("eval_mutable_concept is unimplemented for " + type(self).__name__)

    def get_role_reference(self, symbol: str) -> BakedSymbolReference[ExclusiveRoleDist]:
        """
        Gets a mutable reference to the role distribution of the role with the
        given symbol. This reference can be used to get and set the value of the role.
        """
        raise NotImplementedError("eval_mutable_role is unimplemented for " + type(self).__name__)

    def __str__(self):
        return self.to_str()

    def to_str(self, *, detail_lvl: int = 1, indent_lvl: int = 0):
        """
        A version of str() that allows additional formatting options to be specified.
        Allows specifying the number of levels of detail to include, and the
        indentation to use while formatting.
        """
        raise NotImplementedError("to_str is unimplemented for " + type(self).__name__)


class EmptyContext(TycheContext):
    """
    Provides an empty context for evaluating constant expressions.
    """
    def eval(self, concept: 'Concept') -> float:
        return concept.direct_eval(self)

    def eval_role(self, role: 'Role') -> ExclusiveRoleDist:
        return role.direct_eval(self)

    def get_concept(self, symbol: str) -> float:
        raise TycheLanguageException("Unknown atom {}".format(symbol))

    def get_role(self, symbol: str) -> ExclusiveRoleDist:
        raise TycheLanguageException("Unknown role {}".format(symbol))

    def get_concept_reference(self, symbol: str) -> float:
        raise TycheLanguageException("Unknown atom {}".format(symbol))

    def get_role_reference(self, symbol: str) -> ExclusiveRoleDist:
        raise TycheLanguageException("Unknown role {}".format(symbol))

    def to_str(self, *, detail_lvl: int = 1, indent_lvl: int = 0):
        return f"<EmptyContext>"


class Concept:
    """
    The base class of all nodes in aleatoric description logic formulas.
    """
    @staticmethod
    def cast(concept: CompatibleWithConcept) -> 'Concept':
        if isinstance(concept, Concept):
            return concept
        elif isinstance(concept, str):
            return Atom(concept)
        else:
            raise TycheLanguageException("Incompatible concept type {}".format(type(concept).__name__))

    def get_child_concepts_in_eval_context(self) -> list['Concept']:
        """
        Returns an ordered list of the child concepts of this concept that would be
        evaluated in the same context as this concept is evaluated.
        """
        raise NotImplementedError("get_children_in_eval_context is unimplemented for " + type(self).__name__)

    def copy_with_new_child_concept_from_eval_context(self, index: int, concept: 'Concept'):
        """
        Returns a copy of this concept with the child concept at the given index
        in the list returned by get_children_in_eval_context replaced with the given concept.
        """
        raise NotImplementedError("copy_with_replaced_child_from_eval_context is unimplemented for " + type(self).__name__)

    def __str__(self) -> str:
        """
        gives a compact string representation of the structure of the formula
        in terms of primitive operators
        """
        raise NotImplementedError("__str__ is unimplemented for " + type(self).__name__)

    def __repr__(self) -> str:
        """
        gives the constructor string of the object
        """
        raise NotImplementedError("__repr__ is unimplemented for " + type(self).__name__)

    # maybe also include a function for giving an optimised inline string representation of a formula.

    def __eq__(self, other: 'Concept') -> bool:
        """
        return true if formulas are identical
        """
        raise NotImplementedError("__eq__ is unimplemented for " + type(self).__name__)

    def __lt__(self, other: 'Concept') -> bool:
        """
        establishes a syntactic ordering over formula
        """
        raise NotImplementedError("__lt__ is unimplemented for " + type(self).__name__)

    def direct_eval(self, context: TycheContext) -> float:
        """
        Disclaimer:
        This should _NOT_ be called to evaluate concepts about your model, as the
        TycheContext objects should be given the option to change the method of
        evaluation. Therefore, TycheContext#eval should usually be used instead.

        Evaluates the probability of this concept evaluating to true when sampled
        using the values of atoms and roles from the given context.
        """
        raise NotImplementedError("direct_eval is unimplemented for " + type(self).__name__)

    def normal_form(self) -> 'Concept':
        """
        Returns the tree normal form of the formula, where atoms are ordered alphabetically.
        """
        raise NotImplementedError("normal_form is unimplemented for " + type(self).__name__)

    def is_equivalent(self, concept: 'Concept') -> bool:
        """
        Returns true if this Concept is provably equivalent to concept
        delegates to normal form function.
        Two concepts have the same normal form if and only if
        they have the same evaluation function
        """
        return self.normal_form() == concept.normal_form()

    def is_weaker(self, concept: 'Concept') -> bool:
        """
        Returns true if the probability of this Concept is provably
        necessarily less than or equal to the probability of concept
        """
        # need to consider how to effectively implement this
        # I think move everything to normal form and then some traversal?
        # However, it is more than just inclusion.
        # eg x/\~x is always weaker than y\/~y
        # need to factor out inclusion, then find separating constants
        raise TycheLanguageException("is_weaker is unimplemented for " + type(self).__name__)

    def is_stronger(self, concept: 'Concept') -> bool:
        """
        Returns true if the probability of this concept is provably
        necessarily greater than or equal to the probability of adl_form
        """
        return concept.is_weaker(self)

    def when(self, condition: CompatibleWithConcept) -> 'ConditionalWithoutElse':
        """
        Returns a formula that represents (condition ? self : Always).
        This is equivalent to the formula self -> condition.
        You can use a.when(b).otherwise(c) to represent (b ? a : c).
        """
        return ConditionalWithoutElse(condition, self)

    def complement(self) -> 'Concept':
        """
        inline negation operator
        """
        return NEVER.when(self).otherwise(ALWAYS)

    def __and__(self, concept) -> 'Concept':
        """
        inline conjunction operator,
        note ordering for lazy evaluation
        """
        return concept.when(self).otherwise(NEVER)

    def __or__(self, concept) -> 'Concept':
        """
        inline disjunction operator
        """
        return ALWAYS.when(self).otherwise(concept)

    '''
    Other inline operators to define:
    concept.for(role)
    concept.for(role).given(margin)
    concept.necessary_for(role)
    concept.possible_for(role)
    others...
    '''


class Atom(Concept):
    """
    Represents indivisible concepts such as always, never, constants, and named concepts.
    """
    def __init__(self, symbol: str, *, special_symbol: bool = False, symbol_type_name: str = "Atom"):
        if not special_symbol:
            Atom.check_symbol(symbol, symbol_type_name=type(self).__name__)

        self.symbol = symbol

    def get_child_concepts_in_eval_context(self) -> list[Concept]:
        return []

    def copy_with_new_child_concept_from_eval_context(self, index: int, concept: Concept):
        raise IndexError("Atom's have no child concepts")

    @staticmethod
    def check_symbol(symbol: str, *, symbol_name="symbol", symbol_type_name: str = "Atom", context: str = None):
        """
        Checks a string contains only alphanumeric characters or underscore.
        Raises an error if the symbol is an invalid atom symbol.
        """
        context_suffix = "" if context is None else ". Errored in {}".format(context)

        if symbol is None:
            raise ValueError("{} symbols cannot be None{}".format(symbol_type_name, context_suffix))
        if len(symbol) == 0:
            raise ValueError("{} symbols cannot be empty strings{}".format(symbol_type_name, context_suffix))

        context_suffix = ". Errored for {} '{}'{}".format(
            symbol_name, symbol, "" if context is None else " in {}".format(context)
        )
        if not symbol[0].islower():
            raise ValueError("{} symbols must start with a lowercase letter{}".format(
                symbol_type_name, context_suffix
            ))

        for ch in symbol[1:]:
            if ch != '_' and not ch.isalnum:
                raise ValueError("{} symbols can only contain alpha-numeric or underscore characters{}".format(
                    symbol_type_name, context_suffix
                ))

    def __str__(self):
        return self.symbol

    def __repr__(self):
        return "{}(symbol={})".format(type(self).__name__, self.symbol)

    def __eq__(self, other) -> bool:
        return type(self) == type(other) and self.symbol == cast('Atom', other).symbol

    def __lt__(self, other) -> bool:
        raise TycheLanguageException("not yet implemented")

    def direct_eval(self, context: TycheContext) -> float:
        return context.get_concept(self.symbol)

    def eval_reference(self, context: TycheContext) -> BakedSymbolReference[float]:
        """ Evaluates to a mutable reference to the value of this atom. """
        return context.get_concept_reference(self.symbol)

    def normal_form(self):
        return self

    def is_equivalent(self, concept: Concept) -> bool:
        return self == concept

    def is_weaker(self, concept):
        raise NotImplementedError("is_weaker is unimplemented for " + type(self).__name__)


class Role:
    """
    Represents the relationships between contexts.
    """
    def __init__(self, symbol: str, *, special_symbol: bool = False):
        if not special_symbol:
            Atom.check_symbol(symbol, symbol_type_name=type(self).__name__)

        self.symbol = symbol

    @staticmethod
    def cast(role: CompatibleWithRole) -> 'Role':
        if isinstance(role, Role):
            return role
        elif isinstance(role, str):
            return Role(role)
        else:
            raise TycheLanguageException("Incompatible role type {}".format(type(role).__name__))

    def __str__(self):
        return self.symbol

    def __repr__(self):
        return "{}(symbol={})".format(type(self).__name__, self.symbol)

    def __eq__(self, other) -> bool:
        return type(self) == type(other) and self.symbol == cast('Atom', other).symbol

    def direct_eval(self, context: TycheContext) -> ExclusiveRoleDist:
        return context.get_role(self.symbol)

    def eval_reference(self, context: TycheContext) -> BakedSymbolReference[ExclusiveRoleDist]:
        """ Evaluates to a mutable reference to the value of this role. """
        return context.get_role_reference(self.symbol)


class Constant(Atom):
    """
    A constant named aleatoric probability.
    """
    def __init__(self, symbol: str, probability: float):
        super().__init__(symbol, special_symbol=True)
        self.probability = probability

    def direct_eval(self, context: TycheContext) -> float:
        return self.probability


ALWAYS: Final[Constant] = Constant("\u22A4", 1)
NEVER: Final[Constant] = Constant("\u22A5", 0)


class Conditional(Concept):
    """
    Represents an aleatoric ternary construct (if-then-else).
    """
    def __init__(self, condition: CompatibleWithConcept, if_yes: CompatibleWithConcept, if_no: CompatibleWithConcept):
        self.condition = Concept.cast(condition)
        self.if_yes = Concept.cast(if_yes)
        self.if_no = Concept.cast(if_no)

    def get_child_concepts_in_eval_context(self) -> list[Concept]:
        return [self.condition, self.if_yes, self.if_no]

    def copy_with_new_child_concept_from_eval_context(self, index: int, concept: Concept):
        args = self.get_child_concepts_in_eval_context()
        args[index] = concept
        return Conditional(*args)

    def __str__(self):
        # Shorthand representations.
        if self.if_yes == ALWAYS and self.if_no == NEVER:
            return str(self.condition)
        if self.if_yes == NEVER and self.if_no == ALWAYS:
            return "\u00AC{}".format(str(self.condition))
        if self.if_no == NEVER:
            return "({} \u2227 {})".format(str(self.condition), str(self.if_yes))
        if self.if_yes == ALWAYS:
            return "({} \u2228 {})".format(str(self.condition), str(self.if_no))

        # Standard ternary.
        return "({} ? {} : {})".format(
            str(self.condition), str(self.if_yes), str(self.if_no)
        )

    def __repr__(self):
        return "Conditional(condition={}, if_yes={}, if_no={})".format(
            repr(self.condition), repr(self.if_yes), repr(self.if_no)
        )

    def __eq__(self, obj):
        if type(obj) != type(self):
            return False

        other: 'Conditional' = cast('Conditional', obj)
        return (self.condition == other.condition
                and self.if_yes == other.if_yes
                and self.if_no == other.if_no)

    def __lt__(self, obj):
        raise TycheLanguageException("not yet implemented")

    def direct_eval(self, context: TycheContext):
        cond = context.eval(self.condition)
        if_yes = context.eval(self.if_yes)
        if_no = context.eval(self.if_no)
        return cond * if_yes + (1 - cond) * if_no

    def normal_form(self):
        """
        Returns the tree normal form of the conditional,
        by recursively calling normal for on sub elements.
        """
        raise TycheLanguageException("not yet implemented")

    def is_weaker(self, concept):
        raise TycheLanguageException("not yet implemented")


class ConditionalWithoutElse(Conditional):
    """
    Represents an aleatoric ternary construct of the form (condition ? if_yes : Always).
    """
    def __init__(self, condition: CompatibleWithConcept, if_yes: CompatibleWithConcept):
        super().__init__(condition, if_yes, ALWAYS)

    def otherwise(self, if_no: Concept) -> Conditional:
        return Conditional(self.condition, self.if_yes, if_no)


class Given(Concept):
    """
    Represents a concept that should only be evaluated after the given evaluates to true.
    Due to this, the evaluation of the Given must be handled by the context.
    Therefore, these should be avoided generally, and only used when they are required.
    """
    def __init__(self, concept: CompatibleWithConcept, given: CompatibleWithConcept):
        self.concept = Concept.cast(concept)
        self.given = Concept.cast(given)

    @staticmethod
    def maybe_unpack(
            concept: CompatibleWithConcept,
            given: Optional[CompatibleWithConcept] = None
    ) -> tuple[Concept, Concept]:
        """
        If the concept is a Given, then this unpacks the Given into its concept and the given itself.
        If given is provided whilst the concept is also a Given,
        the resulting given will be the AND of both.
        Returns a tuple of (concept, given).
        """
        given: Optional[Concept] = Concept.cast(given) if given is not None else None
        while isinstance(concept, Given):
            concept_as_given = cast(Given, concept)
            concept = concept_as_given.concept
            given = concept_as_given.given if given is None else given & concept_as_given.given

        if given is None:
            given = ALWAYS
        return concept, given

    def get_child_concepts_in_eval_context(self) -> list[Concept]:
        return []

    def copy_with_new_child_concept_from_eval_context(self, index: int, concept: Concept):
        raise IndexError("Given operators must be specially handled by the context")

    def __str__(self):
        return f"({self.concept} | {self.given})"

    def __repr__(self):
        return f"Given(concept={repr(self.concept)}, given={repr(self.given)})"

    def __eq__(self, obj):
        if type(obj) != type(self):
            return False

        other: 'Given' = cast('Given', obj)
        return self.concept == other.concept and self.given == other.given

    def __lt__(self, obj):
        raise TycheLanguageException("not yet implemented")

    def direct_eval(self, context: TycheContext):
        raise IndexError("The Given operator must be evaluated specially by the context")

    def normal_form(self):
        raise TycheLanguageException("not yet implemented")

    def is_weaker(self, concept):
        raise TycheLanguageException("not yet implemented")


class Expectation(Concept):
    """
    Represents the aleatoric expectation construct,
    'The expectation that concept is true for an individual selected randomly
    from the given role, given that the individual evaluates given to true'.
    If given is not given, then it defaults to always.
    """
    def __init__(
            self,
            role: CompatibleWithRole,
            concept: CompatibleWithConcept,
            given: Optional[CompatibleWithConcept] = None):

        concept, given = Given.maybe_unpack(concept, given)
        self.role: Role = Role.cast(role)
        self.concept: Concept = Concept.cast(concept)
        self.given = Concept.cast(given) if given is not None else ALWAYS

    def get_child_concepts_in_eval_context(self) -> list[Concept]:
        return []

    def copy_with_new_child_concept_from_eval_context(self, index: int, concept: Concept):
        raise IndexError("Expectation operators have no child concepts in the eval context")

    def __str__(self):
        if self.given == ALWAYS:
            return f"(\U0001D53C_{str(self.role)}. {str(self.concept)})"
        else:
            return f"(\U0001D53C_{str(self.role)}. {str(self.concept)} | {str(self.given)})"

    def __repr__(self):
        return f"Expectation(role={repr(self.role)}, concept={repr(self.concept)}, given={repr(self.given)})"

    def __eq__(self, obj):
        if type(obj) != type(self):
            return False

        other: 'Expectation' = cast('Expectation', obj)
        return self.role == other.role and self.concept == other.concept and self.given == other.given

    @staticmethod
    def evaluate_role_under_role(outer_role: ExclusiveRoleDist, inner_role: Role) -> ExclusiveRoleDist:
        """
        Evaluates the expected role over a given set of roles. The chance
        of the contexts in the returned role represents the chance that
        the context is selected from an inner_role selected from outer_role.
        """
        result = ExclusiveRoleDist()
        for outer_context, outer_prob in outer_role:
            if outer_context is None:
                result.add_combining_weights(None, outer_prob)
                continue

            for inner_context, inner_prob in outer_context.get_role(inner_role.symbol):
                result.add_combining_weights(inner_context, outer_prob * inner_prob)

        return result

    @staticmethod
    def evaluate_for_role(role: ExclusiveRoleDist, concept: Concept, given: Concept) -> float:
        """
        Evaluates this expectation over the given role, without requiring a context.
        This evaluation contains an implicit given that the role is non-None. If the
        role only contains None, then this will evaluate to vacuously True.
        """
        concept, given = Given.maybe_unpack(concept, given)

        total_prob = 0.0
        total_given_prob = 0.0
        concept_and_given = concept & given
        for other_context, prob in role:
            if other_context is None:
                continue

            given_prob = other_context.eval(given)
            if given_prob <= 0:
                continue

            true_prob = other_context.eval(concept_and_given)
            total_prob += prob * true_prob
            total_given_prob += prob * given_prob

        # Vacuously True if only None in role, or if all givens evaluated to never.
        if total_given_prob == 0:
            return 1

        # Division for the implicit given non-None.
        return total_prob / total_given_prob

    @staticmethod
    def reverse_observation(
            role: ExclusiveRoleDist, concept: Concept, given: Concept, likelihood: float = 1) -> ExclusiveRoleDist:
        """
        Evaluates to a role that represents the chance that each individual in the role
        was the individual that contributed to the observation. The likelihood gives the
        chance that the observation was true (i.e., a likelihood of 0 represents that the
        observation of this expectation was false).
        """
        entries: list[tuple[TycheContext, float]] = []

        total_given_prob = 0.0
        for other_context, prob in role:
            if other_context is None:
                continue

            given_prob = other_context.eval(given)
            if given_prob <= 0:
                continue

            concept_prob = other_context.eval(concept)
            given_prob = other_context.eval(given)

            matches_observation_prob = likelihood * concept_prob + (1 - likelihood) * (1 - concept_prob)
            chosen_prob = prob * given_prob
            total_given_prob += chosen_prob

            weight = chosen_prob * matches_observation_prob
            if weight > 0:
                entries.append((other_context, weight))

        # If no entries could have possibly matched, then the expectation would have been vacuously true.
        if total_given_prob == 0:
            return ExclusiveRoleDist()
        if len(entries) == 0:
            raise TycheLanguageException("The observation is impossible under this model")

        # Division for the implicit given non-None.
        return ExclusiveRoleDist(entries)

    def direct_eval(self, context: TycheContext):
        """
        Evaluates the concept for all members of the role mapping
        from the given context to other contexts. This evaluation
        contains an implicit given that the role is non-None.
        If the role only contains None, then this will evaluate
        to vacuously True.
        """
        return Expectation.evaluate_for_role(context.eval_role(self.role), self.concept, self.given)


class Exists(Concept):
    """
    Represents the expectation about whether a value for a role exists.
    i.e. The probability that the value of a role is non-None.
    """
    def __init__(self, role: CompatibleWithRole):
        self.role: Role = Role.cast(role)

    def get_child_concepts_in_eval_context(self) -> list[Concept]:
        return []

    def copy_with_new_child_concept_from_eval_context(self, index: int, concept: Concept):
        raise IndexError("Exists concepts have no child concepts")

    def __str__(self):
        return "(Exists_{})".format(str(self.role))

    def __repr__(self):
        return "Exists(role={})".format(repr(self.role))

    def __eq__(self, obj):
        if type(obj) != type(self):
            return False

        other: 'Exists' = cast('Exists', obj)
        return self.role == other.role

    def direct_eval(self, context: TycheContext):
        """
        Evaluates the likelihood that the given role has a non-None value.
        """
        exists_weight = 0
        for other_context, prob in context.eval_role(self.role):
            if other_context is None:
                return 1.0 - prob
            else:
                exists_weight += prob

        # Vacuous False if no entries in role.
        if exists_weight == 0:
            return 0.0

        # Otherwise, there was no None entry in this role, so it always exists.
        return 1.0


class LeastFixedPoint(Concept):
    """
    class for representing the aleatoric fixed point construct in the language.
    This is equivalent to the marginal expectation operator, (a|b).
    See: https://peteroupc.github.io/bernoulli.html
    """
    def __init__(self, variable: CompatibleWithRole, concept: CompatibleWithConcept):
        variable = Role.cast(variable)
        concept = Concept.cast(concept)

        if LeastFixedPoint.is_linear(variable, concept):
            self.variable = variable  # role object
            self.concept = concept  # concept object
        else:
            raise TycheLanguageException("The variable {} is not linear in {}".format(variable, concept))

    def get_child_concepts_in_eval_context(self) -> list[Concept]:
        return [self.concept]

    def copy_with_new_child_concept_from_eval_context(self, index: int, concept: Concept):
        args = self.get_child_concepts_in_eval_context()
        args[index] = concept
        return Conditional(self.variable, *args)

    def __str__(self):
        """
        Use X as the fixed point quantifier,
        if least and greatest not relevant?
        or is assignment appropriate x<=(father.(bald?YES:x)) (GFP is x>=(father.(bald?x:NO)) "all bald on the male line")
        eg LFP-x(father.(bald?YES:x)) the probability of having a bald ancestor on the male line.
        """
        return self.variable + '<=(' + self.concept + ')'

    def __repr__(self):
        return 'LeastFixedPoint(variable=' + repr(self.variable) + ', concept=' + repr(self.concept) + ')'

    def __eq__(self, obj):
        if type(obj) != type(self):
            return False

        other: 'LeastFixedPoint' = cast('LeastFixedPoint', obj)
        return self.variable == other.variable and self.concept == other.concept

    def __lt__(self, other):
        raise TycheLanguageException("not yet implemented")

    def direct_eval(self, context: TycheContext):
        """
        Complex one, needs iteration or equation solving.
        """
        raise TycheLanguageException("not yet implemented")

    def normal_form(self):
        """
        Returns the tree normal form of the conditional,
        by recursively calling normal for on sub elements.
        """
        raise TycheLanguageException("not yet implemented")

    def is_equivalent(self, concept):
        raise TycheLanguageException("not yet implemented")

    def is_weaker(self, concept):
        raise TycheLanguageException("not yet implemented")

    @staticmethod
    def is_linear(variable, concept):
        """
        class method to test whether variable is linear in concept
        """
        raise TycheLanguageException("not yet implemented")
