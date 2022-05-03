"""
The classes for representing and reasoning about aleatoric
description logic (ADL) formulas.

ADL is designed to support both mathematical notion and a formal english notion.
"""
from typing import Final, cast, TypeVar, Optional


class TycheLanguageException(Exception):
    """
    Class for detailing language exceptions.
    """
    def __init__(self, message):
        self.message = "TycheLanguageException: " + message


class RoleDistribution:
    """
    Represents a probability distribution of contexts for a role.
    The items in the probability distribution use weights to
    represent their likelihood of being selected. The probability
    of selecting an item is fixed at 100%.
    """
    def __init__(self):
        self.entries: list[tuple[Optional['TycheContext'], float]] = []
        self.total_weight = 0

    def clear(self):
        """
        Removes all individuals from this distribution.
        """
        self.entries = []
        self.total_weight = 0

    def _index_of(self, individual: Optional['TycheContext']):
        for index, (ctx, _) in enumerate(self.entries):
            if ctx is individual:
                return index
        return None

    def contains(self, individual: Optional['TycheContext']):
        """
        Returns whether this role distribution contains the given individual.
        """
        return self._index_of(individual) is not None

    def _update_total_weight(self):
        total = 0
        for _, weight in self.entries:
            total += weight
        self.total_weight = total

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
            self.entries[existing_index] = entry
        else:
            self.entries.append(entry)

        self._update_total_weight()

    def remove(self, individual: Optional['TycheContext']):
        """
        Removes the given individual from this distribution.
        """
        existing_index = self._index_of(individual)
        if existing_index is None:
            return

        del self.entries[existing_index]
        self._update_total_weight()

    def __iter__(self):
        """
        Yields tuples of TycheContext objects or None, and their associated
        probabilities. The sum of all returned probabilities should
        sum to 1, although there may be some deviance from this due
        to floating point error.
        """
        total_weight = self.total_weight
        if total_weight == 0:
            # If there are no entries, then yield the None entry with probability 1.
            yield None, 1
            return

        for context, weight in self.entries:
            yield context, weight / total_weight


class TycheContext:
    """
    Provides context for relating variables in formulas
    (e.g. a, b, c, etc...) to their related objects.
    Each individual may supply their own context for
    their variables and roles.
    """
    def eval_concept(self, symbol: str) -> float:
        raise TycheLanguageException("eval_concept is unimplemented for " + type(self).__name__)

    def eval_role(self, symbol: str) -> RoleDistribution:
        raise TycheLanguageException("eval_role is unimplemented for " + type(self).__name__)


class EmptyContext(TycheContext):
    """
    Provides an empty context for evaluating constant expressions.
    """
    def eval_concept(self, symbol: str) -> float:
        raise TycheLanguageException("Unknown atom {}".format(symbol))

    def eval_role(self, symbol: str) -> RoleDistribution:
        raise TycheLanguageException("Unknown role {}".format(symbol))


# This is used to allow passing names (e.g. "x", "y", etc...) directly
# to functions that require a concept or role. These names will then
# automatically be converted to an Atom or Role object.
CompatibleWithConcept: type = TypeVar("CompatibleWithConcept", 'Concept', str)
CompatibleWithRole: type = TypeVar("CompatibleWithRole", 'Role', str)


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

    def get_child_concepts(self) -> list['Concept']:
        """
        Returns the child concepts of this concept.
        """
        raise TycheLanguageException("get_child_concepts is unimplemented for " + type(self).__name__)

    def __str__(self) -> str:
        """
        gives a compact string representation of the structure of the formula
        in terms of primitive operators
        """
        raise TycheLanguageException("__str__ is unimplemented for " + type(self).__name__)

    def __repr__(self) -> str:
        """
        gives the constructor string of the object
        """
        raise TycheLanguageException("__repr__ is unimplemented for " + type(self).__name__)

    # maybe also include a function for giving an optimised inline string representation of a formula.

    def __eq__(self, other: 'Concept') -> bool:
        """
        return true if formulas are identical
        """
        raise TycheLanguageException("__eq__ is unimplemented for " + type(self).__name__)

    def __lt__(self, other: 'Concept') -> bool:
        """
        establishes a syntactic ordering over formula
        """
        raise TycheLanguageException("__lt__ is unimplemented for " + type(self).__name__)

    def eval(self, context: TycheContext) -> float:
        """
        returns the probability of this concept,
        given the lambda evaluation of roles and concepts.
        It is assumed concepts is a function that
        maps concept symbols to probabilities,
        and roles is a function that maps role symbols
        to probability distributions of tuples of concept functions and role functions
        """
        raise TycheLanguageException("eval is unimplemented for " + type(self).__name__)

    def normal_form(self) -> 'Concept':
        """
        Returns the tree normal form of the formula, where atoms are ordered alphabetically.
        """
        raise TycheLanguageException("normal_form is unimplemented for " + type(self).__name__)

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

    def complement(self):
        """
        inline negation operator
        """
        return never.when(self).otherwise(always)

    def __and__(self, concept):
        """
        inline conjunction operator,
        note ordering for lazy evaluation
        """
        return concept.when(self).otherwise(never)

    def __or__(self, concept):
        """
        inline disjunction operator
        """
        return always.when(self).otherwise(concept)

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

    def eval(self, context: TycheContext) -> float:
        value = context.eval_concept(self.symbol)
        if value < 0 or value > 1:
            raise TycheLanguageException(
                "Evaluation of atom {} did not fall into the range [0, 1], instead was {}".format(
                    self.symbol, value
                )
            )
        return value

    def normal_form(self):
        return self

    def is_equivalent(self, concept: Concept) -> bool:
        return self == concept

    def is_weaker(self, concept):
        raise TycheLanguageException("is_weaker is unimplemented for " + type(self).__name__)


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

    def eval(self, context: TycheContext) -> RoleDistribution:
        return context.eval_role(self.symbol)


class Constant(Atom):
    """
    A constant named aleatoric probability.
    """
    def __init__(self, symbol: str, probability: float):
        super().__init__(symbol, special_symbol=True)
        self.probability = probability

    def eval(self, context: TycheContext) -> float:
        return self.probability


always: Final[Constant] = Constant("\u22A4", 1)
never: Final[Constant] = Constant("\u22A5", 0)


class Conditional(Concept):
    """
    Represents an aleatoric ternary construct (if-then-else).
    """
    def __init__(self, condition: CompatibleWithConcept, if_yes: CompatibleWithConcept, if_no: CompatibleWithConcept):
        self.condition = Concept.cast(condition)
        self.if_yes = Concept.cast(if_yes)
        self.if_no = Concept.cast(if_no)

    def __str__(self):
        # Shorthand representations.
        if self.if_yes == always and self.if_no == never:
            return str(self.condition)
        if self.if_yes == never and self.if_no == always:
            return "\u00AC{}".format(str(self.condition))
        if self.if_no == never:
            return "({} \u2227 {})".format(str(self.condition), str(self.if_yes))
        if self.if_yes == always:
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

    def eval(self, context: TycheContext):
        cond = self.condition.eval(context)
        if_yes = self.if_yes.eval(context)
        if_no = self.if_no.eval(context)
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
        super().__init__(condition, if_yes, always)

    def otherwise(self, if_no: Concept) -> Conditional:
        return Conditional(self.condition, self.if_yes, if_no)


class Expectation(Concept):
    """
    Represents the aleatoric expectation construct.
    """
    def __init__(self, role: CompatibleWithRole, concept: CompatibleWithConcept):
        self.role: Role = Role.cast(role)
        self.concept: Concept = Concept.cast(concept)

    def __str__(self):
        return "(\U0001D53C_{}. {})".format(self.role, str(self.concept))

    def __repr__(self):
        return 'Expectation(role=' + repr(self.role) + ', concept=' + repr(self.concept) + ')'

    def __eq__(self, obj):
        if type(obj) != type(self):
            return False

        other: 'Expectation' = cast('Expectation', obj)
        return self.role == other.role and self.concept == other.concept

    def __lt__(self, other):
        raise TycheLanguageException("not yet implemented")

    def eval(self, context: TycheContext):
        """
        Evaluates the concept for all members of the role mapping
        from the given context to other contexts. This evaluation
        contains an implicit given that the role is non-None.
        If the role only contains None, then this will evaluate
        to vacuously True.
        """
        total_prob = 0.0
        prob_weight = 1.0
        for other_context, prob in self.role.eval(context):
            if other_context is None:
                prob_weight = 1.0 - prob
            else:
                total_prob += prob * self.concept.eval(other_context)

        # Vacuous True if only None in role.
        if prob_weight == 0:
            return 1

        # Division for the implicit given non-None.
        return total_prob / prob_weight

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


class LeastFixedPoint(Concept):
    """
    class for representing the aleatoric fixed point construct in the language.
    This is equivalent to the marginal expectation operator, (a|b).
    See: https://peteroupc.github.io/bernoulli.html
    """
    def __init__(self, variable, concept):
        if LeastFixedPoint.is_linear(variable, concept):
            self.variable = variable  # role object
            self.concept = concept  # concept object
        else:
            raise TycheLanguageException("The variable {} is not linear in {}".format(variable, concept))

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

    def eval(self, context: TycheContext):
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
