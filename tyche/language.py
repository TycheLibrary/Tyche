"""
The classes for representing and reasoning about aleatoric
description logic (ADL) formulas.

ADL is designed to support both mathematical notion and a formal english notion.
"""
from typing import Final, cast, get_type_hints, TypeVar, Generic, Callable, Type


class TycheLanguageException(Exception):
    """
    Class for detailing language exceptions.
    """
    def __init__(self, message):
        self.message = "TycheLanguageException: " + message


class RoleProbabilityDistribution:
    """
    Represents a probability distribution of contexts for a role.
    The items in the probability distribution use weights to
    represent their likelihood of being selected. The probability
    of selecting an item is fixed at 100%.
    """
    def __init__(self):
        self.values: list[tuple['TycheContext', float]] = []
        self.total_weight = 0

    def add(self, context: 'TycheContext', weight: float):
        if weight <= 0:
            raise TycheLanguageException("Value weights must be positive, not {:.3f}".format(weight))

        self.values.append((context, weight))
        self.total_weight += weight

    def __iter__(self):
        total_weight = self.total_weight
        for context, weight in self.values:
            yield context, weight / total_weight


class TycheContext:
    """
    Provides context for relating variables in formulas
    (e.g. a, b, c, etc...) to their related objects.
    Each individual may supply their own context for
    their variables and roles.
    """
    def eval_atom(self, symbol: str) -> float:
        raise TycheLanguageException("eval_atom is unimplemented for " + type(self).__name__)

    def eval_role(self, symbol: str) -> RoleProbabilityDistribution:
        raise TycheLanguageException("eval_role is unimplemented for " + type(self).__name__)


Probability = TypeVar('Probability', float, int)


class Role:
    def __init__(self, fn: Callable[[], RoleProbabilityDistribution]):
        self.fn = fn

    def __set_name__(self, owner, name):
        if not hasattr(owner, "_Tyche_roles"):
            setattr(owner, "_Tyche_roles", {})
        getattr(owner, "_Tyche_roles")[name] = self.fn

        setattr(owner, name, self.fn)

    @staticmethod
    def eval_role_of(instance: 'Individual', symbol: str) -> RoleProbabilityDistribution:
        role_fn = None
        if hasattr(instance, "_Tyche_roles"):
            roles = getattr(instance, "_Tyche_roles")
            if symbol in roles:
                role_fn = roles[symbol]

        if role_fn is None:
            raise TycheLanguageException("Unknown role {}".format(symbol))

        return role_fn(instance)


class Individual(TycheContext):
    """
    A helper class for creating TycheContext objects using class objects
    to represent the individuals, and annotations to mark fields and
    methods as atoms and roles.
    """
    def __init__(self):
        super().__init__()
        pass

    def eval_atom(self, symbol: str) -> float:
        # This is probably pretty slow, we should cache this.
        hints = get_type_hints(type(self))
        if symbol not in hints:
            raise TycheLanguageException("Unknown atom {}".format(symbol))
        if hints[symbol] != Probability:
            raise TycheLanguageException("The variable of this class, {}, is not marked as a Probability".format(symbol))

        return getattr(self, symbol)

    def eval_role(self, symbol: str) -> RoleProbabilityDistribution:
        return Role.eval_role_of(self, symbol)


class Concept:
    """
    The base class of all nodes in aleatoric description logic formulas.
    """
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

    def when(self, condition: 'Concept') -> 'ConditionalWithoutElse':
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
    concpet.necessary_for(role)
    concept.possible_for(role)
    others...
    '''


class Atom(Concept):
    """
    This class is used to represent indivisible concepts such
    as always, never, or other named concepts.
    """
    def __init__(self, symbol, *, special_symbol=False):
        if not special_symbol:
            Atom.check_atom_symbol(symbol)

        self.symbol = symbol

    @staticmethod
    def check_atom_symbol(symbol):
        """
        Checks a string contains only alphanumeric characters or underscore.
        Raises an error if the symbol is an invalid atom symbol.
        """
        if symbol is None:
            raise ValueError("Symbol cannot be None")
        if len(symbol) == 0:
            raise ValueError("Symbol cannot be an empty string")
        if not symbol[0].islower():
            raise ValueError("atom symbols must start with a lowercase letter")

        for ch in symbol[1:]:
            if ch != '_' and not ch.isalnum:
                raise ValueError("Symbol can only contain alpha-numeric or underscore characters")

    def __str__(self):
        return self.symbol

    def __repr__(self):
        return "{}(symbol={})".format(type(self).__name__, self.symbol)

    def __eq__(self, other) -> bool:
        return type(self) == type(other) and self.symbol == cast('Atom', other).symbol

    def __lt__(self, other) -> bool:
        raise TycheLanguageException("not yet implemented")

    def eval(self, context: TycheContext) -> float:
        return context.eval_atom(self.symbol)

    def normal_form(self):
        return self

    def is_equivalent(self, concept: Concept) -> bool:
        return self == concept

    def is_weaker(self, concept):
        raise TycheLanguageException("is_weaker is unimplemented for " + type(self).__name__)


class Constant(Atom):
    """
    A constant aleatoric probability, where each evaluation
    of the constant is independent.
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
    Class for representing the aleatoric ternary construct (if-then-else).
    """
    def __init__(self, condition, if_yes, if_no):
        self.condition = condition
        self.if_yes = if_yes
        self.if_no = if_no

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
    A conditional that represents the ternary (condition ? if_yes : Always).
    """
    def __init__(self, condition: Concept, if_yes: Concept):
        super().__init__(condition, if_yes, always)

    def otherwise(self, if_no: Concept) -> Conditional:
        return Conditional(self.condition, self.if_yes, if_no)


class Expectation(Concept):
    """
    class for representing the aleatoric expectation construct in the language
    """
    def __init__(self, role: str, concept: Concept):
        self.role: str = role
        self.concept: Concept = concept

    def __str__(self):
        return "(\U0001D53C_{}. {})".format(self.role, str(self.concept))

    def __repr__(self):
        return 'Expectation(role=' + repr(self.role) + ', concept=' + repr(self.concept) + ')'

    def __eq__(self, obj):
        if type(obj) != type(self):
            return False

        other: 'Expectation' = cast('Expectation', obj)
        return (self.role == other.role and
                self.concept == other.concept)

    def __lt__(self, other):
        raise TycheLanguageException("not yet implemented")

    def eval(self, context: TycheContext):
        """
        Evaluates the concept for all members of the role mapping
        from the given context to other contexts.
        """
        total_prob = 0.0
        for other_context, prob in context.eval_role(self.role):
            total_prob += prob * self.concept.eval(other_context)
        return total_prob

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
