"""
The classes for representing and reasoning about aleatoric
description logic (ADL) formulas.

ADL is designed to support both mathematical notion and a formal english notion.
"""
from typing import Final, cast

'''
Refactor
Simpler class system. No need for abstract classes of formula.
The base classes are: No, Yes, Concept, Variable, Conditional, Expectation, LeastFixedPoint.
Each formula class must provide the methods 
__str__, 
__repr__, 
eval (acts on lambdas, returns a lambda)
normal_form (produces a normal form of the formula)
__equal__ (for an ordering of formulas)
__lt__ (for ordering)

'''


class TycheFormulaException(Exception):
    """
    Class for detailing ADL exceptions
    """
    def __init__(self, message):
        self.message = "TycheLanguageException: " + message


class Concept:
    """
    The base class of all nodes in aleatoric description logic formulas.
    """
    def get_child_concepts(self) -> list['Concept']:
        """
        Returns the child concepts of this concept.
        """
        raise TycheFormulaException("get_child_concepts is unimplemented for " + type(self).__name__)

    def __str__(self) -> str:
        """
        gives a compact string representation of the structure of the formula
        in terms of primitive operators
        """
        raise TycheFormulaException("__str__ is unimplemented for " + type(self).__name__)

    def __repr__(self) -> str:
        """
        gives the constructor string of the object
        """
        raise TycheFormulaException("__repr__ is unimplemented for " + type(self).__name__)

    # maybe also include a function for giving an optimised inline string representation of a formula.

    def __eq__(self, other: 'Concept') -> bool:
        """
        return true if formulas are identical
        """
        raise TycheFormulaException("__eq__ is unimplemented for " + type(self).__name__)

    def __lt__(self, other: 'Concept') -> bool:
        """
        establishes a syntactic ordering over formula
        """
        raise TycheFormulaException("__lt__ is unimplemented for " + type(self).__name__)

    def eval(self, concepts, roles) -> float:
        """
        returns the probability of this concept,
        given the lambda evaluation of roles and concepts.
        It is assumed concepts is a function that
        maps concept symbols to probabilities,
        and roles is a function that maps role symbols
        to probability distributions of tuples of concept functions and role functions
        """
        raise TycheFormulaException("eval is unimplemented for " + type(self).__name__)

    def normal_form(self) -> 'Concept':
        """
        Returns the tree normal form of the formula, where atoms are ordered alphabetically.
        """
        raise TycheFormulaException("normal_form is unimplemented for " + type(self).__name__)

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
        raise TycheFormulaException("is_weaker is unimplemented for " + type(self).__name__)

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
    def __init__(self, symbol):
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
        raise TycheFormulaException("this requires some more thought")

    def eval(self, concepts, roles) -> float:
        return concepts(self.symbol)

    def normal_form(self):
        return self

    def is_equivalent(self, concept: Concept) -> bool:
        return self == concept

    def is_weaker(self, concept):
        raise TycheFormulaException("is_weaker is unimplemented for " + type(self).__name__)


class Constant(Atom):
    """
    A constant aleatoric probability, where each evaluation
    of the constant is independent.
    """
    def __init__(self, symbol: str, probability: float):
        super().__init__(symbol)
        self.probability = probability

    def eval(self, concepts, roles) -> float:
        return self.probability


always: Final[Constant] = Constant("always", 1)
never: Final[Constant] = Constant("never", 0)


class Conditional(Concept):
    """
    Class for representing the aleatoric ternary construct (if-then-else).
    """
    def __init__(self, condition, if_yes, if_no):
        self.condition = condition
        self.if_yes = if_yes
        self.if_no = if_no

    def __str__(self):
        return "({} ? {} : {})".format(str(self.condition), str(self.if_yes), str(self.if_no))

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
        raise TycheFormulaException("this requires some more thought")

    def eval(self, concepts, roles):
        cond = self.condition.eval(concepts, roles)
        return cond * self.if_yes.eval(concepts, roles) + \
               (1 - cond) * self.if_no.eval(concepts, roles)

    def normal_form(self):
        """
        Returns the tree normal form of the conditional,
        by recursively calling normal for on sub elements.
        """
        pass  # to come. Long hack

    def is_weaker(self, concept):
        pass


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
    class for representng the aleatoric expectation  construct in the language
    """
    def __init__(self, role, concept):
        self.role = role  # role object
        self.concept = concept  # concept object

    def __str__(self):
        return str(self.role) + '.' + self.concept

    def __repr__(self):
        return 'Expectation(role=' + repr(self.role) + ', concept=' + repr(self.concept) + ')'

    def __eq__(self, obj):
        if type(obj) != type(self):
            return False

        other: 'Expectation' = cast('Expectation', obj)
        return (self.role == other.role and
                self.concept == other.concept)

    def __lt__(self, other):
        raise TycheFormulaException("this requires some more thought")

    def eval(self, concepts, roles):
        """
        Complex one, need to extract a distribution of roles:
        """
        dist = roles(self.role)  # dist is a dictionary mapping (concepts, roles) lambda to probabilities
        prob = 0.0
        for (c, r) in dist.keys:
            prob = prob + dist[(c, r)] * self.concept.eval(c, r)
        return prob

    def normal_form(self):
        """
        Returns the tree normal form of the conditional,
        by recursively calling normal for on sub elements.
        """
        pass

    def is_equivalent(self, concept):
        pass

    def is_weaker(self, concept):
        pass


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
            raise TycheFormulaException("The variable {} is not linear in {}".format(variable, concept))

    def __str__(self):
        """
        Use X as the fixed point quantifier,
        if least and greatest not relavant?
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
        return self.variable == other.variable and \
               self.concept == other.concept

    def __lt__(self, other):
        raise TycheFormulaException("this requires some more thought")

    def eval(self, concepts, roles):
        """
        Complex one, needs iteration or equation solving.
        """
        pass

    def normal_form(self):
        """
        Returns the tree normal form of the conditional,
        by recursively calling normal for on sub elements.
        """
        pass

    def is_equivalent(self, concept):
        pass

    def is_weaker(self, concept):
        pass

    @staticmethod
    def is_linear(variable, concept):
        """
        class method to test whether variable is linear in concept
        """
        pass


class Role(Atom):
    """
    A class for representing all ADL roles.
    Abstract class laying out the methods.

    We currently just use atomic roles.
    Dynamic roles will be realised as abbreviations using complex concepts..
    """
    def __init__(self, symbol):
        """
        Creates an atomic concept, with the symbol symbol
        symbol should be an alpha-numeric+underscore string, starting with a lower case letter.
        a '.' is prepended to the symbol to distinguish it from a concept.
        """
        super().__init__(symbol)
