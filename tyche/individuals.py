"""
Provides convenience classes for setting up a model using classes.
"""
from typing import TypeVar, Callable, get_type_hints, Final, Type, cast, Generic, Optional

import numpy as np

from tyche.language import WeightedRoleDistribution, TycheLanguageException, TycheContext, Atom, Concept, Expectation, \
    Role, RoleDistributionEntries

# Marks instance variables of classes as probabilities that
# may be accessed by Tyche formulas.
from tyche.reference import MutableReference, MutableVariableReference, GuardedMutableReference

TycheConceptField = TypeVar("TycheConceptField", float, int, bool)
TycheRoleField = TypeVar("TycheRoleField", bound=WeightedRoleDistribution)


class TycheIndividualsException(Exception):
    """
    Class for detailing exceptions with individuals.
    """
    def __init__(self, message: str):
        self.message = "TycheIndividualsException: " + message


AccessedValueType = TypeVar("AccessedValueType")


class TycheAccessorStore(Generic[AccessedValueType]):
    """
    Stores a set of ways to access concepts or roles from a Tyche individual.
    """
    def __init__(self, type_name: str, variables: set[str], functions: set[str]):
        self.type_name = type_name
        self.variables = variables
        self.functions = functions
        self.all_symbols = variables.union(functions)

    def get(self, obj: any, symbol: str) -> AccessedValueType:
        """ Accesses the given symbol from the given object. """
        if symbol in self.variables:
            return getattr(obj, symbol)
        elif symbol in self.functions:
            return getattr(obj, symbol)()
        else:
            raise TycheLanguageException("Unknown {} {} for type {}".format(
                self.type_name, symbol, type(obj).__name__
            ))

    def get_mutable(self, obj: any, symbol: str) -> MutableReference[AccessedValueType, AccessedValueType]:
        """ Accesses the given symbol from the given object. """
        if symbol in self.variables:
            return MutableVariableReference(obj, symbol)
        elif symbol in self.functions:
            raise TycheLanguageException(
                f"The {self.type_name} {symbol} for type {type(obj).__key__} is not mutable. "
                f"It is a function, not a variable"
            )
        else:
            raise TycheLanguageException("Unknown {} {} for type {}".format(
                self.type_name, symbol, type(obj).__key__
            ))

    @staticmethod
    def get_or_populate_for(
            obj_type: type, accessors_key: str,
            type_name: str, functions_key: str,
            var_type_hint: type) -> 'TycheAccessorStore':

        """ Gets or populates the set of accessors associated with obj_type, and returns them. """
        existing_accessors = getattr(obj_type, accessors_key, None)
        if existing_accessors is not None:
            return existing_accessors

        # Find the set of function and variable symbols.
        functions: set[str] = getattr(obj_type, functions_key, set())
        variables: set[str] = set()
        for symbol, type_hint in get_type_hints(obj_type).items():
            if type_hint == var_type_hint:
                variables.add(symbol)
                if symbol in functions:
                    raise TycheIndividualsException(
                        "The {} {} in type {} cannot be provided as both a variable and a function".format(
                            type_name, symbol, obj_type.__name__
                        ))

        # Check that there are no duplicate names.
        intersection = variables.intersection(functions)
        if len(intersection) > 0:
            raise TycheIndividualsException("The symbol {} cannot be provided as both a variable and a function".format(
                list(intersection)[0]
            ))

        # Check that all the symbol names are valid.
        for symbol_name, name_set in [("variable", variables), ("method", functions)]:
            symbol_type_name = type_name.capitalize()
            context = "type {}".format(obj_type.__name__)
            for name in name_set:
                Atom.check_symbol(name, symbol_name=symbol_name, symbol_type_name=symbol_type_name, context=context)

        # Store the accessors in the type object.
        accessors = TycheAccessorStore(type_name, variables, functions)
        setattr(obj_type, accessors_key, accessors)
        return accessors


class IndividualPropertyDecorator:
    """ A decorator to mark methods as providing the value of a concept or role. """
    def __init__(self, functions_key: str, fn: Callable[[], WeightedRoleDistribution]):
        self.functions_key = functions_key
        self.fn = fn

    def __set_name__(self, owner, name):
        if not hasattr(owner, self.functions_key):
            setattr(owner, self.functions_key, set())

        # Add this symbol to the set of function symbols.
        getattr(owner, self.functions_key).add(name)
        # Replace this decorated function with the actual function.
        setattr(owner, name, self.fn)


class concept(IndividualPropertyDecorator):
    """
    Marks that a method provides the value of a concept for use in Tyche formulas.
    The name of the function is used as the name of the concept in formulas.
    """
    accessors_key: Final[str] = "_Tyche_concepts"
    functions_key: Final[str] = "_Tyche_concept_functions"
    var_type_hint: Final[type] = TycheConceptField

    def __init__(self, fn: Callable[[], WeightedRoleDistribution]):
        super().__init__(concept.functions_key, fn)

    @staticmethod
    def get(obj_type: Type['Individual']) -> TycheAccessorStore:
        return TycheAccessorStore.get_or_populate_for(
            obj_type, concept.accessors_key,
            "concept", concept.functions_key, concept.var_type_hint
        )


class role(IndividualPropertyDecorator):
    """
    Marks that a method provides the value of a role for use in Tyche formulas.
    The name of the function is used as the name of the role in formulas.
    """
    accessors_key: Final[str] = "_Tyche_roles"
    functions_key: Final[str] = "_Tyche_role_functions"
    var_type_hint: Final[type] = TycheRoleField

    def __init__(self, fn: Callable[[], WeightedRoleDistribution]):
        super().__init__(role.functions_key, fn)

    @staticmethod
    def get(obj_type: Type['Individual']) -> TycheAccessorStore:
        return TycheAccessorStore.get_or_populate_for(
            obj_type, role.accessors_key,
            "role", role.functions_key, role.var_type_hint
        )


class Individual(TycheContext):
    """
    A helper class for representing individual entities in an aleatoric knowledge base.
    These roles can be used as contexts to use to evaluate Tyche expressions within.

    The concepts and roles about the individual may be stored as instance variables,
    or as methods that supply their value.
    * Variables can be marked as concepts by giving them the TycheConcept type hint.
    * Variables can be marked as roles by giving them the TycheRole type hint.
    * Methods can be marked as concepts using the concept decorator.
    * Methods can be marked as roles using the role decorator.
    """
    name: Optional[str]
    concepts: TycheAccessorStore
    roles: TycheAccessorStore

    def __init__(self, name: Optional[str] = None):
        super().__init__()
        self.name = name
        self.concepts = concept.get(type(self))
        self.roles = role.get(type(self))

    @staticmethod
    def get_concept_names(obj_type: Type['Individual']) -> set[str]:
        return concept.get(obj_type).all_symbols

    @staticmethod
    def get_role_names(obj_type: Type['Individual']) -> set[str]:
        return role.get(obj_type).all_symbols

    @classmethod
    def coerce_concept_value(cls: type, value: any) -> float:
        """
        Coerces concept values to a float value in the range [0, 1],
        and raises errors if this is not possible.
        """
        if np.isscalar(value):
            if value < 0:
                raise TycheIndividualsException(
                    f"Error in {cls.__name__}: Concept values must be >= to 0, not {value}"
                )
            if value > 1:
                raise TycheIndividualsException(
                    f"Error in {cls.__name__}: Concept values must be <= to 1, not {value}"
                )

            return float(value)

        if isinstance(value, bool):
            return 1 if value else 0

        raise TycheIndividualsException(
            f"Error in {cls.__name__}: Concept values be of type float, int or bool, not {type(value).__name__}"
        )

    def eval_concept(self, symbol: str) -> float:
        value = self.concepts.get(self, symbol)
        return self.coerce_concept_value(value)

    def eval_mutable_concept(self, symbol: str) -> MutableReference[float, float]:
        ref = self.concepts.get_mutable(self, symbol)
        return GuardedMutableReference(ref, self.coerce_concept_value, self.coerce_concept_value)

    @classmethod
    def coerce_role_value(cls: type, value: any) -> WeightedRoleDistribution:
        """
        Coerces role values to only allow WeightedRoleDistribution.
        In the future, this should accept other types of role distributions.
        """
        if isinstance(value, WeightedRoleDistribution):
            return value

        raise TycheIndividualsException(
            f"Error in {cls.__name__}: Role values must be of type "
            f"WeightedRoleDistribution, not {type(value).__name__}"
        )

    def eval_role(self, symbol: str) -> WeightedRoleDistribution:
        value = self.roles.get(self, symbol)
        return self.coerce_role_value(value)

    def eval_mutable_role(self, symbol: str) -> MutableReference[WeightedRoleDistribution, WeightedRoleDistribution]:
        ref = self.roles.get_mutable(self, symbol)
        return GuardedMutableReference(ref, self.coerce_role_value, self.coerce_role_value)

    def observe(self, concept: Concept):
        """
        Attempts to update the beliefs of this individual based upon
        an observation of the given concept.
        """
        if isinstance(concept, Expectation):
            # If an expectation over a role is observed, then we can simply apply Bayes' rule.
            expectation: Expectation = cast(Expectation, concept)
            observed_role_ref = expectation.role.eval_mutable(self)
            curr_role_value = observed_role_ref.get()
            new_role_value = curr_role_value.apply_bayes_rule(expectation.concept)
            observed_role_ref.set(new_role_value)
        else:
            raise Exception(f"Updating of beliefs based upon observations of type {type(concept)} are not supported")

    def __str__(self):
        # Key-values of concepts.
        concept_values = [f"{sym}={self.eval_concept(sym):.3f}" for sym in self.concepts.all_symbols]

        # We don't want to list out the entirety of the roles.
        role_values = []
        for role_symbol in self.roles.all_symbols:
            role = self.eval_role(role_symbol)
            if role.is_empty():
                role_values.append(f"{role_symbol}=<empty role>")
            else:
                role_values.append(f"{role_symbol}=<role of {len(role)}>")

        name = self.name if self.name is not None else ""
        key_values = ", ".join(concept_values + role_values)
        return f"{name}({key_values})"


class IdentityIndividual(TycheContext):
    """
    Implicitly represents a role over the set of possible states that an individual could take.
    Evaluation of concepts and roles with this context will implicitly perform an expectation
    over the set of possible individuals in the id role.

    The None-individual is not supported for identity roles.
    """
    name: Optional[str]
    _id: TycheRoleField

    def __init__(self, *, name: Optional[str] = None, entries: RoleDistributionEntries = None):
        super().__init__()
        self.name = name
        self._id = WeightedRoleDistribution(entries)
        for ctx in self._id.contexts():
            if ctx is None:
                raise TycheIndividualsException("None individuals are not supported by IndividualIdentity")

    def is_empty(self) -> bool:
        """ Returns whether no possible individuals have been added. """
        return self._id.is_empty()

    def _verify_not_empty(self):
        if self.is_empty():
            raise Exception("This IndividualIdentity has not been given any possible individuals")

    def add(self, individual: TycheContext, weight: float = 1):
        """
        Adds a possible individual with the given weight. The weights
        represent the relative weight of the individual against the
        weights of other individuals. If the given individual already
        exists, this will replace their weight.

        If no weight is given, the default weight of 1 will be used.
        """
        if individual is None:
            raise TycheIndividualsException("None individuals are not supported by IndividualIdentity")

        self._id.add(individual, weight)

    def remove(self, individual: TycheContext):
        """ Removes the given individual from this identity individual. """
        return self._id.remove(individual)

    def eval_concept(self, symbol: str) -> float:
        self._verify_not_empty()
        return Expectation.evaluate_for_role(self._id, Atom(symbol))

    def eval_role(self, symbol: str) -> WeightedRoleDistribution:
        self._verify_not_empty()
        return Expectation.evaluate_role_under_role(self._id, Role(symbol))

    def eval_mutable_concept(self, symbol: str) -> MutableReference[float, float]:
        raise TycheIndividualsException(
            f"Cannot evaluate mutable concepts for instances of {type(self).__name__}"
        )

    def eval_mutable_role(self, symbol: str) -> MutableReference[WeightedRoleDistribution, WeightedRoleDistribution]:
        raise TycheIndividualsException(
            f"Cannot evaluate mutable roles for instances of {type(self).__name__}"
        )

    def observe(self, concept: Concept):
        """
        Updates the belief about what individual is the true individual
        based upon an observation of a concept.
        """
        self._verify_not_empty()

        # It may not be totally expected that this observation _only_
        # updates the id role, but ignores updates to any roles within
        # the individuals that could be updated.
        self._id = self._id.apply_bayes_rule(concept)

    def __iter__(self):
        """
        Yields tuples of the possible individuals that could
        represent this individual, and their probability.
        """
        self._verify_not_empty()
        for ctx, prob in self._id:
            yield ctx, prob

    def __str__(self):
        name = self.name if f"{self.name} " is not None else ""
        if self._id.is_empty():
            return f"{name}{{<empty {type(self).__name__}>}}"

        individual_percentages = [f"{100 * prob:.0f}%: {ctx}" for ctx, prob in self._id]
        possible_individuals = ", ".join(individual_percentages)
        return f"{name}{{{possible_individuals}}}"
