"""
Provides convenience classes for setting up a model using classes.
"""
from typing import TypeVar, Callable, get_type_hints, Final, Type, cast, Generic, Optional

import numpy as np

from tyche.language import ExclusiveRoleDist, TycheLanguageException, TycheContext, Atom, Concept, Expectation, \
    Role, RoleDistributionEntries, always, CompatibleWithConcept, CompatibleWithRole

# Marks instance variables of classes as probabilities that
# may be accessed by Tyche formulas.
from tyche.reference import MutableReference, MutableVariableReference, GuardedMutableReference

TycheConceptField = TypeVar("TycheConceptField", float, int, bool)
TycheRoleField = TypeVar("TycheRoleField", bound=ExclusiveRoleDist)


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
    # Stores all the accessors for atoms & roles of Individual subclass types.
    __accessor_maps: Final[dict[type, dict[type, 'TycheAccessorStore']]] = {}
    # Stores all the accessors function names for atoms & roles of Individual subclass types.
    __accessor_function_maps: Final[dict[type, dict[type, set[str]]]] = {}

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
    def get_accessors_map(accessor_type: type) -> dict[type, 'TycheAccessorStore']:
        """ Returns a map from Individual subclass types to their accessor stores for the given type of accessors. """
        accessor_maps = TycheAccessorStore.__accessor_maps
        if accessor_type not in accessor_maps:
            accessor_maps[accessor_type] = {}
        return accessor_maps[accessor_type]

    @staticmethod
    def get_accessor_functions_map(accessor_type: type) -> dict[type, set[str]]:
        """ Returns a map from Individual subclass types to their accessor stores for the given type of accessors. """
        accessor_function_maps = TycheAccessorStore.__accessor_function_maps
        if accessor_type not in accessor_function_maps:
            accessor_function_maps[accessor_type] = {}
        return accessor_function_maps[accessor_type]

    @staticmethod
    def get_or_populate_for(obj_type: type, accessor_type: type, field_type_hint: type) -> 'TycheAccessorStore':
        """
        Gets or populates the set of accessors associated with obj_type, and returns them.
        """
        accessor_map = TycheAccessorStore.get_accessors_map(accessor_type)
        if obj_type in accessor_map:
            return accessor_map[obj_type]

        # Get the set of functions that can be called on objects of obj_type.
        # We need to get the functions of all the type's parent classes as well.
        functions: set[str] = set()
        functions_map = TycheAccessorStore.get_accessor_functions_map(accessor_type)
        for method_resolution_type in obj_type.mro():
            if method_resolution_type in functions_map:
                functions.update(functions_map[method_resolution_type])

        # Get the set of variables that can be accessed from obj_type.
        # These include the fields from parent classes automatically.
        variables: set[str] = set()
        for symbol, type_hint in get_type_hints(obj_type).items():
            if type_hint == field_type_hint:
                variables.add(symbol)
                if symbol in functions:
                    raise TycheIndividualsException(
                        "The {} {} in type {} cannot be provided as both a variable and a function".format(
                            accessor_type.__name__, symbol, obj_type.__name__
                        ))

        # Check that there are no duplicate names.
        intersection = variables.intersection(functions)
        if len(intersection) > 0:
            raise TycheIndividualsException("The symbol {} cannot be provided as both a variable and a function".format(
                list(intersection)[0]
            ))

        # Check that all the symbol names are valid.
        for symbol_name, name_set in [("variable", variables), ("method", functions)]:
            symbol_type_name = accessor_type.__name__.capitalize()
            context = "type {}".format(obj_type.__name__)
            for name in name_set:
                Atom.check_symbol(name, symbol_name=symbol_name, symbol_type_name=symbol_type_name, context=context)

        # Store the accessors in the type object.
        accessors = TycheAccessorStore(accessor_type.__name__, variables, functions)
        accessor_map[obj_type] = accessors
        return accessors


class IndividualPropertyDecorator:
    """ A decorator to mark methods as providing the value of a concept or role. """
    def __init__(self, fn: Callable[[], ExclusiveRoleDist]):
        self.fn = fn

    def __set_name__(self, owner: type, name: str):
        functions_map = TycheAccessorStore.get_accessor_functions_map(type(self))
        if owner not in functions_map:
            functions_map[owner] = set()

        # Add this symbol to the set of function symbols.
        functions_map[owner].add(name)
        # Replace this decorated function with the actual function.
        setattr(owner, name, self.fn)


class concept(IndividualPropertyDecorator):
    """
    Marks that a method provides the value of a concept for use in Tyche formulas.
    The name of the function is used as the name of the concept in formulas.
    """
    field_type_hint: Final[type] = TycheConceptField

    def __init__(self, fn: Callable[[], ExclusiveRoleDist]):
        super().__init__(fn)

    @staticmethod
    def get(obj_type: Type['Individual']) -> TycheAccessorStore:
        return TycheAccessorStore.get_or_populate_for(obj_type, concept, concept.field_type_hint)


class role(IndividualPropertyDecorator):
    """
    Marks that a method provides the value of a role for use in Tyche formulas.
    The name of the function is used as the name of the role in formulas.
    """
    field_type_hint: Final[type] = TycheRoleField

    def __init__(self, fn: Callable[[], ExclusiveRoleDist]):
        super().__init__(fn)

    @staticmethod
    def get(obj_type: Type['Individual']) -> TycheAccessorStore:
        return TycheAccessorStore.get_or_populate_for(obj_type, role, role.field_type_hint)


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

    def eval(self, concept: CompatibleWithConcept) -> float:
        return Concept.cast(concept).direct_eval(self)

    def eval_role(self, role: CompatibleWithRole) -> ExclusiveRoleDist:
        return Role.cast(role).direct_eval(self)

    @staticmethod
    def describe(obj_type: Type['Individual']) -> str:
        """ Returns a string describing the concepts and roles of the given Individual type. """
        atoms = sorted(list(Individual.get_concept_names(obj_type)))
        roles = sorted(list(Individual.get_role_names(obj_type)))
        return f"{obj_type.__name__} {{atoms={atoms}, roles={roles}}}"

    @staticmethod
    def get_concept_names(obj_type: Type['Individual']) -> set[str]:
        """ Returns all the concept names of the given Individual type. """
        return concept.get(obj_type).all_symbols

    @staticmethod
    def get_role_names(obj_type: Type['Individual']) -> set[str]:
        """ Returns all the role names of the given Individual type. """
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

    def get_concept(self, symbol: str) -> float:
        value = self.concepts.get(self, symbol)
        return self.coerce_concept_value(value)

    def get_mutable_concept(self, symbol: str) -> MutableReference[float, float]:
        ref = self.concepts.get_mutable(self, symbol)
        return GuardedMutableReference(ref, self.coerce_concept_value, self.coerce_concept_value)

    @classmethod
    def coerce_role_value(cls: type, value: any) -> ExclusiveRoleDist:
        """
        Coerces role values to only allow WeightedRoleDistribution.
        In the future, this should accept other types of role distributions.
        """
        if isinstance(value, ExclusiveRoleDist):
            return value

        raise TycheIndividualsException(
            f"Error in {cls.__name__}: Role values must be of type "
            f"WeightedRoleDistribution, not {type(value).__name__}"
        )

    def get_role(self, symbol: str) -> ExclusiveRoleDist:
        value = self.roles.get(self, symbol)
        return self.coerce_role_value(value)

    def get_mutable_role(self, symbol: str) -> MutableReference[ExclusiveRoleDist, ExclusiveRoleDist]:
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
        concept_values = [f"{sym}={self.get_concept(sym):.3f}" for sym in self.concepts.all_symbols]
        concept_values.sort()

        # We don't want to list out the entirety of the roles.
        role_values = []
        for role_symbol in self.roles.all_symbols:
            role = self.get_role(role_symbol)
            if role.is_empty():
                role_values.append(f"{role_symbol}=<empty role>")
            else:
                role_values.append(f"{role_symbol}=<role of {len(role)}>")
        role_values.sort()

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
        self._id = ExclusiveRoleDist(entries)
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

    def eval(self, concept: 'Concept') -> float:
        return Expectation.evaluate_for_role(self._id, concept, always)

    def eval_role(self, role: 'Role') -> ExclusiveRoleDist:
        return role.direct_eval(self)

    def get_role(self, symbol: str) -> ExclusiveRoleDist:
        self._verify_not_empty()
        return Expectation.evaluate_role_under_role(self._id, Role(symbol))

    def get_concept(self, symbol: str) -> float:
        raise TycheIndividualsException(
            f"Cannot evaluate atoms for instances of {type(self).__name__}. eval should be used instead")

    def get_mutable_concept(self, symbol: str) -> MutableReference[float, float]:
        raise TycheIndividualsException(
            f"Cannot evaluate mutable concepts for instances of {type(self).__name__}")

    def get_mutable_role(self, symbol: str) -> MutableReference[ExclusiveRoleDist, ExclusiveRoleDist]:
        raise TycheIndividualsException(
            f"Cannot evaluate mutable roles for instances of {type(self).__name__}")

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
