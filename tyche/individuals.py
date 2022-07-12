"""
Provides convenience classes for setting up a model using classes.
"""
from typing import TypeVar, Callable, get_type_hints, Final, Type, cast, Generic, Optional

import numpy as np

from tyche.language import ExclusiveRoleDist, TycheLanguageException, TycheContext, Atom, Concept, Expectation, \
    Role, RoleDistributionEntries, ALWAYS, CompatibleWithConcept, CompatibleWithRole, NEVER, Constant, Given

# Marks instance variables of classes as probabilities that
# may be accessed by Tyche formulas.
from tyche.probability import uncertain_bayes_rule
from tyche.reference import SymbolReference, PropertySymbolReference, GuardedSymbolReference, FunctionSymbolReference, \
    BakedSymbolReference

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
    __accessor_store_maps: Final[dict[type, dict[type, 'TycheAccessorStore']]] = {}
    # Stores all the non-field accessor references for atoms & roles of Individual subclass types.
    __accessor_ref_maps: Final[dict[type, dict[type, dict[str, SymbolReference[AccessedValueType]]]]] = {}

    def __init__(self, type_name: str, accessors: dict[str, SymbolReference[AccessedValueType]]):
        self.type_name = type_name
        self.accessors = accessors
        self.all_symbols = set(accessors.keys())

    def get_reference(self, symbol: str) -> SymbolReference[AccessedValueType]:
        """ Returns the reference to access the given symbol. """
        try:
            return self.accessors[symbol]
        except KeyError:
            raise TycheLanguageException("Unknown {} {}".format(
                self.type_name, symbol
            ))

    def get(self, obj: any, symbol: str) -> AccessedValueType:
        """ Accesses the given symbol from the given object. """
        return self.get_reference(symbol).get(obj)

    def set(self, obj: any, symbol: str, value: AccessedValueType):
        """ Modifies the given symbol in the given object. """
        return self.get_reference(symbol).set(obj, value)

    @staticmethod
    def get_accessor_stores_map(accessor_type: type) -> dict[type, 'TycheAccessorStore']:
        """ Returns a map from Individual subclass types to their accessor stores for the given type of accessors. """
        accessor_store_maps = TycheAccessorStore.__accessor_store_maps
        if accessor_type not in accessor_store_maps:
            accessor_store_maps[accessor_type] = {}
        return accessor_store_maps[accessor_type]

    @staticmethod
    def get_accessor_ref_map(accessor_type: type) -> dict[type, dict[str, SymbolReference[AccessedValueType]]]:
        """ Returns a map from Individual subclass types to their accessor stores for the given type of accessors. """
        accessor_ref_maps = TycheAccessorStore.__accessor_ref_maps
        if accessor_type not in accessor_ref_maps:
            accessor_ref_maps[accessor_type] = {}
        return accessor_ref_maps[accessor_type]

    @staticmethod
    def get_or_populate_for(obj_type: type, accessor_type: type, field_type_hint: type) -> 'TycheAccessorStore':
        """
        Gets or populates the set of accessors associated with obj_type, and returns them.
        """
        stores_cache_map = TycheAccessorStore.get_accessor_stores_map(accessor_type)
        if obj_type in stores_cache_map:
            return stores_cache_map[obj_type]

        # Get the references populated by annotations.
        refs_map = TycheAccessorStore.get_accessor_ref_map(accessor_type)
        method_references: dict[str, SymbolReference[AccessedValueType]] = {}
        for parent_or_obj_type in obj_type.mro():
            if parent_or_obj_type in refs_map:
                method_references.update(refs_map[parent_or_obj_type])

        # Get the set of variables that can be accessed from obj_type.
        # These include the fields from parent classes automatically.
        variables: set[str] = set()
        for symbol, type_hint in get_type_hints(obj_type).items():
            if type_hint == field_type_hint:
                variables.add(symbol)
                if symbol in method_references:
                    raise TycheIndividualsException(
                        "The {} {} in type {} cannot be provided as both a variable and a method".format(
                            accessor_type.__name__, symbol, obj_type.__name__
                        ))

        # Check that all the symbol names are valid.
        for symbol_name, name_set in [("variable", variables), ("method", method_references.keys())]:
            symbol_type_name = accessor_type.__name__.capitalize()
            context = "type {}".format(obj_type.__name__)
            for name in name_set:
                Atom.check_symbol(name, symbol_name=symbol_name, symbol_type_name=symbol_type_name, context=context)

        # Store the accessors in the type object.
        var_references = {symbol: PropertySymbolReference(symbol) for symbol in variables}
        all_references = {**method_references, **var_references}
        accessors = TycheAccessorStore(accessor_type.__name__, all_references)
        stores_cache_map[obj_type] = accessors
        return accessors


class IndividualPropertyDecorator(Generic[AccessedValueType]):
    """ A decorator to mark methods as providing the value of a concept or role. """
    def __init__(self, fget: Callable[[], AccessedValueType], *, symbol: Optional[str] = None):
        self.fget = fget
        self.symbol = symbol
        self.fset: Optional[Callable[[AccessedValueType], None]] = None

    def updater(self, fset: Optional[Callable[[AccessedValueType], None]]):
        """
        This can be used as a decorator to register a function-based setter for this value.
        This is called 'updater' instead of 'setter' due to erroneous errors that the function
        names should match if the method is called 'setter'. I believe this error is to help
        people when they use @property decorators, but it erroneously gets triggered here as well.
        """
        self.fset = fset
        return fset

    def __set_name__(self, owner: type, name: str):
        ref_map = TycheAccessorStore.get_accessor_ref_map(type(self))
        if owner not in ref_map:
            ref_map[owner] = {}

        # Allow the user to override the symbol that is used.
        symbol = name if self.symbol is None else self.symbol

        # Add this symbol to the set of function symbols.
        ref_map[owner][name] = FunctionSymbolReference(symbol, self.fget, self.fset)

        # Replace this decorator object with the original function in the object.
        setattr(owner, name, self.fget)


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
        # The Given operator does nothing when evaluated at a regular individual.
        if isinstance(concept, Given):
            return self.eval(cast(Given, concept).concept)

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

    def get_concept_reference(self, symbol: str) -> BakedSymbolReference[float]:
        ref = self.concepts.get_reference(symbol)
        coerced_ref = GuardedSymbolReference(ref, self.coerce_concept_value, self.coerce_concept_value)
        return coerced_ref.bake(self)

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
            f"{type(ExclusiveRoleDist).__name__}, not {type(value).__name__}"
        )

    def get_role(self, symbol: str) -> ExclusiveRoleDist:
        value = self.roles.get(self, symbol)
        return self.coerce_role_value(value)

    def get_role_reference(self, symbol: str) -> SymbolReference[ExclusiveRoleDist]:
        ref = self.roles.get_reference(symbol)
        coerced_ref = GuardedSymbolReference(ref, self.coerce_role_value, self.coerce_role_value)
        return coerced_ref.bake(self)

    def observe(self, observation: Concept, likelihood: float = 1, learning_rate: float = 1):
        # If an expectation over a role is observed, then we can apply Bayes' rule to the role.
        if isinstance(observation, Expectation):
            expectation: Expectation = cast(Expectation, observation)
            observed_role_ref = expectation.role.eval_reference(self)
            prev_role_value = observed_role_ref.get()
            new_role_value = prev_role_value.apply_bayes_rule(expectation.concept, likelihood, learning_rate)
            observed_role_ref.set(new_role_value)

            # Propagate the observation!
            possible_matching_individuals = Expectation.reverse_observation(
                prev_role_value, expectation.concept, expectation.given, likelihood
            )
            concept_given = Given(expectation.concept, expectation.given)
            for ctx, prob in possible_matching_individuals:
                if ctx is not None:
                    ctx.observe(concept_given, likelihood, learning_rate * prob)

        else:
            # Otherwise, we can recurse through the observation.
            observation_prob = self.eval(observation)
            obs_matches_expected_prob = likelihood * observation_prob + (1 - likelihood) * (1 - observation_prob)
            if obs_matches_expected_prob <= 0:
                raise TycheIndividualsException(
                    f"The observation is impossible under this model "
                    f"({observation} with likelihood {likelihood} @ {self.name})"
                )

            child_concepts = observation.get_child_concepts_in_eval_context()
            for index, child_concept in enumerate(child_concepts):
                if isinstance(child_concept, Constant):
                    continue  # Quick skip

                obs_given_child = observation.copy_with_new_child_concept_from_eval_context(index, ALWAYS)
                obs_given_not_child = observation.copy_with_new_child_concept_from_eval_context(index, NEVER)

                child_prob = self.eval(child_concept)
                not_child_prob = 1 - child_prob
                obs_given_child_prob = self.eval(obs_given_child)
                obs_given_not_child_prob = self.eval(obs_given_not_child)

                child_true_prob = uncertain_bayes_rule(
                    child_prob, observation_prob, obs_given_child_prob, likelihood)
                child_false_prob = uncertain_bayes_rule(
                    not_child_prob, observation_prob, obs_given_not_child_prob, likelihood)

                # Corresponds to the learning_rate parameter.
                child_influence = abs(obs_given_child_prob - obs_given_not_child_prob)
                if child_influence <= 0:
                    continue

                # Corresponds to the likelihood parameter.
                if child_true_prob + child_false_prob <= 0:
                    continue
                child_likelihood = child_true_prob / (child_true_prob + child_false_prob)

                # Propagate!
                self.observe(child_concept, child_likelihood, learning_rate * child_influence)

    def to_str(self, *, detail_lvl: int = 1, indent_lvl: int = 0):
        if detail_lvl <= 0:
            return self.name

        sub_detail_lvl = detail_lvl - 1

        # Key-values of concepts.
        concept_values = [f"{sym}={self.get_concept(sym):.3f}" for sym in self.concepts.all_symbols]
        concept_values.sort()

        # We don't want to list out the entirety of the roles.
        role_values = []
        for role_symbol in self.roles.all_symbols:
            role = self.get_role(role_symbol)
            role_values.append(f"{role_symbol}={role.to_str(detail_lvl=sub_detail_lvl, indent_lvl=indent_lvl)}")
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
        return Expectation.evaluate_for_role(self._id, concept, ALWAYS)

    def eval_role(self, role: 'Role') -> ExclusiveRoleDist:
        return role.direct_eval(self)

    def get_role(self, symbol: str) -> ExclusiveRoleDist:
        self._verify_not_empty()
        return Expectation.evaluate_role_under_role(self._id, Role(symbol))

    def get_concept(self, symbol: str) -> float:
        raise TycheIndividualsException(
            f"Cannot evaluate atoms for instances of {type(self).__name__}. eval should be used instead")

    def get_concept_reference(self, symbol: str) -> SymbolReference[float]:
        raise TycheIndividualsException(
            f"Cannot evaluate mutable concepts for instances of {type(self).__name__}")

    def get_role_reference(self, symbol: str) -> SymbolReference[ExclusiveRoleDist]:
        raise TycheIndividualsException(
            f"Cannot evaluate mutable roles for instances of {type(self).__name__}")

    def observe(self, observation: 'Concept', likelihood: float = 1, learning_rate: float = 1):
        if learning_rate <= 0:
            return
        self._verify_not_empty()

        prev_id_value = self._id
        self._id = prev_id_value.apply_bayes_rule(observation, likelihood, learning_rate)

        # Propagate the observation!
        obs_is_given = isinstance(observation, Given)
        concept = observation if not obs_is_given else cast(Given, observation).concept
        given = ALWAYS if not obs_is_given else cast(Given, observation).given
        possible_matching_individuals = Expectation.reverse_observation(
            prev_id_value, concept, given, likelihood
        )
        for ctx, prob in possible_matching_individuals:
            if ctx is not None:
                ctx.observe(observation, likelihood, learning_rate * prob)

    def __iter__(self):
        """
        Yields tuples of the possible individuals that could
        represent this individual, and their probability.
        """
        self._verify_not_empty()
        for ctx, prob in self._id:
            yield ctx, prob

    def to_str(self, *, detail_lvl: int = 1, indent_lvl: int = 0):
        if detail_lvl <= 0:
            return self.name

        return f"{self.name}{self._id.to_str(detail_lvl=detail_lvl, indent_lvl=indent_lvl)}"
