"""
This module contains classes used to construct ontological knowledge bases
of individuals, the probabilistic beliefs about them (concepts), and the
probabilistic relationships between them (roles). These belief models
can be used as contexts to evaluate aleatoric description logic (ADL)
sentences. This module also contains many learning strategies that may
be used to update your belief models based upon ADL observations.
"""
from typing import TypeVar, Callable, get_type_hints, Final, Type, cast, Generic, Optional

import numpy as np

from tyche.language import ExclusiveRoleDist, TycheLanguageException, TycheContext, Concept, ADLNode, Expectation, \
    Role, RoleDistributionEntries, ALWAYS, CompatibleWithADLNode, CompatibleWithRole, NEVER, Constant, Given, \
    ReferenceBackedRole, RoleDist

from tyche.probability import uncertain_bayes_rule
from tyche.references import SymbolReference, FieldSymbolReference, GuardedSymbolReference, FunctionSymbolReference, \
    BakedSymbolReference


TycheConceptValue = TypeVar("TycheConceptValue", float, int, bool)
TycheRoleValue = TypeVar("TycheRoleValue", bound=RoleDist)

# Marks instance variables of classes as probabilities that
# may be accessed by Tyche formulas.
TycheConceptField = TypeVar("TycheConceptField", float, int, bool)
TycheRoleField = TypeVar("TycheRoleField", bound=RoleDist)


class TycheIndividualsException(Exception):
    """
    An exception type that is thrown when errors occur in
    the construction or use of individuals.
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

    def contains(self, symbol: str) -> bool:
        """ Returns whether this store contains a reference to the given symbol. """
        return symbol in self.accessors

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
    def get_or_populate_for(
            obj_type: type, accessor_type: type, field_type_hint: type, type_name: str
    ) -> 'TycheAccessorStore':
        """
        Gets or populates the set of accessors associated with obj_type, and returns them.
        """
        stores_cache_map = TycheAccessorStore.get_accessor_stores_map(accessor_type)
        if obj_type in stores_cache_map:
            return stores_cache_map[obj_type]

        # Get the references populated by annotations.
        refs_map = TycheAccessorStore.get_accessor_ref_map(accessor_type)
        method_references: dict[str, SymbolReference[AccessedValueType]] = {}
        for parent_or_obj_type in reversed(obj_type.mro()):
            if parent_or_obj_type in refs_map:
                method_references.update(refs_map[parent_or_obj_type])

        # Get the set of variables that can be accessed from obj_type.
        # These include the fields from parent classes automatically.
        fields: set[str] = set()
        for symbol, type_hint in get_type_hints(obj_type).items():
            if type_hint == field_type_hint:
                fields.add(symbol)
                if symbol in method_references:
                    raise TycheIndividualsException(
                        "The {} {} in type {} cannot be provided as both a variable and a method".format(
                            type_name, symbol, obj_type.__name__
                        ))

        # Check that all the symbol names are valid.
        for symbol_name, name_set in [("field", fields), ("method", method_references.keys())]:
            symbol_type_name = type_name.capitalize()
            context = "type {}".format(obj_type.__name__)
            for name in name_set:
                Concept.check_symbol(name, symbol_name=symbol_name, symbol_type_name=symbol_type_name, context=context)

        # Store the accessors in the type object.
        var_references = {symbol: FieldSymbolReference(symbol) for symbol in fields}
        all_references = {**method_references, **var_references}
        accessors = TycheAccessorStore(type_name, all_references)
        stores_cache_map[obj_type] = accessors
        return accessors


class ConceptFunctionSymbolReference(FunctionSymbolReference):
    """ Represents a reference to a concept, with additional information about the concept attached. """
    def __init__(
            self, symbol: str,
            fget: Callable[[any], TycheConceptValue],
            fset: Optional[Callable[[any, TycheConceptValue], None]] = None,
            learning_strat: Optional['ConceptLearningStrategy'] = None
    ):
        super().__init__(symbol, fget, fset)
        self.learning_strat = learning_strat


class RoleFunctionSymbolReference(FunctionSymbolReference):
    """ Represents a reference to a concept, with additional information about the concept attached. """
    def __init__(
            self, symbol: str,
            fget: Callable[[any], TycheRoleValue],
            fset: Optional[Callable[[any, TycheRoleValue], None]] = None,
            learning_strat: Optional['RoleLearningStrategy'] = None
    ):
        super().__init__(symbol, fget, fset)
        self.learning_strat = learning_strat


LearningStrategyType = TypeVar("LearningStrategyType", bound='LearningStrategy')
SelfType_IndividualPropertyDecorator = TypeVar(
    "SelfType_IndividualPropertyDecorator", bound="IndividualPropertyDecorator")


class IndividualPropertyDecorator(Generic[AccessedValueType, LearningStrategyType]):
    """
    A decorator to mark methods as providing the value of a concept or role,
    and to provide additional metadata or learning functions.
    """
    def __init__(
            self: SelfType_IndividualPropertyDecorator,
            type_name: str,
            fget: Callable[['Individual'], AccessedValueType], *,
            symbol: Optional[str] = None):

        self.type_name = type_name
        self.custom_symbol: Optional[str] = symbol
        self.symbol: str = symbol if symbol is not None else "this symbol"
        self.fget: Callable[['Individual'], AccessedValueType] = fget
        self.fset: Optional[Callable[['Individual', AccessedValueType], None]] = None
        self.learning_strat: Optional[LearningStrategyType] = None

    def __call__(self, instance: Optional['Individual'] = None) -> AccessedValueType:
        if instance is None:
            raise TycheIndividualsException(f"Cannot access {self.symbol} without an instance")
        return self.fget(instance)

    def learning_func(
            self, learning_strat: Optional[LearningStrategyType] = None
    ) -> Callable[
        [Callable[['Individual', AccessedValueType], None]],
        Callable[['Individual', AccessedValueType], None]
    ]:
        """
        This can be used as a decorator to register a method-based setter for this symbol.
        """
        def decorator(
                fset: Callable[['Individual', AccessedValueType], None]
        ) -> Callable[['Individual', AccessedValueType], None]:

            if self.fset is not None or self.learning_strat is not None:
                raise TycheIndividualsException(f"{self.symbol} already has a learning function set")
            self.fset = fset
            self.learning_strat = learning_strat

            # Try to detect a potentially common error that is otherwise hard to spot.
            if self.fset is not None and self.fget.__name__ is not None and self.fget.__name__ == self.fset.__name__:
                raise TycheIndividualsException(
                    f"The name of the {self.fget.__name__} {self.type_name}'s getter and learning "
                    f"functions must not be the same")

            return fset

        return decorator

    def _create_symbol_reference(self) -> 'FunctionSymbolReference':
        return FunctionSymbolReference(
            self.symbol,
            cast(Callable[[object], AccessedValueType], self.fget),
            self.fset
        )

    def __set_name__(self, owner: type, name: str):
        ref_map = TycheAccessorStore.get_accessor_ref_map(type(self))
        if owner not in ref_map:
            ref_map[owner] = {}

        # Allow the user to override the symbol that is used.
        self.symbol = name if self.custom_symbol is None else self.custom_symbol

        # Add this symbol to the set of function symbols.
        ref_map[owner][self.symbol] = self._create_symbol_reference()

        # Replace this decorator object with the original get function in the class.
        setattr(owner, name, self.fget)


class TycheConceptDecorator(IndividualPropertyDecorator[TycheConceptValue, 'ConceptLearningStrategy']):
    """
    Marks that a method provides the value of a concept for use in Tyche formulas.
    The name of the function is used as the name of the concept in formulas.
    """
    field_type_hint: Final[type] = TycheConceptField

    def __init__(
            self: SelfType_IndividualPropertyDecorator,
            fn: Callable[[], TycheConceptValue],
            *, symbol: Optional[str] = None):

        super().__init__("concept", fn, symbol=symbol)

    def _create_symbol_reference(self) -> 'FunctionSymbolReference':
        return ConceptFunctionSymbolReference(
            self.symbol, self.fget, self.fset, learning_strat=self.learning_strat
        )

    @staticmethod
    def get(obj_type: Type['Individual']) -> TycheAccessorStore:
        return TycheAccessorStore.get_or_populate_for(
            obj_type, TycheConceptDecorator, TycheConceptDecorator.field_type_hint, "concept"
        )


class TycheRoleDecorator(IndividualPropertyDecorator[TycheRoleValue, 'RoleLearningStrategy']):
    """
    Marks that a method provides the value of a role for use in Tyche formulas.
    The name of the function is used as the name of the role in formulas.
    """
    field_type_hint: Final[type] = TycheRoleField

    def __init__(
            self: SelfType_IndividualPropertyDecorator,
            fn: Callable[[], TycheRoleValue],
            *, symbol: Optional[str] = None):

        super().__init__("role", fn, symbol=symbol)

    def _create_symbol_reference(self) -> 'FunctionSymbolReference':
        return RoleFunctionSymbolReference(
            self.symbol, self.fget, self.fset, learning_strat=self.learning_strat
        )

    @staticmethod
    def get(obj_type: Type['Individual']) -> TycheAccessorStore:
        return TycheAccessorStore.get_or_populate_for(
            obj_type, TycheRoleDecorator, TycheRoleDecorator.field_type_hint, "role"
        )


def concept(
        *, symbol: Optional[str] = None
) -> Callable[[Callable[['Individual'], TycheConceptValue]], TycheConceptDecorator]:
    """
    Registers a method as supplying the value of a concept for the evaluation of Tyche expressions.
    """
    def annotator(inner_fn: Callable[[], TycheConceptValue]):
        return TycheConceptDecorator(inner_fn, symbol=symbol)

    return annotator


def role(
        *, symbol: Optional[str] = None
) -> Callable[[Callable[[], TycheRoleValue]], TycheRoleDecorator]:
    """
    Registers a method as supplying the value of a role for the evaluation of Tyche expressions.
    """
    def annotator(inner_fn: Callable[[], TycheRoleValue]):
        return TycheRoleDecorator(inner_fn, symbol=symbol)

    return annotator


SelfType_LearningStrategy = TypeVar("SelfType_LearningStrategy", bound="LearningStrategy")


class LearningStrategy:
    """ Applies changes to individuals to update them based upon observations. """
    def __init__(self: SelfType_LearningStrategy):
        pass

    def init_for_new_usage(self) -> SelfType_LearningStrategy:
        """
        If a learning strategy requires per-reference per-individual state, then this can
        clone the learning strategy for each new individual and reference. Otherwise, the
        state of this learning strategy will be shared for every individual and reference
        that uses it (which is fine for stateless learning strategies).
        """
        return self


class ConceptLearningStrategy(LearningStrategy):
    """
    Applies changes to individuals to update them based upon observations of concepts.
    """
    def apply(
            self: SelfType_LearningStrategy,
            individual: TycheContext,
            concept_ref: ConceptFunctionSymbolReference,
            observation: ADLNode,
            likelihood: float,
            learning_rate: float):
        """
        Modifies the individual to learn the given concept from the given observation of it.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement apply")


class RoleLearningStrategy(LearningStrategy):
    """
    Applies changes to individuals to update them based upon observations over roles.
    """
    def apply(
            self: SelfType_LearningStrategy,
            individual: TycheContext,
            role_ref: RoleFunctionSymbolReference,
            observation: ADLNode,
            likelihood: float,
            learning_rate: float):
        """
        Modifies the individual to learn the given concept from the given observation of it.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement apply")


class DirectConceptLearningStrategy(ConceptLearningStrategy):
    """
    The most basic learning strategy that simply updates concepts
    to the values they were observed as. A learning rate can be
    used to limit the changes to the concept's value, to stop a
    single true observation marking the concept always true.
    """
    def __init__(self, learning_rate: float = 1):
        super().__init__()
        if learning_rate <= 0:
            raise TycheIndividualsException(
                f"The learning rate must be greater than 0, not {learning_rate:.3f}")
        if learning_rate > 1:
            raise TycheIndividualsException(
                f"The learning rate must be less than or equal to 1, not {learning_rate:.3f}")

        self.learning_rate = learning_rate

    def apply(
            self: SelfType_LearningStrategy,
            individual: TycheContext,
            concept_ref: ConceptFunctionSymbolReference,
            observation: ADLNode,
            likelihood: float,
            learning_rate: float):

        # Limit the learning by the learning rate applied for the strategy.
        learning_rate *= self.learning_rate

        # Limit the learning by the confidence in the new value of the concept.
        # If likelihood is 50%, then that is useless to tell us the value of the concept.
        confidence = abs(likelihood - 0.5) * 2
        learning_rate *= confidence

        # Calculate the weighted sum of the current and new probabilities.
        curr_prob = concept_ref.get(individual)
        new_value = likelihood * learning_rate + curr_prob * (1 - learning_rate)

        # Update the individual!
        concept_ref.set(individual, new_value)

    def __str__(self):
        return f"{type(self).__name__}(learning_rate={self.learning_rate:.2f})"

    def __repr__(self):
        return f"{type(self).__name__}(learning_rate={self.learning_rate})"


class StatisticalConceptLearningStrategy(ConceptLearningStrategy):
    """
    This learning strategy accumulates a running mean of observations
    about a concept, and uses it to learn the value of the concept.
    An initial value bias can be used to favour the initial value of
    the concept until enough observations have been made.

    If an initial value is not supplied on construction, then the initial
    value is considered to be the value of the concept when the first
    observation is made that calls this learning strategy.
    """
    def __init__(self, initial_value_weight: float = 1,
                 *, decay_rate: float = 1, decay_rate_for_decay_rate: float = 1):

        super().__init__()
        if initial_value_weight < 0:
            raise TycheIndividualsException(
                f"The initial value weight must be greater than or equal to 0, not {initial_value_weight:.3f}")
        if decay_rate < 0:
            raise TycheIndividualsException(
                f"The decay rate must be greater than or equal to 0, not {decay_rate:.3f}")
        if decay_rate_for_decay_rate < 0:
            raise TycheIndividualsException(
                f"The decay rate of the decay rate must be greater than or equal to 0, "
                f"not {decay_rate_for_decay_rate:.3f}")

        self.initial_value_weight = initial_value_weight
        self.decay_rate = decay_rate
        self.decay_rate_for_decay_rate = decay_rate_for_decay_rate
        self.initial_value: Optional[float] = None
        self.running_learning_rate_sum: float = 0.0
        self.running_likelihood_sum: float = 0.0

    def init_for_new_usage(self) -> SelfType_LearningStrategy:
        """
        This is required so that the state of each reference that uses an instance
        of this learning strategy is kept separate.
        """
        return type(self)(
            self.initial_value_weight,
            decay_rate=self.decay_rate,
            decay_rate_for_decay_rate=self.decay_rate_for_decay_rate
        )

    def apply(
            self: SelfType_LearningStrategy,
            individual: TycheContext,
            concept_ref: ConceptFunctionSymbolReference,
            observation: ADLNode,
            likelihood: float,
            learning_rate: float):

        # Decay old observations over time.
        self.running_learning_rate_sum *= self.decay_rate
        self.running_likelihood_sum *= self.decay_rate

        # Decay the decay rate over time.
        self.decay_rate = 1 - (1 - self.decay_rate) * self.decay_rate_for_decay_rate

        # If we were not given an initial value on construction, read it now.
        if self.initial_value is None:
            self.initial_value = concept_ref.get(individual)
            self.running_learning_rate_sum: float = self.initial_value_weight
            self.running_likelihood_sum: float = self.initial_value * self.initial_value_weight
        else:
            # Update the running totals.
            self.running_learning_rate_sum += learning_rate
            self.running_likelihood_sum += likelihood * learning_rate

        # Calculate the running mean observed value of the concept.
        if self.running_learning_rate_sum <= 0:
            return
        concept_ref.set(individual, self.running_likelihood_sum / self.running_learning_rate_sum)

    def __str__(self):
        return f"{type(self).__name__}" \
               f"(initial_value_weight={self.initial_value_weight:.3f}, decay_rate={self.decay_rate:.3f})"


class BayesRuleLearningStrategy(RoleLearningStrategy):
    """
    TODO
    """
    def __init__(self, learning_rate: float = 1.0):
        super().__init__()
        self.learning_rate = learning_rate

    def apply(
            self: SelfType_LearningStrategy,
            individual: TycheContext,
            role_ref: RoleFunctionSymbolReference,
            observation: ADLNode,
            likelihood: float,
            learning_rate: float):

        if isinstance(observation, Expectation):
            # Update the role using Bayes' rule.
            expectation = cast(Expectation, observation)
            curr_role_value = role_ref.get(individual)
            new_role_value = curr_role_value.apply_bayes_rule(
                Given(expectation.eval_node, expectation.given_node),
                likelihood, learning_rate * self.learning_rate
            )
            role_ref.set(individual, new_role_value)


class StatisticalRoleLearningStrategy(RoleLearningStrategy):
    """
    TODO
    """
    def __init__(self, initial_value_weight: float = 1,
                 *, decay_rate: float = 1, decay_rate_for_decay_rate: float = 1):

        super().__init__()
        if initial_value_weight < 0:
            raise TycheIndividualsException(
                f"The initial value weight must be greater than or equal to 0, not {initial_value_weight:.3f}")
        if decay_rate < 0:
            raise TycheIndividualsException(
                f"The decay rate must be greater than or equal to 0, not {decay_rate:.3f}")
        if decay_rate_for_decay_rate < 0:
            raise TycheIndividualsException(
                f"The decay rate of the decay rate must be greater than or equal to 0, "
                f"not {decay_rate_for_decay_rate:.3f}")

        self.initial_value_weight = initial_value_weight
        self.decay_rate = decay_rate
        self.decay_rate_for_decay_rate = decay_rate_for_decay_rate
        self.initial_value: Optional[RoleDist] = None
        self.running_learning_rate_sums: dict[Optional[TycheContext], float] = {}
        self.running_likelihood_sums: dict[Optional[TycheContext], float] = {}

    def init_for_new_usage(self) -> SelfType_LearningStrategy:
        """
        This is required so that the state of each reference that uses an instance
        of this learning strategy is kept separate.
        """
        return type(self)(
            self.initial_value_weight,
            decay_rate=self.decay_rate,
            decay_rate_for_decay_rate=self.decay_rate_for_decay_rate
        )

    def apply(
            self: SelfType_LearningStrategy,
            individual: TycheContext,
            role_ref: RoleFunctionSymbolReference,
            observation: ADLNode,
            likelihood: float,
            learning_rate: float):

        # We only know how to deal with Expectation nodes here.
        if not isinstance(observation, Expectation):
            return
        expectation = cast(Expectation, observation)

        # Decay old observations over time.
        for ctx in self.running_likelihood_sums.keys():
            self.running_learning_rate_sums[ctx] *= self.decay_rate
            self.running_likelihood_sums[ctx] *= self.decay_rate

        # Decay the decay rate over time.
        self.decay_rate = 1 - (1 - self.decay_rate) * self.decay_rate_for_decay_rate

        # If we were not given an initial value on construction, read it now.
        curr_role_value = role_ref.get(individual)
        if self.initial_value is None:
            self.initial_value = curr_role_value
            for ctx, prob in curr_role_value:
                self.running_learning_rate_sums[ctx] = self.initial_value_weight
                self.running_likelihood_sums[ctx] = prob * self.initial_value_weight
        else:
            # Apply Bayes' rule to update the current value of the role.
            bayes_role_value = curr_role_value.apply_bayes_rule(
                Given(expectation.eval_node, expectation.given_node),
                likelihood, 1.0
            )

            # Update the running totals.
            for ctx, prob in bayes_role_value:
                if ctx not in self.running_likelihood_sums:
                    self.running_learning_rate_sums[ctx] = 0
                    self.running_likelihood_sums[ctx] = 0

                self.running_learning_rate_sums[ctx] += learning_rate
                self.running_likelihood_sums[ctx] += prob * learning_rate

        # Calculate the running mean observed value of the role.
        new_role_value = ExclusiveRoleDist()
        for ctx in self.running_likelihood_sums.keys():
            ctx_learning_rate_sum = self.running_learning_rate_sums[ctx]
            ctx_prob_sum = self.running_likelihood_sums[ctx]
            weight = ctx_prob_sum / ctx_learning_rate_sum
            if weight > 0:
                new_role_value.add(ctx, weight)

        if new_role_value.is_empty():
            return
        role_ref.set(individual, new_role_value)

    def __str__(self):
        return f"{type(self).__name__}" \
               f"(initial_value_weight={self.initial_value_weight:.3f}, decay_rate={self.decay_rate:.3f})"


class Individual(TycheContext):
    """
    A helper class for representing individual entities in an aleatoric knowledge base.
    These roles can be used as contexts to use to evaluate Tyche expressions within.

    The concepts and roles about the individual may be stored as instance variables,
    or as methods that supply their value.
    * Fields can be marked as concepts by giving them the TycheConcept type hint.
    * Fields can be marked as roles by giving them the TycheRole type hint.
    * Methods can be marked as concepts using the @concept decorator.
    * Methods can be marked as roles using the @role decorator.
    """
    name: Optional[str]
    concepts: TycheAccessorStore
    roles: TycheAccessorStore
    concept_learning_strats: dict[str, ConceptLearningStrategy]
    role_learning_strats: dict[str, RoleLearningStrategy]

    def __init__(self, name: Optional[str] = None):
        super().__init__()
        self.name = name
        self.concepts = TycheConceptDecorator.get(type(self))
        self.roles = TycheRoleDecorator.get(type(self))

        # Initialise all the learning strategies for this individual.
        self.concept_learning_strats = {}
        for symbol, ref in self.concepts.accessors.items():
            if not isinstance(ref, ConceptFunctionSymbolReference):
                continue

            concept_ref = cast(ConceptFunctionSymbolReference, ref)
            if concept_ref.learning_strat is not None:
                self.concept_learning_strats[symbol] = concept_ref.learning_strat.init_for_new_usage()

        self.role_learning_strats = {}
        for symbol, ref in self.roles.accessors.items():
            if not isinstance(ref, RoleFunctionSymbolReference):
                continue

            role_ref = cast(RoleFunctionSymbolReference, ref)
            if role_ref.learning_strat is not None:
                self.role_learning_strats[symbol] = role_ref.learning_strat.init_for_new_usage()

    def eval(self, node: CompatibleWithADLNode) -> float:
        # The Given operator does nothing when evaluated on a regular individual,
        # as the given and the sentence within the node are independent.
        if isinstance(node, Given):
            return self.eval(cast(Given, node).node)

        return ADLNode.cast(node).direct_eval(self)

    def eval_role(self, role: CompatibleWithRole) -> RoleDist:
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
        return TycheConceptDecorator.get(obj_type).all_symbols

    @staticmethod
    def get_role_names(obj_type: Type['Individual']) -> set[str]:
        """ Returns all the role names of the given Individual type. """
        return TycheRoleDecorator.get(obj_type).all_symbols

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
    def coerce_role_value(cls: type, value: any) -> RoleDist:
        """
        Coerces role values to only allow WeightedRoleDistribution.
        In the future, this should accept other types of role distributions.
        """
        if isinstance(value, RoleDist):
            return value

        raise TycheIndividualsException(
            f"Error in {cls.__name__}: Role values must be of type "
            f"{type(RoleDist).__name__}, not {type(value).__name__}"
        )

    def get_role(self, symbol: str) -> RoleDist:
        value = self.roles.get(self, symbol)
        return self.coerce_role_value(value)

    def get_role_reference(self, symbol: str) -> SymbolReference[RoleDist]:
        ref = self.roles.get_reference(symbol)
        coerced_ref = GuardedSymbolReference(ref, self.coerce_role_value, self.coerce_role_value)
        return coerced_ref.bake(self)

    def _observe_expectation(self, expectation: Expectation, likelihood: float, learning_rate: float):
        """
        Applies role learning to the role, and propagates the observation.
        """
        # Apply any learning strategy that may be present.
        symbol = expectation.role.symbol
        prev_role_value = self.get_role(symbol)
        if symbol in self.role_learning_strats:
            ref = cast(RoleFunctionSymbolReference, self.roles.get_reference(symbol))
            self.role_learning_strats[symbol].apply(self, ref, expectation, likelihood, learning_rate)

        # Propagate the observation!
        possible_matching_individuals = prev_role_value.reverse_expectation_learning_params(
            expectation.eval_node, expectation.given_node, likelihood
        )
        concept_given = Given(expectation.eval_node, expectation.given_node)
        for ctx, child_likelihood, child_influence in possible_matching_individuals:
            if child_influence > 0:
                ctx.observe(concept_given, child_likelihood, learning_rate * child_influence)

    def _observe_atom(self, node: Concept, likelihood: float, learning_rate: float):
        """
        If a learning strategy is set for the observed concept, this applies it.
        """
        if node.symbol in self.concept_learning_strats:
            ref = cast(ConceptFunctionSymbolReference, self.concepts.get_reference(node.symbol))
            self.concept_learning_strats[node.symbol].apply(self, ref, node, likelihood, learning_rate)

    def _observe_child_nodes(self, observation: ADLNode, likelihood: float, learning_rate: float):
        """
        Propagates the observation to its child nodes.
        e.g. observe (A and B) -> observe A and observe B
        """
        # Loop through all the child nodes of the observation.
        child_nodes = observation.get_child_nodes_in_eval_context()
        for index, child_node in enumerate(child_nodes):
            if isinstance(child_node, Constant):
                continue  # Quick skip

            # We have to calculate this within the loop, as the loop updates the model.
            observation_prob = self.eval(observation)
            obs_matches_expected_prob = likelihood * observation_prob + (1 - likelihood) * (1 - observation_prob)
            if obs_matches_expected_prob <= 0:
                raise TycheIndividualsException(
                    f"The observation is impossible under this model "
                    f"({observation} with likelihood {likelihood} @ {self.name})"
                )

            obs_given_child = observation.copy_with_new_child_node_from_eval_context(index, ALWAYS)
            obs_given_not_child = observation.copy_with_new_child_node_from_eval_context(index, NEVER)

            child_prob = self.eval(child_node)
            obs_given_child_prob = self.eval(obs_given_child)
            obs_given_not_child_prob = self.eval(obs_given_not_child)

            child_likelihood = uncertain_bayes_rule(
                child_prob, observation_prob, obs_given_child_prob, likelihood)

            # Corresponds to the learning_rate parameter.
            child_influence = abs(obs_given_child_prob - obs_given_not_child_prob)
            if child_influence <= 0:
                continue

            # Propagate!
            self.observe(child_node, child_likelihood, learning_rate * child_influence)

    def observe(self, observation: ADLNode, likelihood: float = 1, learning_rate: float = 1):
        """
        Learns the roles and concepts within this individual based upon the observation,
        and propagate the effects of the observation node to its child nodes.
        """
        if isinstance(observation, Expectation):
            self._observe_expectation(cast(Expectation, observation), likelihood, learning_rate)
        elif isinstance(observation, Concept):
            self._observe_atom(cast(Concept, observation), likelihood, learning_rate)
        elif isinstance(observation, Given):
            self.observe(cast(Given, observation).node, likelihood, learning_rate)
        else:
            self._observe_child_nodes(observation, likelihood, learning_rate)

    def to_str(self, *, detail_lvl: int = 1, indent_lvl: int = 0):
        if detail_lvl <= 0:
            return self.name if self.name is not None else f"<{type(self).__name__}>"

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
    learning_strat: Optional[RoleLearningStrategy]
    id_role_value: ExclusiveRoleDist
    id_role_ref: FieldSymbolReference = FieldSymbolReference("<id>", field_name="id_role_value")
    id_role: Role = ReferenceBackedRole(id_role_ref)

    def __init__(
            self,
            id_role_value: Optional[ExclusiveRoleDist] = None, *,
            name: Optional[str] = None,
            entries: RoleDistributionEntries = None,
            learning_strat: Optional[RoleLearningStrategy] = BayesRuleLearningStrategy()):

        super().__init__()
        self.name = name
        self.learning_strat = None if learning_strat is None else learning_strat.init_for_new_usage()

        if id_role_value is not None and entries is not None:
            raise TycheIndividualsException(
                "An id_role_value distribution and a set of entries for the id role cannot both be supplied at once")

        self.id_role_value = id_role_value if id_role_value is not None else ExclusiveRoleDist(entries)
        for ctx in self.id_role_value.contexts():
            if ctx is None:
                raise TycheIndividualsException("None individuals are not supported by IndividualIdentity")

    def is_empty(self) -> bool:
        """ Returns whether no possible individuals have been added. """
        return self.id_role_value.is_empty()

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

        self.id_role_value.add(individual, weight)

    def remove(self, individual: TycheContext):
        """ Removes the given individual from this identity individual. """
        return self.id_role_value.remove(individual)

    def eval(self, node: 'ADLNode') -> float:
        return self.id_role_value.calculate_expectation(node, ALWAYS)

    def eval_role(self, role: 'Role') -> RoleDist:
        return role.direct_eval(self)

    def get_role(self, symbol: str) -> RoleDist:
        self._verify_not_empty()
        return Expectation.evaluate_role_under_role(self.id_role_value, Role(symbol))

    def get_concept(self, symbol: str) -> float:
        raise TycheIndividualsException(
            f"Cannot evaluate atoms for instances of {type(self).__name__}. eval should be used instead")

    def get_concept_reference(self, symbol: str) -> SymbolReference[float]:
        raise TycheIndividualsException(
            f"Cannot evaluate mutable concepts for instances of {type(self).__name__}")

    def get_role_reference(self, symbol: str) -> SymbolReference[RoleDist]:
        raise TycheIndividualsException(
            f"Cannot evaluate mutable roles for instances of {type(self).__name__}")

    def observe(self, observation: 'ADLNode', likelihood: float = 1, learning_rate: float = 1):
        if learning_rate <= 0:
            return
        self._verify_not_empty()

        prev_id_value = self.id_role_value
        node, given = Given.maybe_unpack(observation)

        # Apply any learning strategy that is set for this IdentityIndividual.
        if self.learning_strat is not None:
            implicit_expectation = Expectation(self.id_role, node, given)
            self.learning_strat.apply(self, self.id_role_ref, implicit_expectation, likelihood, learning_rate)

        # Propagate the observation!
        possible_matching_individuals = prev_id_value.reverse_expectation_learning_params(
            node, given, likelihood
        )
        for ctx, child_likelihood, child_influence in possible_matching_individuals:
            if child_influence > 0:
                ctx.observe(observation, child_likelihood, learning_rate * child_influence)

    def __iter__(self):
        """
        Yields tuples of the possible individuals that could
        represent this individual, and their probability.
        """
        self._verify_not_empty()
        for ctx, prob in self.id_role_value:
            yield ctx, prob

    def to_str(self, *, detail_lvl: int = 1, indent_lvl: int = 0):
        if detail_lvl <= 0:
            return self.name

        return f"{self.name}{self.id_role_value.to_str(detail_lvl=detail_lvl, indent_lvl=indent_lvl)}"
