"""
Provides convenience classes for setting up a model using classes.
"""
from typing import TypeVar, Callable, get_type_hints, Final, Type
from tyche.language import RoleDistribution, TycheLanguageException, TycheContext, Atom


# Marks instance variables of classes as probabilities that
# may be accessed by Tyche formulas.
TycheConcept = TypeVar("TycheConcept", float, int)
TycheRole = TypeVar("TycheRole", bound=RoleDistribution)


class TycheIndividualsException(Exception):
    """
    Class for detailing exceptions with individuals.
    """
    def __init__(self, message):
        self.message = "TycheIndividualsException: " + message


class TycheAccessorStore:
    """
    Stores a set of ways to access concepts or roles from a Tyche individual.
    """
    def __init__(self, type_name: str, variables: set[str], functions: set[str]):
        self.type_name = type_name
        self.variables = variables
        self.functions = functions
        self.all_symbols = variables.union(functions)

    def get(self, obj: any, symbol: str):
        """ Accesses the given symbol from the given object. """
        if symbol in self.variables:
            return getattr(obj, symbol)
        elif symbol in self.functions:
            return getattr(obj, symbol)()
        else:
            raise TycheLanguageException("Unknown {} {} for type {}".format(
                self.type_name, symbol, type(obj).__key__
            ))

    @staticmethod
    def get_or_populate_for(obj_type: type, accessors_key: str,
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
    def __init__(self, functions_key: str, fn: Callable[[], RoleDistribution]):
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
    var_type_hint: Final[type] = TycheConcept

    def __init__(self, fn: Callable[[], RoleDistribution]):
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
    var_type_hint: Final[type] = TycheRole

    def __init__(self, fn: Callable[[], RoleDistribution]):
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
    concepts: TycheAccessorStore
    roles: TycheAccessorStore

    def __init__(self):
        super().__init__()
        self.concepts = concept.get(type(self))
        self.roles = role.get(type(self))

    @staticmethod
    def get_concept_names(obj_type: Type['Individual']) -> set[str]:
        return concept.get(obj_type).all_symbols

    @staticmethod
    def get_role_names(obj_type: Type['Individual']) -> set[str]:
        return role.get(obj_type).all_symbols

    def eval_concept(self, symbol: str) -> float:
        return self.concepts.get(self, symbol)

    def eval_role(self, symbol: str) -> RoleDistribution:
        return self.roles.get(self, symbol)
