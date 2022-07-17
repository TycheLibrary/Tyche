"""
Contains functionality to represent references to variables.
"""
from typing import Generic, TypeVar, Callable, Optional


class TycheReferencesException(Exception):
    """
    Class for detailing exceptions with references.
    """
    def __init__(self, message: str):
        self.message = "TycheReferencesException: " + message


SymbolType = TypeVar("SymbolType")


class SymbolReference(Generic[SymbolType]):
    """ Represents a reference to the value of a symbol. """
    symbol: str

    def __init__(self, symbol: str):
        self.symbol = symbol

    def is_mutable(self) -> bool:
        raise NotImplementedError(f"is_mutable is not implemented for {type(self).__name__}")

    def get(self, obj: object) -> SymbolType:
        raise NotImplementedError(f"get is not implemented for {type(self).__name__}")

    def set(self, obj: object, value: SymbolType):
        raise NotImplementedError(f"set is not implemented for {type(self).__name__}")

    def bake(self, obj: object) -> 'BakedSymbolReference':
        return BakedSymbolReference(self, obj)


class BakedSymbolReference(Generic[SymbolType]):
    """
    Represents a reference to the value of a symbol,
    with the object that the value is accessed from baked in.
    """
    def __init__(self, ref: SymbolReference[SymbolType], obj: object):
        self.ref = ref
        self.obj = obj

    def is_mutable(self) -> bool:
        return self.ref.is_mutable()

    def get(self) -> SymbolType:
        return self.ref.get(self.obj)

    def set(self, value: SymbolType):
        self.ref.set(self.obj, value)


class PropertySymbolReference(Generic[SymbolType], SymbolReference[SymbolType]):
    """ Represents a reference to a mutable property in an object. """
    def __init__(self, symbol: str):
        super().__init__(symbol)

    def is_mutable(self) -> bool:
        return True

    def get(self, obj: object) -> SymbolType:
        return getattr(obj, self.symbol)

    def set(self, obj: object, value: SymbolType):
        if not hasattr(obj, self.symbol):
            raise TycheReferencesException(f"The object does not contain the property {self.symbol}. The object is {obj}")
        setattr(obj, self.symbol, value)


class FunctionSymbolReference(Generic[SymbolType], SymbolReference[SymbolType]):
    """ Represents a reference to a mutable property in an object. """
    def __init__(
            self, symbol: str,
            fget: Callable[[object], SymbolType],
            fset: Optional[Callable[[object, SymbolType], None]] = None
    ):
        super().__init__(symbol)
        self.symbol = symbol
        self.fget = fget
        self.fset = fset

    def is_mutable(self) -> bool:
        return self.fset is not None

    def get(self, obj: object) -> SymbolType:
        return self.fget(obj)

    def set(self, obj: object, value: SymbolType):
        if self.fset is None:
            raise TycheReferencesException(
                f"This function reference to {self.symbol} is not mutable")

        self.fset(obj, value)


GuardedSymbolType = TypeVar("GuardedSymbolType")


class GuardedSymbolReference(
    Generic[SymbolType, GuardedSymbolType],
    SymbolReference[SymbolType]
):
    """
    Represents a mutable reference where the accessed value is
    different to the value stored in the reference. The referenced
    value is transformed when accessed or modified.
    """
    def __init__(
            self, ref: SymbolReference[GuardedSymbolType],
            get_transform: Callable[[GuardedSymbolType], SymbolType],
            set_transform: Optional[Callable[[SymbolType], GuardedSymbolType]] = None
    ):

        super().__init__(ref.symbol)
        self.ref = ref
        self.get_transform = get_transform
        self.set_transform = set_transform

    def is_mutable(self) -> bool:
        return self.set_transform is not None and self.ref.is_mutable()

    def get(self, obj: object) -> SymbolType:
        return self.get_transform(self.ref.get(obj))

    def set(self, obj: object, value: SymbolType):
        if self.set_transform is None:
            raise TycheReferencesException(
                f"This guarded reference to {self.symbol} is not mutable")

        self.ref.set(obj, self.set_transform(value))
