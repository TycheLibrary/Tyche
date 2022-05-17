"""
Contains functionality to represent references to variables.
"""
from typing import Generic, TypeVar, Callable


VariableGetType = TypeVar("VariableGetType")
VariableSetType = TypeVar("VariableSetType")


class MutableReference(Generic[VariableGetType, VariableSetType]):
    """ Represents a reference to a mutable value. """
    def get(self) -> VariableGetType:
        raise NotImplementedError(f"get is not implemented for {type(self).__name__}")

    def set(self, value: VariableSetType):
        raise NotImplementedError(f"set is not implemented for {type(self).__name__}")


TransformedType = TypeVar("TransformedType")


class GuardedMutableReference(
    Generic[VariableGetType, VariableSetType, TransformedType],
    MutableReference[VariableGetType, VariableSetType]
):
    """ Represents a mutable reference with get/set transform functions. """
    def __init__(
            self,
            ref: MutableReference[VariableGetType, VariableSetType],
            get_transform: Callable[[VariableGetType], TransformedType],
            set_transform: Callable[[TransformedType], VariableSetType]):

        self.ref = ref
        self.get_transform = get_transform
        self.set_transform = set_transform

    def get(self) -> VariableGetType:
        return self.get_transform(self.ref.get())

    def set(self, value: TransformedType):
        self.ref.set(self.set_transform(value))


VariableType = TypeVar("VariableType")


class MutableVariableReference(Generic[VariableType], MutableReference[VariableType, VariableType]):
    """ Represents a reference to a mutable variable in an object. """
    def __init__(self, obj: any, symbol: str):
        self.obj = obj
        self.symbol = symbol

    def get(self) -> VariableType:
        return getattr(self.obj, self.symbol)

    def set(self, value: VariableType):
        setattr(self.obj, self.symbol, value)
