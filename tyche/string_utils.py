"""
This file contains utility methods to help the
formatting of strings in Tyche.
"""
from typing import TypeVar, Callable, Iterable, Union

KEY = TypeVar("KEY")
VAL = TypeVar("VAL")


def format_dict(
        dict_value: Union[dict[KEY, VAL], Iterable[tuple[KEY, VAL]]],
        *,
        key_format_fn: Callable[[KEY], str] = str,
        val_format_fn: Callable[[VAL], str] = str,
        indent_lvl: int = 0,
        indent_str: str = "\t",
        prefix: str = "{",
        suffix: str = "}"
) -> str:
    """
    Formats a dictionary into a string, allowing custom key and value
    string formatting, indentation, and prefix/suffix modification.
    """
    if isinstance(dict_value, dict):
        dict_value = dict_value.items()
    key_values = [f"{key_format_fn(key)}: {val_format_fn(val)}" for key, val in dict_value]

    if indent_lvl > 0:
        indentation = indent_str * indent_lvl
        sub_indentation = indent_str * (indent_lvl - 1)
        join_by = f",\n{indentation}"
        prefix = f"{prefix}\n{indentation}"
        suffix = f"\n{sub_indentation}{suffix}"
    else:
        join_by = ", "

    return f"{prefix}{join_by.join(key_values)}{suffix}"
