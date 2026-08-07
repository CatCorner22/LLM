"""Plugin registry for extensible component discovery."""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar, overload

from pioneer.core.exceptions import ValidationError

T = TypeVar("T")


class Registry(Generic[T]):
    """Name-to-component registry with duplicate detection."""

    def __init__(self, namespace: str) -> None:
        self._namespace = namespace
        self._entries: dict[str, T] = {}

    @overload
    def register(self, name: str, obj: T) -> T: ...

    @overload
    def register(self, name: str) -> Callable[[T], T]: ...

    def register(self, name: str, obj: T | None = None) -> T | Callable[[T], T]:
        """Register an object or decorator."""

        def decorator(item: T) -> T:
            self._register(name, item)
            return item

        if obj is not None:
            return decorator(obj)
        return decorator

    def _register(self, name: str, obj: T) -> None:
        if name in self._entries:
            raise ValidationError(
                f"Duplicate registration in '{self._namespace}': {name}",
                details={"namespace": self._namespace, "name": name},
            )
        self._entries[name] = obj

    def get(self, name: str) -> T:
        if name not in self._entries:
            available = sorted(self._entries)
            raise ValidationError(
                f"Unknown {self._namespace}: '{name}'",
                details={"available": available},
            )
        return self._entries[name]

    def list(self) -> list[str]:
        return sorted(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)
