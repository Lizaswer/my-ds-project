"""Scratch script to verify linters and type checker are wired."""


def add(a: int, b: int) -> int:
    return a + b


def greet(name: str) -> str:
    """Return a greeting string."""
    return f"Hello, {name}!"


if __name__ == "__main__":
    print(add(1, 2))
    print(greet("world"))
