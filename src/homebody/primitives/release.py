"""The marker for interfaces this release declares but does not implement."""


class NotReleased(NotImplementedError):
    """The interface is declared but its implementation is not in this release."""

    def __init__(self, what):
        super().__init__(f"{what} is not part of this release")
