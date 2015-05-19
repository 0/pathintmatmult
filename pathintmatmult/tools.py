"""
Assorted tools.
"""

from functools import wraps


def cached(f):
    """
    A simple cache for constant instance methods.

    Requires a _cached dict on the instance.
    """

    @wraps(f)
    def wrapped(self, *args, **kwargs):
        if f not in self._cached:
            self._cached[f] = f(self, *args, **kwargs)

        return self._cached[f]

    return wrapped
