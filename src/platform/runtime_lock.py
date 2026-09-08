"""Serialize mutations of the single supported local data runtime.

Only hold this lock around synchronous mutation/commit sections, never across
an awaited network request. Threaded restore and async route commits share it.
"""
from functools import wraps
from threading import RLock

RUNTIME_MUTATION_LOCK = RLock()


def runtime_mutation(function):
    @wraps(function)
    def guarded(*args, **kwargs):
        with RUNTIME_MUTATION_LOCK:
            return function(*args, **kwargs)
    return guarded
