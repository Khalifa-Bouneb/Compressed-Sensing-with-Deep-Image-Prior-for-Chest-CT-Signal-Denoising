"""Opt-in profiling markers and exact execution counters for DIP-ADMM."""

import os
from collections import Counter
from contextlib import nullcontext
from functools import wraps

from torch.profiler import record_function


_COUNTERS = Counter()


def profiling_enabled():
    return os.environ.get("DIP_PROFILE") == "1"


def profile_region(name):
    return record_function(name) if profiling_enabled() else nullcontext()


def profiled(name):
    def decorator(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with profile_region(name):
                return function(*args, **kwargs)
        return wrapped
    return decorator


def increment_counter(name, amount=1):
    if profiling_enabled():
        _COUNTERS[name] += amount


def reset_counters():
    _COUNTERS.clear()


def counter_snapshot():
    return dict(_COUNTERS)


def profile_iteration_setting(environment_name, default):
    """Read a profiling-only iteration override, preserving normal behavior."""
    if not profiling_enabled():
        return default
    return int(os.environ.get(environment_name, default))
