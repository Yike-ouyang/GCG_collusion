"""Seeds that are identical in every Python process.

`hash()` of a string changes between processes (PYTHONHASHSEED), so seeds built from string keys made LLM
calls irreproducible and defeated the response cache. `hash()` of a tuple of ints is stable, which is why
legacy move seeds (and tools that recover runs from them, such as plot_pot_logs.py) keep using it.
"""

from __future__ import annotations

import zlib


def stable_seed(*key) -> int:
    """Deterministic seed in [0, 2**31) from any key made of ints, floats and strings."""
    return zlib.crc32(repr(key).encode()) % 2**31


def legacy_seed(*key) -> int:
    """The historical seed: `hash()` for all-int keys (stable across processes), `stable_seed` otherwise."""
    if all(isinstance(x, int) for x in key):
        return hash(key) % 2**31
    return stable_seed(*key)
