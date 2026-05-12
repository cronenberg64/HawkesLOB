#!/usr/bin/env python3
"""Patch tick library for Python 3.12+ compatibility.

tick 0.8.x has a metaclass-based __setattr__ that rejects setting attributes
not explicitly declared in _attrinfos. Several internal attributes (events,
dtype, etc.) were never declared, which breaks on Python 3.12+ where the class
creation order changed slightly.

This script patches the installed tick/base/base.py to use a permissive
fallback instead of raising AttributeError.

Run once after installing tick:
    python scripts/patch_tick.py
"""

import importlib
import re

import tick.base.base as base_mod

path = base_mod.__file__

with open(path, "r") as f:
    content = f.read()

# Patch 1: __setattr__ fallback
old_setattr = (
    '                raise AttributeError("\'%s\' object has no settable attribute "\n'
    '                                     "\'%s\'" % (class_name, key))'
)
new_setattr = (
    '                # Fallback: allow setting undeclared attributes\n'
    '                # (patched for Python 3.12 compatibility)\n'
    '                object.__setattr__(self, key, val)'
)

# Patch 2: _set fallback
old_set = (
    '            if key not in attrinfos:\n'
    '                raise AttributeError("\'%s\' object has no settable attribute "\n'
    '                                     "\'%s\'" % (class_name, key))'
)
new_set = (
    '            if key not in attrinfos:\n'
    '                # Fallback: allow undeclared attributes (Python 3.12 compat)\n'
    '                object.__setattr__(self, BaseMeta.hidden_attr(key), val)\n'
    '                return'
)

patched = content
n_patches = 0

if old_setattr in patched:
    patched = patched.replace(old_setattr, new_setattr)
    n_patches += 1

if old_set in patched:
    patched = patched.replace(old_set, new_set)
    n_patches += 1

if n_patches == 0:
    print("tick/base/base.py is already patched (or has a different version).")
else:
    with open(path, "w") as f:
        f.write(patched)
    print(f"Patched {path} ({n_patches} changes)")
    print("tick is now compatible with Python 3.12+")
