"""DiSE program frontends.

A frontend takes program source in some input language and produces
a Python callable suitable for :func:`dise.estimate`'s ``program``
argument, together with metadata identifying the nondeterministic
inputs and any property/assertion in the source.

The Python frontend is implicit (DiSE was originally built for
Python).  This package adds:

* :mod:`dise.frontends.svcomp_c` --- a tractable subset of C
  consumed by the SV-COMP verification competition.  Handles
  integer/Boolean arithmetic, control flow, and the
  ``__VERIFIER_*`` idioms; rejects pointers, structs, heap, and
  floats.
"""

from __future__ import annotations

from .svcomp_c import (
    Untranslatable,
    transpile_c_program,
    transpile_c_source,
)

__all__ = [
    "Untranslatable",
    "transpile_c_program",
    "transpile_c_source",
]
