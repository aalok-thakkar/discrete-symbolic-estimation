"""Third-party integrations.

Adapter modules bridging DiSE to ecosystem libraries. Each submodule
is *optional* — the dependency it wraps is **not** declared as a
required install dependency, so the adapter must be imported
explicitly and degrades gracefully (raises a helpful
:class:`ImportError`) when the underlying library is absent.

Available adapters
==================

* :mod:`dise.integrations.hypothesis` — converts
  ``hypothesis.strategies.SearchStrategy`` objects into DiSE
  :class:`~dise.distributions.Distribution` instances. Realises the
  "operational property-based testing" framing discussed in
  :doc:`/docs/hypothesis-integration`: Hypothesis answers
  *"does any input fail?"*, DiSE answers *"what fraction of
  operational inputs fails — with certified confidence?"*.

Roadmap for additional adapters (not yet implemented):

* ``dise.integrations.pyro`` / ``dise.integrations.numpyro`` —
  numerical operational distributions on integer-valued sub-domains.
* ``dise.integrations.cbmc`` — pass DiSE-refined path constraints to
  CBMC / ESBMC for orthogonal counterexample search.
"""
