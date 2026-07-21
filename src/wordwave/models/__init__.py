"""Model wrappers.

Kept as a namespace package with no eager torch imports so upstream tests
can import subpackages that only transitively touch these files (e.g.
`wordwave.evaluation.metrics`) without needing torch installed.

Import concrete loaders from the submodule:

    from wordwave.models.policy import load_policy_and_tokenizer
    from wordwave.models.prm    import load_prm, PRMScorer
    from wordwave.models.nli    import load_nli, ContradictionScorer
"""
