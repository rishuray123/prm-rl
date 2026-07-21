"""Model wrappers.

Kept as a namespace package with no eager torch imports so upstream tests
can import subpackages that only transitively touch these files (e.g.
`prm_rl.evaluation.metrics`) without needing torch installed.

Import concrete loaders from the submodule:

    from prm_rl.models.policy import load_policy_and_tokenizer
    from prm_rl.models.prm    import load_prm, PRMScorer
    from prm_rl.models.nli    import load_nli, ContradictionScorer
"""
