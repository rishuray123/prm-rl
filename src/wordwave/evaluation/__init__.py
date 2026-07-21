"""Evaluation utilities.

Submodules are *not* eagerly imported so scalar/behavioral metrics can be
used without pulling in torch. Import from the concrete submodule:

    from wordwave.evaluation.metrics import final_answer_accuracy
    from wordwave.evaluation.traps   import exploit_rate
    from wordwave.evaluation.crhs    import composite_reward_hacking_score
"""
