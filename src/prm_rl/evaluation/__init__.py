"""Evaluation utilities.

Submodules are *not* eagerly imported so scalar/behavioral metrics can be
used without pulling in torch. Import from the concrete submodule:

    from prm_rl.evaluation.metrics import final_answer_accuracy
    from prm_rl.evaluation.traps   import exploit_rate
    from prm_rl.evaluation.crhs    import composite_reward_hacking_score
"""
