"""Evaluation utilities.

The core scalar / behavioral / trap metrics are re-exported here for
convenience so `scripts/evaluate.py` and notebooks can import flat:

    from prm_rl.evaluation import (
        final_answer_accuracy, process_correctness,
        behavioral_scores, evaluator_stress_test,
        load_trap_scenarios, exploit_rate,
        composite_reward_hacking_score,
    )

None of the submodules re-exported below import torch at module scope
(torch is only pulled in lazily inside functions that need a PRM/NLI
scorer), so `import prm_rl.evaluation` remains cheap.

The heavier `faithfulness` / `cma` submodules (paired-intervention
protocols) are intentionally NOT re-exported — import them directly:

    from prm_rl.evaluation.faithfulness import ...
    from prm_rl.evaluation.cma          import ...
"""
from .behavioral import behavioral_scores
from .crhs       import composite_reward_hacking_score
from .est        import evaluator_stress_test
from .metrics    import final_answer_accuracy, process_correctness
from .traps      import exploit_rate, load_trap_scenarios

__all__ = [
    "behavioral_scores",
    "composite_reward_hacking_score",
    "evaluator_stress_test",
    "exploit_rate",
    "final_answer_accuracy",
    "load_trap_scenarios",
    "process_correctness",
]
