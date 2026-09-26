"""Quality evaluation adapters."""

from huyawo_quant.evaluation.lm_eval import (
    build_hellaswag_evaluation_profile,
    build_hellaswag_task_config,
)

__all__ = [
    "build_hellaswag_evaluation_profile",
    "build_hellaswag_task_config",
]
