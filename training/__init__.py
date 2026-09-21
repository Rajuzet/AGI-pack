"""Free-Tier Cloud Training Pipeline for Open-Source LLMs (Colab/Kaggle).

Exposes QLoRA fine-tuning configuration, dataset preparation, and GCS synchronization.
"""

from training.kaggle_lora_train import (
    LoRATrainingConfig,
    format_research_dataset,
    run_lora_training,
)

__all__ = [
    "LoRATrainingConfig",
    "format_research_dataset",
    "run_lora_training",
]
