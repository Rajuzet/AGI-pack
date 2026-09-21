"""Unit and integration tests for free-tier cloud training pipeline."""

import json
from pathlib import Path
import tempfile
import pytest

from training.kaggle_lora_train import (
    LoRATrainingConfig,
    detect_runtime_environment,
    format_research_dataset,
    run_lora_training,
)


def test_lora_training_config_defaults():
    """Verify default LoRA parameters target 14GB VRAM bounds, 4-bit quantization, and gradient checkpointing."""
    config = LoRATrainingConfig()

    assert config.load_in_4bit is True
    assert config.bnb_4bit_quant_type == "nf4"
    assert config.bnb_4bit_use_double_quant is True
    assert config.target_modules == ["q_proj", "v_proj", "k_proj", "o_proj"]
    assert config.lora_r == 16
    assert config.lora_alpha == 32
    assert config.task_type == "CAUSAL_LM"
    assert config.optim == "paged_adamw_8bit"

    # Strictly verify 14GB VRAM constraint parameters
    assert config.per_device_train_batch_size == 1
    assert config.gradient_accumulation_steps == 4
    assert config.gradient_checkpointing is True


def test_format_research_dataset_from_records():
    """Verify research records are correctly formatted into ChatML instruction-tuning pairs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_jsonl = Path(tmpdir) / "sample_papers.jsonl"
        sample_records = [
            {
                "title": "Quantum Attention Mechanisms in Transformers",
                "content": "We propose a quantum-inspired attention kernel that scales quadratically less in memory.",
                "source": "arxiv",
                "metadata": {"authors": ["Dr. Alice", "Dr. Bob"], "link": "https://arxiv.org/abs/2402.12345"},
            }
        ]
        with tmp_jsonl.open("w", encoding="utf-8") as f:
            for r in sample_records:
                f.write(json.dumps(r) + "\n")

        dataset = format_research_dataset(data_path=tmp_jsonl)
        assert len(dataset) == 1
        sample = dataset[0]

        assert "instruction" in sample
        assert "input" in sample
        assert "output" in sample
        assert "text" in sample

        assert "Quantum Attention Mechanisms" in sample["input"]
        assert "<|im_start|>system" in sample["text"]
        assert "<|im_start|>user" in sample["text"]
        assert "<|im_start|>assistant" in sample["text"]


def test_format_research_dataset_synthetic_fallback():
    """Verify empty directory triggers synthetic seed pairs without crashing."""
    with tempfile.TemporaryDirectory() as empty_dir:
        dataset = format_research_dataset(data_path=empty_dir)
        assert len(dataset) >= 1
        assert "instruction" in dataset[0]


def test_detect_runtime_environment():
    """Verify environment detection returns structured dictionary."""
    env = detect_runtime_environment()
    assert "is_kaggle" in env
    assert "is_colab" in env
    assert "device" in env
    assert "vram_gb" in env


def test_run_lora_training_dry_run():
    """Verify end-to-end dry-run execution writes checkpoints and produces summary report."""
    with tempfile.TemporaryDirectory() as tmp_out:
        config = LoRATrainingConfig(
            output_dir=tmp_out,
            dry_run=True,
            upload_gcs=True,  # In dry run mode, this simulates GCS upload
        )

        summary = run_lora_training(config=config)

        assert "run_id" in summary
        assert summary["quantization"] == "4-bit NF4"
        assert summary["lora_target_modules"] == ["q_proj", "v_proj", "k_proj", "o_proj"]
        assert summary["training_samples"] >= 1
        assert summary["training_loss"] > 0

        # Verify output checkpoint artifacts
        run_dir = Path(summary["local_checkpoint_dir"])
        assert run_dir.exists()
        assert (run_dir / "adapter_config.json").exists()
        assert (run_dir / "adapter_model.safetensors").exists()
        assert (run_dir / "training_args.json").exists()
        assert (run_dir / "train_dataset.json").exists()
        assert (run_dir / "run_summary.json").exists()

        # Check adapter_config content
        with (run_dir / "adapter_config.json").open("r", encoding="utf-8") as f:
            adapter_conf = json.load(f)
            assert adapter_conf["r"] == 16
            assert adapter_conf["lora_alpha"] == 32
            assert adapter_conf["target_modules"] == ["q_proj", "v_proj", "k_proj", "o_proj"]

        # Check GCS simulation
        gcs_sync = summary["gcs_sync"]
        assert gcs_sync.get("status") == "dry_run_simulated"
        assert "adapter_config.json" in gcs_sync.get("uploaded_files", [])


def test_configure_bnb_quantization():
    """Verify BitsAndBytesConfig enforces 4-bit loading and torch.float16 or bfloat16 compute dtype."""
    import torch
    from training.kaggle_lora_train import configure_bnb_quantization

    config = LoRATrainingConfig(load_in_4bit=True, torch_dtype="float16")
    bnb_config, compute_dtype = configure_bnb_quantization(config)

    assert bnb_config.load_in_4bit is True
    assert bnb_config.bnb_4bit_compute_dtype in (torch.float16, torch.bfloat16)
    assert bnb_config.bnb_4bit_quant_type == "nf4"
    assert bnb_config.bnb_4bit_use_double_quant is True


def test_build_training_arguments_compatibility(tmp_path):
    """Verify dynamic SFTConfig selection and graceful TrainingArguments fallback."""
    from training.kaggle_lora_train import build_training_arguments

    config = LoRATrainingConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
    )

    # 1. Test with SFTConfig available (modern trl)
    args_sft, mode_sft = build_training_arguments(config, tmp_path / "out1")
    assert mode_sft == "sft_config"
    assert args_sft.per_device_train_batch_size == 1
    assert args_sft.gradient_accumulation_steps == 4
    assert args_sft.gradient_checkpointing is True

    # 2. Test fallback to TrainingArguments (when SFTConfig is unavailable)
    args_fallback, mode_fallback = build_training_arguments(
        config, tmp_path / "out2", sft_config_cls=False
    )
    assert mode_fallback == "training_args"
    assert args_fallback.per_device_train_batch_size == 1
    assert args_fallback.gradient_accumulation_steps == 4
    assert args_fallback.gradient_checkpointing is True
