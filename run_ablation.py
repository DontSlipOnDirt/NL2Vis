"""
Ablation study runner for NL2Vis agent configurations.

Runs all agent/prompt ablation configs for the currently loaded vLLM model.
Sends Discord notifications on each run start, completion, and failure,
plus a final summary when all runs are done.

Ablation configurations tested:
  full                  - Processor + full CoT Composer + Validator (baseline)
  no_processor          - Raw schema + full CoT Composer + Validator
  no_composer_template  - Processor + simplified Composer prompt + Validator
  no_proc_no_comp_tmpl  - Raw schema + simplified Composer prompt + Validator

Usage:
    # Make sure vLLM server is running with the target model, then:
    python run_ablation.py --model qwen14b
    python run_ablation.py --model qwen30b

    # Dry-run (skips evaluation, just tests Discord + config)
    python run_ablation.py --model qwen14b --dry-run
"""

import argparse
import json
import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

# ---------------------------------------------------------------------------
# Model registry — add new models here
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "qwen7b": "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
    "qwen14b": "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ",
    # "qwen27b": "Qwen/Qwen3.6-27B-FP8",
    "qwen30b": "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
    "qwen32b": "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ",
}

# ---------------------------------------------------------------------------
# Ablation configurations: (label, skip_processor, skip_composer_template)
# ---------------------------------------------------------------------------
ABLATION_CONFIGS = [
    # ("full",                 False, False),
    # ("no_processor",         True,  False),
    # ("no_composer_template", False, True),
    ("no_proc_no_comp_tmpl", True,  True),
]


# ---------------------------------------------------------------------------
# Discord helpers
# ---------------------------------------------------------------------------

def send_discord(description: str, title: str = "NL2Vis Ablation", color: int = 0x5865F2) -> None:
    """POST an embed to the Discord webhook. Silently logs on failure."""
    if not DISCORD_WEBHOOK_URL:
        print(f"[Discord] No webhook configured — skipping: {title}")
        return
    payload = {
        "embeds": [{
            "title": title,
            "description": description,
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
        }]
    }
    try:
        resp = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        print(f"[Discord] Failed to send notification: {exc}")


# ---------------------------------------------------------------------------
# Single ablation run
# ---------------------------------------------------------------------------

def run_single_ablation(
    config_name: str,
    skip_processor: bool,
    skip_composer_template: bool,
    model_key: str,
    model_full: str,
    dry_run: bool = False,
) -> dict | None:
    """
    Execute one ablation configuration and return its score dict, or None on failure.
    Mutates core.config flags before running and restores them afterwards.
    """
    from core import config as app_config

    # Apply ablation flags
    app_config.ABLATION_SKIP_PROCESSOR = skip_processor
    app_config.ABLATION_SKIP_COMPOSER_TEMPLATE = skip_composer_template

    tag = f"`{config_name}` | `{model_key}`"
    print(f"\n{'='*60}")
    print(f"ABLATION: {config_name}  |  Model: {model_key}")
    print(f"  skip_processor={skip_processor}, skip_composer_template={skip_composer_template}")
    print(f"{'='*60}")

    send_discord(
        f"**Config:** `{config_name}`\n"
        f"**Model:** `{model_full}`\n"
        f"- Skip Processor LLM: {skip_processor}\n"
        f"- Simplified Composer Prompt: {skip_composer_template}",
        title="🔬 Ablation Run Started",
        color=0x5865F2,
    )

    if dry_run:
        print("[DRY RUN] Skipping evaluation.")
        send_discord(
            f"**Config:** {tag} — dry run, no evaluation performed.",
            title="⚙️ Dry Run",
            color=0xFEE75C,
        )
        return {"pass_rate": 0.0, "quality_score": 0.0, "_dry_run": True}

    try:
        from run_evaluate import setup_vision_model, get_text_model_name, run_evaluation, save_results
        from core.chat_manager import ChatManager
        from viseval import Dataset, Evaluator

        folder = app_config.DATASET_FOLDER
        library = app_config.LIBRARY
        log_folder = Path(app_config.LOG_FOLDER)

        vision_model, vision_model_name = setup_vision_model()
        text_model_name = get_text_model_name()

        dataset = Dataset(Path(folder))
        agent = ChatManager(data_path=folder, log_path=f"./ablation_{config_name}_{model_key}_agents.log")
        evaluator = Evaluator(webdriver_path=None, vision_model=vision_model)

        if app_config.USE_OPENAI_VISION:
            try:
                from core.openai_vision_client import init_log_path
                init_log_path(str(log_folder / "evaluation.log"))
            except ImportError:
                pass

        eval_config = {"library": library, "logs": log_folder}
        result = run_evaluation(agent, dataset, evaluator, eval_config)

        detailed_csv_path, scores_json_path, score = save_results(
            result, text_model_name, vision_model_name, library, "all", log_folder,
        )

        # Rename result folder to include ablation label
        run_folder = detailed_csv_path.parent
        ablation_folder_name = f"{run_folder.name}_ablation_{config_name}"
        ablation_folder = run_folder.parent / ablation_folder_name
        run_folder.rename(ablation_folder)
        detailed_csv_path = ablation_folder / detailed_csv_path.name
        scores_json_path = ablation_folder / scores_json_path.name

        print(f"\nResults saved to: {ablation_folder}")

        score_lines = "\n".join(
            f"- {k}: `{v:.4f}`" if isinstance(v, float) else f"- {k}: `{v}`"
            for k, v in score.items()
        )
        send_discord(
            f"**Config:** `{config_name}`\n"
            f"**Model:** `{model_full}`\n\n"
            f"**Scores:**\n{score_lines}\n\n"
            f"📁 `{ablation_folder_name}`",
            title="✅ Ablation Run Complete",
            color=0x57F287,
        )
        return score

    except Exception as exc:
        tb = traceback.format_exc()
        print(f"ERROR in ablation '{config_name}':\n{tb}")
        send_discord(
            f"**Config:** `{config_name}`\n"
            f"**Model:** `{model_full}`\n\n"
            f"```\n{str(exc)[:800]}\n```",
            title="❌ Ablation Run Failed",
            color=0xED4245,
        )
        return None

    finally:
        # Always restore flags to safe defaults
        app_config.ABLATION_SKIP_PROCESSOR = False
        app_config.ABLATION_SKIP_COMPOSER_TEMPLATE = False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run NL2Vis agent ablation study for a given (already-loaded) vLLM model."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Short key for the model currently running in vLLM.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=[c[0] for c in ABLATION_CONFIGS],
        default=None,
        help="Subset of ablation configs to run (default: all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test Discord notifications and config wiring without running evaluation.",
    )
    args = parser.parse_args()

    model_key = args.model
    model_full = MODEL_REGISTRY[model_key]

    configs_to_run = (
        [c for c in ABLATION_CONFIGS if c[0] in args.configs]
        if args.configs
        else ABLATION_CONFIGS
    )

    config_labels = ", ".join(f"`{c[0]}`" for c in configs_to_run)
    send_discord(
        f"**Model:** `{model_full}`\n"
        f"**Configs:** {config_labels}\n"
        f"**Total runs:** {len(configs_to_run)}"
        + (" *(dry run)*" if args.dry_run else ""),
        title="🚀 Ablation Study Started",
        color=0xFEE75C,
    )

    all_scores: dict[str, dict | None] = {}
    start_time = datetime.now()

    for config_name, skip_proc, skip_comp in configs_to_run:
        scores = run_single_ablation(
            config_name=config_name,
            skip_processor=skip_proc,
            skip_composer_template=skip_comp,
            model_key=model_key,
            model_full=model_full,
            dry_run=args.dry_run,
        )
        all_scores[config_name] = scores

    elapsed = datetime.now() - start_time

    # Build summary
    summary_lines = [f"**Model:** `{model_full}`\n"]
    for config_name, scores in all_scores.items():
        if scores is None:
            summary_lines.append(f"❌ **{config_name}** — FAILED")
        else:
            pr = scores.get("pass_rate", "N/A")
            qs = scores.get("quality_score", "N/A")
            inv = scores.get("invalid_rate", "N/A")
            dr = " *(dry)*" if scores.get("_dry_run") else ""
            pr_str = f"{pr:.4f}" if isinstance(pr, float) else str(pr)
            qs_str = f"{qs:.4f}" if isinstance(qs, float) else str(qs)
            inv_str = f"{inv:.4f}" if isinstance(inv, float) else str(inv)
            summary_lines.append(
                f"✅ **{config_name}**{dr} | pass_rate: `{pr_str}` | "
                f"quality: `{qs_str}` | invalid: `{inv_str}`"
            )

    summary_lines.append(f"\n⏱ Total time: `{str(elapsed).split('.')[0]}`")

    send_discord(
        "\n".join(summary_lines),
        title="📊 Ablation Study Complete",
        color=0x57F287,
    )

    print("\n" + "="*60)
    print("ABLATION STUDY COMPLETE")
    print(f"Model: {model_full}")
    print(f"Elapsed: {str(elapsed).split('.')[0]}")
    print("="*60)
    for config_name, scores in all_scores.items():
        status = "✓" if scores is not None else "✗"
        pr = f"{scores['pass_rate']:.4f}" if scores and isinstance(scores.get('pass_rate'), float) else "N/A"
        print(f"  [{status}] {config_name:<25} pass_rate={pr}")


if __name__ == "__main__":
    main()
