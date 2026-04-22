"""
Run evaluation on visEval_dataset
Usage: python run_evaluate.py
Example:
uv run core/vllm_server.py
uv run run_evaluate.py

grep -oF "evaluation finished." evaluate_logs/evaluation.log | wc -l
find evaluate_logs -type f -name '*.svg' | wc -l
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

from core import config as app_config
from core.chat_manager import ChatManager
from viseval import Dataset, Evaluator


def setup_vision_model() -> Tuple[Optional[object], Optional[str]]:
    """
    Configure and initialize the vision model.
    
    Returns:
        Tuple of (vision_model, vision_model_name)
    """
    # Try OpenAI Vision first
    if app_config.USE_OPENAI_VISION:
        try:
            from core.openai_vision_client import get_vision_model as get_openai_vision_model
            print(f"Using OpenAI vision model: {app_config.OPENAI_VISION_MODEL_NAME}...")
            return get_openai_vision_model(), app_config.OPENAI_VISION_MODEL_NAME
        except ImportError:
            print("Vision modules not configured, vision model disabled")
            return None, None
            
    print("Vision model critic disabled in config (USE_OPENAI_VISION=False)")
    return None, None


def get_text_model_name() -> str:
    """
    Get the name of the text model being used.
    
    Returns:
        Model name string
    """
    return app_config.MODEL_NAME


def run_evaluation(agent, dataset, evaluator, config: dict):
    """
    Execute the evaluation process.
    
    Args:
        agent: ChatManager instance
        dataset: Dataset instance
        evaluator: Evaluator instance
        config: Evaluation configuration dict
        
    Returns:
        EvaluationResult object
    """
    return evaluator.evaluate(agent, dataset, config)


def save_results(result, text_model_name: str, vision_model_name: Optional[str], 
                 library: str, table_type: str, 
                 log_folder: Path = Path(app_config.LOG_FOLDER),
                 agent_log_path: str = app_config.AGENT_LOG_FILE):
    """
    Save evaluation results to CSV and JSON files.
    
    Args:
        result: EvaluationResult object
        text_model_name: Name of text model used
        vision_model_name: Name of vision model critic used (or None)
        library: Visualization library used
        table_type: single, multiple, or all
        log_folder: Path to evaluation log folder
        agent_log_path: Path to agent log file
        
    Returns:
        Tuple of (detailed_csv_path, scores_json_path)
    """
    # Create unique timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract model names for filename (remove path prefix)
    text_model_short = text_model_name.split('/')[-1] if '/' in text_model_name else text_model_name
    vision_model_short = vision_model_name.split('/')[-1] if vision_model_name and '/' in vision_model_name else (vision_model_name or 'NoVision')
    
    # Get dataset size from evaluation.log (count completed evaluations)
    eval_log_path = Path("evaluate_logs") / "evaluation.log"
    if eval_log_path.exists():
        with open(eval_log_path, 'r') as f:
            log_content = f.read()
            dataset_size = log_content.count("evaluation finished")
    else:
        # Fallback to counting from dataset JSON if no log exists
        dataset_size = len(json.load(open(f'{app_config.DATASET_FOLDER}/visEval.json'))) if table_type == 'all' else len(json.load(open(f'{app_config.DATASET_FOLDER}/visEval_' + table_type + '.json')))

    # Load full dataset to classify instances as single/multi-table
    with open(f'{app_config.DATASET_FOLDER}/visEval.json') as f:
        full_dataset = json.load(f)
    
    # Load detailed results
    detailed_df = result.detail_records()
    
    # Add classification for single vs multi-table
    def is_multi_table(instance_id):
        if instance_id in full_dataset:
            vql = full_dataset[instance_id].get("vis_query", {}).get("VQL", "")
            return "JOIN" in vql.upper()
        return False
    
    detailed_df['is_multi_table'] = detailed_df['id'].apply(is_multi_table)
    
    # Calculate single and multi-table scores
    def calc_scores(df_subset):
        if len(df_subset) == 0:
            return {
                "query_count": 0, 
                "invalid_rate": 0, 
                "illegal_rate": 0, 
                "pass_rate": 0,
                "readability_score": 0,
                "quality_score": 0
            }
        
        scores = {
            "query_count": len(df_subset),
            "invalid_rate": df_subset['invalid_rate'].mean(),
            "illegal_rate": df_subset['illegal rate'].mean(),
            "pass_rate": df_subset['pass_rate'].mean(),
        }
        
        if 'readability_score' in df_subset.columns:
            scores['readability_score'] = df_subset['readability_score'].mean()
        
        if 'quality_score' in df_subset.columns:
            scores['quality_score'] = df_subset['quality_score'].mean()
            
        return scores
    
    single_scores = calc_scores(detailed_df[~detailed_df['is_multi_table']])
    multi_scores = calc_scores(detailed_df[detailed_df['is_multi_table']])

    # Create results directory
    results_dir = Path("results")
    run_folder_name = f"{timestamp}_{text_model_short}_{vision_model_short}"
    run_folder = results_dir / run_folder_name
    run_folder.mkdir(parents=True, exist_ok=True)
    
    # Create unique filenames
    detailed_csv_path = run_folder / "detailed_results.csv"
    scores_json_path = run_folder / "scores.json"
    
    # Prepare metadata
    run_metadata = {
        "text_model": text_model_name,
        "vision_model": vision_model_name,
        "timestamp": datetime.now().isoformat(),
        "library": library,
        "table_type": table_type,
        "dataset_size": dataset_size
    }
    
    # Save detailed results with model info
    detailed_df['text_model'] = text_model_name
    detailed_df['vision_model'] = vision_model_name if vision_model_name else 'None'
    detailed_df.to_csv(detailed_csv_path, index=False)
    
    # Save scores with metadata and single/multi breakdown
    score = result.score()
    final_output = {
        "metadata": run_metadata,
        "scores": score,
        "single_table_scores": single_scores,
        "multi_table_scores": multi_scores
    }
    with open(scores_json_path, "w") as f:
        json.dump(final_output, f, indent=2)

    # Move logs to results folder
    try:
        if (Path("api_trace.json").exists()):
            shutil.move("api_trace.json", run_folder / "api_trace.json")
        if (Path(agent_log_path).exists()):
            shutil.move(agent_log_path, run_folder / Path(agent_log_path).name)
        
        # Move OpenAI vision trace if it exists
        openai_trace_path = log_folder / "openai_vision_trace.json"
        if openai_trace_path.exists():
            shutil.move(str(openai_trace_path), run_folder / "openai_vision_trace.json")

        # Move entire evaluate_logs folder into the result folder
        if log_folder.exists():
            shutil.move(str(log_folder), str(run_folder / log_folder.name))
            
    except Exception as e:
        print(f"Warning: Could not move log files: {e}")
    
    return detailed_csv_path, scores_json_path, score


def print_results(text_model_name: str, vision_model_name: Optional[str], 
                  score: dict, detailed_csv_path: Path, scores_json_path: Path,
                  log_folder: Path):
    """
    Print evaluation results to console.
    
    Args:
        text_model_name: Name of text model used
        vision_model_name: Name of vision model used (or None)
        score: Dictionary of evaluation scores
        detailed_csv_path: Path to detailed CSV file
        scores_json_path: Path to scores JSON file
        log_folder: Path to evaluation log folder
    """
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Text Model: {text_model_name}")
    print(f"Vision Model: {vision_model_name or 'None'}")
    print("="*60)
    print(f"\nFinal Scores:")
    for metric, value in score.items():
        print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")
    
    print(f"\n✅ Results saved to:")
    print(f"   - {detailed_csv_path}")
    print(f"   - {scores_json_path}")
    print(f"\n📁 Evaluation cache: {log_folder / 'evaluation.log'}")
    print("="*60)


def main():
    """Main evaluation pipeline."""
    # Configuration
    folder = app_config.DATASET_FOLDER
    table_type = "all" # single, multiple, or all
    library = app_config.LIBRARY
    log_folder = Path(app_config.LOG_FOLDER)
    # webdriver_path = app_config.WEBDRIVER_PATH # set path to chrome driver
    
    # Setup models
    vision_model, vision_model_name = setup_vision_model()
    text_model_name = get_text_model_name()
    
    # Initialize components
    dataset = Dataset(Path(folder))
    agent = ChatManager(data_path=folder, log_path=f"./{app_config.AGENT_LOG_FILE}")
    evaluator = Evaluator(webdriver_path=None, vision_model=vision_model)
    
    # Initialize OpenAI logger if using OpenAI vision
    if app_config.USE_OPENAI_VISION:
        try:
            from core.openai_vision_client import init_log_path
            msg = init_log_path(str(log_folder / "evaluation.log"))
        except ImportError:
            pass

    # Run evaluation
    config = {"library": library, "logs": log_folder}
    result = run_evaluation(agent, dataset, evaluator, config)
    
    # Save and display results
    detailed_csv_path, scores_json_path, score = save_results(
        result, text_model_name, vision_model_name, library, table_type, log_folder
    )
    print_results(text_model_name, vision_model_name, score, 
                  detailed_csv_path, scores_json_path, log_folder)


if __name__ == "__main__":
    main()