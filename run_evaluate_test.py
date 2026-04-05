"""
Run evaluation on a small subset for testing.
Usage: uv run run_evaluate_test.py [num_samples]
Example: uv run run_evaluate_test.py 50
"""
import sys
from pathlib import Path
from core import config as app_config
from core.chat_manager import ChatManager
from viseval import Dataset, Evaluator
from run_evaluate import (
    setup_vision_model,
    get_text_model_name,
    run_evaluation,
    save_results,
    print_results
)


def main():
    """Run evaluation on a small subset."""
    
    # Get number of samples from command line
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    
    # Configuration
    folder = app_config.DATASET_FOLDER
    library = app_config.LIBRARY
    log_folder = Path(f"{app_config.LOG_FOLDER}_test_{num_samples}")
    webdriver_path = app_config.WEBDRIVER_PATH
    
    print(f"\n{'='*60}")
    print(f"Running TEST evaluation on {num_samples} samples")
    print(f"{'='*60}\n")
    
    # Setup models
    vision_model, vision_model_name = setup_vision_model() # Uses core.config internally now
    text_model_name = get_text_model_name()
    
    # Initialize components
    dataset = Dataset(Path(folder))
    
    # LIMIT THE DATASET - take only first N samples
    import itertools
    dataset.benchmark = itertools.islice(dataset.benchmark, num_samples)
    
    agent = ChatManager(data_path=folder, log_path=f"./agent_logs_test_{num_samples}.txt")
    evaluator = Evaluator(webdriver_path=None, vision_model=vision_model)
    
    # Initialize OpenAI logger if using OpenAI vision
    if app_config.USE_OPENAI_VISION:
        try:
            from core.openai_vision_client import init_log_path
            init_log_path(str(log_folder / "evaluation.log"))
        except ImportError:
            pass

    # Run evaluation
    config = {"library": library, "logs": log_folder}
    result = run_evaluation(agent, dataset, evaluator, config)
    
    # Save and display results
    detailed_csv_path, scores_json_path, score = save_results(
        result, text_model_name, vision_model_name, library, "all",
        log_folder=log_folder,
        agent_log_path=f"agent_logs_test_{num_samples}.txt"
    )
    print_results(text_model_name, vision_model_name, score, 
                  detailed_csv_path, scores_json_path, log_folder)
    
    print(f"\n{'='*60}")
    print(f"TEST evaluation complete! ({num_samples} samples)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
