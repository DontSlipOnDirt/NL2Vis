"""
Analyze results split by single-table vs multi-table queries.
Usage: python analyze_single_multi.py <path_to_detailed_results.csv>
"""
import sys
import pandas as pd
import json
from pathlib import Path


def is_multi_table(instance_id, dataset):
    """Check if an instance uses multi-table queries (has JOIN)."""
    if instance_id in dataset:
        vql = dataset[instance_id].get("vis_query", {}).get("VQL", "")
        return "JOIN" in vql.upper()
    return False


def calculate_scores(subset_df):
    """Calculate evaluation scores for a subset of results."""
    total = len(subset_df)
    if total == 0:
        return {"count": 0, "invalid_rate": 0, "illegal_rate": 0, "pass_rate": 0}
    
    return {
        "count": total,
        "invalid_rate": subset_df['invalid_rate'].mean(),
        "illegal_rate": subset_df['illegal rate'].mean(),  # Note: space in column name
        "pass_rate": subset_df['pass_rate'].mean(),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_single_multi.py <path_to_detailed_results.csv>")
        print("Example: python analyze_single_multi.py results/20251118_123456_Qwen2.5-7B-Instruct_NoVision/detailed_results.csv")
        sys.exit(1)
    
    csv_path = Path(sys.argv[1])
    
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    
    print(f"Analyzing: {csv_path}\n")
    
    # Load detailed results
    df = pd.read_csv(csv_path)
    
    # Load the full dataset to get VQL for each instance
    dataset_path = Path("visEval_dataset/visEval.json")
    if not dataset_path.exists():
        print(f"Error: Dataset not found: {dataset_path}")
        sys.exit(1)
    
    with open(dataset_path) as f:
        dataset = json.load(f)
    
    # Add classification column
    df['is_multi_table'] = df['id'].apply(lambda x: is_multi_table(x, dataset))
    
    # Split into single and multi-table subsets
    single_df = df[~df['is_multi_table']]
    multi_df = df[df['is_multi_table']]
    
    # Calculate scores
    single_scores = calculate_scores(single_df)
    multi_scores = calculate_scores(multi_df)
    all_scores = calculate_scores(df)
    
    # Print results
    print("=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    print(f"  Total Instances: {all_scores['count']}")
    print(f"  Pass Rate:       {all_scores['pass_rate']:.4f} ({all_scores['pass_rate']*100:.2f}%)")
    print(f"  Invalid Rate:    {all_scores['invalid_rate']:.4f} ({all_scores['invalid_rate']*100:.2f}%)")
    print(f"  Illegal Rate:    {all_scores['illegal_rate']:.4f} ({all_scores['illegal_rate']*100:.2f}%)")
    
    print("\n" + "=" * 60)
    print("SINGLE-TABLE RESULTS")
    print("=" * 60)
    print(f"  Count:           {single_scores['count']}")
    print(f"  Pass Rate:       {single_scores['pass_rate']:.4f} ({single_scores['pass_rate']*100:.2f}%)")
    print(f"  Invalid Rate:    {single_scores['invalid_rate']:.4f} ({single_scores['invalid_rate']*100:.2f}%)")
    print(f"  Illegal Rate:    {single_scores['illegal_rate']:.4f} ({single_scores['illegal_rate']*100:.2f}%)")
    
    print("\n" + "=" * 60)
    print("MULTI-TABLE RESULTS")
    print("=" * 60)
    print(f"  Count:           {multi_scores['count']}")
    print(f"  Pass Rate:       {multi_scores['pass_rate']:.4f} ({multi_scores['pass_rate']*100:.2f}%)")
    print(f"  Invalid Rate:    {multi_scores['invalid_rate']:.4f} ({multi_scores['invalid_rate']*100:.2f}%)")
    print(f"  Illegal Rate:    {multi_scores['illegal_rate']:.4f} ({multi_scores['illegal_rate']*100:.2f}%)")


if __name__ == "__main__":
    main()
