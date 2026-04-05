#!/usr/bin/env python3
"""
Analyze test results to understand illegal rate drivers
Usage: python results/analyze_results.py <result_directory>
Example: python results/analyze_results.py results/20251118_032946_Qwen2.5-7B-Instruct_NoVision
"""

import sys
import pandas as pd
import json
from pathlib import Path

# take result directory as command line argument
result_dir = Path(sys.argv[1])

# Read CSV
df = pd.read_csv(result_dir / "detailed_results.csv")

print("="*60)
print("ILLEGAL RATE ANALYSIS BY CHART TYPE")
print("="*60)
chart_breakdown = df.groupby('chart').agg({
    'illegal rate': ['mean', 'min', 'max', 'count'],
    'data check_fail_rate': 'mean',
    'deconstruction_fail_rate': 'mean',
    'chart type check_fail_rate': 'mean',
    'order check_fail_rate': 'mean'
}).round(3)
print(chart_breakdown)

print("\n" + "="*60)
print("ILLEGAL RATE BY SINGLE VS MULTI-TABLE")
print("="*60)
table_breakdown = df.groupby('is_multi_table').agg({
    'illegal rate': ['mean', 'min', 'max', 'count'],
    'data check_fail_rate': 'mean',
    'deconstruction_fail_rate': 'mean',
}).round(3)
print(table_breakdown)

print("\n" + "="*60)
print("DISTRIBUTION OF OUTCOMES")
print("="*60)
print(f"Total row count: {len(df)}")
print(f"Queries with 0 illegal rate: {len(df[df['illegal rate'] == 0])}")
print(f"Queries with 1.0 illegal rate: {len(df[df['illegal rate'] == 1.0])}")
print(f"Queries with pass_rate=1.0: {len(df[df['pass_rate'] == 1.0])}")
print(f"Queries with pass_rate=0: {len(df[df['pass_rate'] == 0])}")

print("\n" + "="*60)
print("MOST PROBLEMATIC COMBINATIONS")
print("="*60)
problem_combos = df.groupby(['chart', 'is_multi_table']).agg({
    'illegal rate': 'mean',
    'data check_fail_rate': 'mean',
}).round(3).sort_values('illegal rate', ascending=False)
print(problem_combos)

# Look at PIE charts specifically
print("\n" + "="*60)
print("PIE CHART ANALYSIS (All Queries)")
print("="*60)
pies = df[df['chart'] == 'Pie']
print(f"Total PIE queries: {len(pies)}")
print(f"Avg illegal rate: {pies['illegal rate'].mean():.3f}")
print(f"Avg data check failure: {pies['data check_fail_rate'].mean():.3f}")
print(f"Data check = 1.0 (all fail): {len(pies[pies['data check_fail_rate'] == 1.0])}")
print(f"Data check = 0 (all pass): {len(pies[pies['data check_fail_rate'] == 0])}")

print("\n" + "="*60)
print("SPECIFIC PIE QUERY IDS WITH ALL DATA FAILURES")
print("="*60)
bad_pies = pies[pies['data check_fail_rate'] == 1.0][['id', 'chart', 'is_multi_table', 'data check_fail_rate', 'illegal rate']].head(10)
print(bad_pies)

print("\n" + "="*60)
print("MULTI-TABLE PIE ANALYSIS")
print("="*60)
multi_pies = df[(df['chart'] == 'Pie') & (df['is_multi_table'] == True)]
print(f"Multi-table PIE queries: {len(multi_pies)}")
print(f"Avg illegal rate: {multi_pies['illegal rate'].mean():.3f}")
print(f"Avg data check failure: {multi_pies['data check_fail_rate'].mean():.3f}")

print("\nDone!")
