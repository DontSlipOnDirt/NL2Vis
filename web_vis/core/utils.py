# -*- coding: utf-8 -*-
import os
import re
import random
import json
import time
import sqlite3
from typing import Dict
import sqlglot


def parse_response(response: str) -> Dict:
    # 实现解析响应的逻辑，提取关键信息
    # 这里需要根据实际输出格式进行调整
    result = {
        "filtered_schema": {},
        "new_schema": "",
        "augmented_explanation": "",
        "query_difficulty": "",
    }

    # 解析 Filtered Schema
    filtered_schema_match = re.search(r'【Filtered Schema】\n(.*?)\n\n【New Schema】', response, re.DOTALL)
    if filtered_schema_match:
        result["filtered_schema"] = filtered_schema_match.group(1).strip()

    # 解析 database Schema
    new_schema_match = re.search(r'【New Schema】\n(.*?)\n\n【Augmented Explanation】', response, re.DOTALL)
    if new_schema_match:
        result["new_schema"] = new_schema_match.group(1).strip()

    # 解析 Format Explanation
    augmented_explanation_match = re.search(r'【Augmented Explanation】\n(.*?)\n\n【Classification】', response, re.DOTALL)
    if augmented_explanation_match:
        result["augmented_explanation"] = augmented_explanation_match.group(1).strip()

    # 解析 Query Difficulty
    query_difficulty_match = re.search(r'【Classification】\n(\w+)', response)
    if query_difficulty_match:
        result["query_difficulty"] = query_difficulty_match.group(1).strip()

    return result


def has_order_by(vql):
    return bool(re.search(r'\bORDER\s+BY\b', vql, re.IGNORECASE))

def validate_select_order(vql: str) -> bool:
    # 解析VQL
    match = re.search(r'Visualize\s+([\w\s]+)\s+SELECT\s+(.*?)\s+FROM', vql, re.IGNORECASE | re.DOTALL)
    if not match:
        return False

    vis_type = match.group(1).upper().strip()
    select_columns = [col.strip() for col in match.group(2).split(',')]

    if vis_type in ['BAR', 'PIE', 'LINE', 'SCATTER']:
        return len(select_columns) == 2
    elif vis_type in ['STACKED BAR', 'GROUPED LINE', 'GROUPED SCATTER']:
        return len(select_columns) == 3
    else:
        return False

def add_group_by(sql, new_group_by_column):
    # 解析 SQL 语句
    parsed = sqlglot.parse_one(sql)

    # 查找现有的 GROUP BY 语句
    group_by = parsed.find(sqlglot.expressions.Group)

    # 如果已有 GROUP BY 语句，添加新的列
    if group_by:
        # 获取当前 GROUP BY 中的所有列
        group_by_columns = group_by.expressions

        # 检查新列是否已经存在
        if not any(col.name == new_group_by_column for col in group_by_columns):
            # 如果新列不存在，则添加
            group_by_columns.append(sqlglot.exp.Column(this=new_group_by_column))
    else:
        # 如果没有 GROUP BY 语句，则创建一个新的
        group_by = sqlglot.exp.Group(expressions=[sqlglot.exp.Column(this=new_group_by_column)])
        parsed.set("group", group_by)  # 设置新的 GROUP BY

    # 生成新的 SQL
    return parsed.sql()

def show_svg(plt, svg_name: str):
    """Show a plot as a SVG inline."""
    from io import StringIO
    f = StringIO()
    plt.savefig(f, format="svg")
    if svg_name:
        plt.savefig(f"{svg_name}")
    svg_content = f.getvalue()
    plt.close()

    return svg_content

def parse_vql_from_string(response: str):
    # 使用正则表达式查找以 "Visualize" 开头的最后一句话
    vql_matches = re.findall(r'Visualize\s+.*', response, re.IGNORECASE | re.MULTILINE)
    if vql_matches:
        # 返回最后一个匹配项，即最后一个VQL语句
        return vql_matches[-1].strip()
    else:
        # 如果没有找到VQL语句，返回None或抛出异常

        return None  # 或者 raise ValueError("No VQL found in the response")

def parse_code_from_string(response: str):
    # 使用正则表达式查找所有的 ```python 和 ``` 之间的内容
    code_matches = re.findall(r'```python\n(.*?)\n```', response, re.DOTALL)

    if code_matches:
        # 返回最后一个匹配到的代码内容
        return code_matches[-1].strip()
    else:
        # 如果没有找到Python代码块，返回None或抛出异常
        return None  # 或者 raise ValueError("No Python code found in the response")

def is_valid_date(date_str):
    if (not isinstance(date_str, str)):
        return False
    date_str = date_str.split()[0]
    if len(date_str) != 10:
        return False
    pattern = r'^\d{4}-\d{2}-\d{2}$'
    if re.match(pattern, date_str):
        year, month, day = map(int, date_str.split('-'))
        if year < 1 or month < 1 or month > 12 or day < 1 or day > 31:
            return False
        else:
            return True
    else:
        return False


def is_valid_date_column(col_value_lst):
    for col_value in col_value_lst:
        if not is_valid_date(col_value):
            return False
    return True

def is_email(string):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    match = re.match(pattern, string)
    if match:
        return True
    else:
        return False

def extract_world_info(message_dict: dict):
    info_dict = {}
    info_dict['idx'] = message_dict.get('idx', 0)
    info_dict['query'] = message_dict['query']
    info_dict['difficulty'] = message_dict.get('difficulty', '')
    info_dict['ground_truth'] = message_dict.get('ground_truth', '')
    info_dict['send_to'] = message_dict.get('send_to', '')
    return info_dict

def load_json_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        print(f"load json file from {path}")
        return json.load(f)
