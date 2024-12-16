from core.utils import parse_response, validate_select_order, add_group_by, parse_code_from_string, parse_vql_from_string, extract_world_info, is_email, is_valid_date_column
from func_timeout import FunctionTimedOut

LLM_API_FUC = None

try:
    from core import api

    LLM_API_FUC = api.safe_call_llm
    print(f"Use func from core.api in agents.py")
except:
    from core import llm

    LLM_API_FUC = llm.safe_call_llm
    print(f"Use func from core.llm in agents.py")

from core.const import *
from typing import List

import matplotlib.pyplot as plt
import os
import duckdb
import seaborn as sns
import sqlglot
import re
import time
import abc
import sys
import io
import traceback
from contextlib import redirect_stdout, redirect_stderr
import pandas as pd
from pathlib import Path
from tqdm import trange


class BaseAgent(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    def talk(self, message: dict):
        pass


class Processor(BaseAgent):
    """
    Get database description, extract relative tables & columns, generate augmented explanation and query difficulty
    """
    name = PROCESSOR_NAME
    description = "Get database description, extract relative tables & columns, generate augmented explanation and query difficulty"

    def __init__(self):
        super().__init__()
        self._message = {}

    def _load_table_info(self, table_path: str):
        table = pd.read_csv(table_path)
        table_name = Path(table_path).stem
        column_names = table.columns.tolist()
        column_types = [str(dtype) for dtype in table.dtypes]
        value_count = len(table)

        return {
            'table_name': table_name,
            'column_names': column_names,
            'column_types': column_types,
            'value_count': value_count
        }

    def _get_column_attributes(self, table_path):  
        table = pd.read_csv(table_path)
        column_names = table.columns.tolist()
        column_types = [str(dtype) for dtype in table.dtypes]
        return column_names, column_types

    def _get_unique_column_values_str(self, table_path, column_names, column_types): 
        table = pd.read_csv(table_path)
        col_to_values_str_lst = []
        col_to_values_str_dict = {}
        for idx, column_name in enumerate(column_names):
           
            lower_column_name: str = column_name.lower()
            # if lower_column_name ends with [id, email, url], just use empty str
            if lower_column_name.endswith('email') or \
                    lower_column_name.endswith('url'):
                values_str = ''
                col_to_values_str_dict[column_name] = values_str
                continue

            grouped = table.groupby(column_name)  
            group_counts = grouped.size()  
            sorted_counts = group_counts.sort_values(ascending=False) 
            values = sorted_counts.index.values  
            dtype = sorted_counts.index.dtype  

            values_str = ''
            # try to get value examples str, if exception, just use empty str
            try:
                values_str = self._get_value_examples_str(values, column_types[idx])
            except Exception as e:
                print(f"\nerror: get_value_examples_str failed, Exception:\n{e}\n")

            col_to_values_str_dict[column_name] = values_str
        
        for column_name in column_names:
            values_str = col_to_values_str_dict.get(column_name, '')
            col_to_values_str_lst.append([column_name, values_str])
        return col_to_values_str_lst

    def _get_value_examples_str(self, values: List[object], col_type: str): 
        if not len(values):
            return ''

        vals = []
        has_null = False
        for v in values:
            if v is None:
                has_null = True
            else:
                tmp_v = str(v).strip()
                if tmp_v == '':
                    continue
                else:
                    vals.append(v)
        if not vals:
            return ''

        if len(values) > 10 and col_type in ['int64', 'float64']:
            vals = vals[:4] 
            if has_null:
                vals.insert(0, None)
            return str(vals)

        # drop meaningless values of text type
        if col_type == 'object':
            new_values = []
            for v in vals:
                if not isinstance(v, str):
                    new_values.append(v)
                else:
                    if v == '':  # exclude empty string
                        continue
                    elif ('https://' in v) or ('http://' in v):  # exclude url
                        return ''
                    elif is_email(v):  # exclude email
                        return ''
                    else:
                        new_values.append(v)
            vals = new_values
            tmp_vals = [len(str(a)) for a in vals]
            if not tmp_vals:
                return ''
            max_len = max(tmp_vals)
            if max_len > 50:
                return ''
        if not vals:
            return ''
        vals = vals[:6]
        is_date_column = is_valid_date_column(vals)
        if is_date_column:
            vals = vals[:1]
        if has_null:
            vals.insert(0, None)
        val_str = str(vals)
        return val_str

    def _load_db_info(self, tables: List[str]) -> dict:
        table2coldescription = {}
        table_unique_column_values = {}

        for table_path in tables:
            table_info = self._load_table_info(table_path)
            table_name = table_info['table_name']

            col2dec_lst = []
            all_column_names, all_column_types = self._get_column_attributes(table_path)
            col_values_str_lst = self._get_unique_column_values_str(table_path, all_column_names, all_column_types)
            table_unique_column_values[table_name] = col_values_str_lst

            for x, column_name in enumerate(all_column_names):
                lower_column_name = column_name.lower()
                column_desc = ''
                col_type = all_column_types[x]
                if lower_column_name.endswith('id'):
                    column_desc = 'this is an id type column'
                elif lower_column_name.endswith('url'):
                    column_desc = 'this is a url type column'
                elif lower_column_name.endswith('email'):
                    column_desc = 'this is an email type column'
                elif table_info['value_count'] > 10 and col_type in ['int64', 'float64'] and col_values_str_lst[x][
                    1] == '':
                    column_desc = 'this is a number type column'

                full_col_name = column_name.replace('_', ' ').lower()
                col2dec_lst.append([full_col_name, column_desc])

            table2coldescription[table_name] = col2dec_lst

        result = {
            "desc_dict": table2coldescription,
            "value_dict": table_unique_column_values
        }
        return result

    def _build_table_schema_list_str(self, table_name, new_columns_desc, new_columns_val):
        
        table_desc: str = table_name.lower()
        table_desc = table_desc.replace('_', ' ')
        schema_desc_str = ''
        schema_desc_str += f"# Table: {table_name}, ({table_desc})\n"
        extracted_column_infos = []
        for (col_full_name, col_extra_desc), (col_name, col_values_str) in zip(new_columns_desc, new_columns_val):
            col_extra_desc = 'And ' + str(col_extra_desc) if col_extra_desc != '' and str(
                col_extra_desc) != 'nan' else ''
            col_extra_desc = col_extra_desc[:100]

            col_line_text = ''
            col_line_text += f'  ('
            col_line_text += f"{col_name}, "
            col_line_text += f"{col_full_name},"
            if col_values_str != '':
                col_line_text += f" Value examples: {col_values_str}."
            if col_extra_desc != '':
                col_line_text += f" {col_extra_desc}"
            col_line_text += '),'
            extracted_column_infos.append(col_line_text)
        schema_desc_str += '[\n' + '\n'.join(extracted_column_infos).strip(',') + '\n]' + '\n'
        return schema_desc_str

    def _get_db_desc_str(self, tables: List[str]):
        db_info = self._load_db_info(tables)
        desc_info = db_info['desc_dict']
        value_info = db_info['value_dict']

        schema_desc_str = ''
        for table_name in desc_info.keys():
            columns_desc = desc_info[table_name]
            columns_val = value_info[table_name]
            new_columns_desc = columns_desc.copy()
            new_columns_val = columns_val.copy()

            schema_desc_str += self._build_table_schema_list_str(table_name, new_columns_desc, new_columns_val)

        return schema_desc_str.strip()

    def _process(self, db_id: str, query: str, db_schema: str) -> dict:
        
        prompt = processor_template.format(db_id=db_id, query=query, db_schema=db_schema)
        world_info = extract_world_info(self._message)
        reply = LLM_API_FUC(prompt, **world_info)
        result = parse_response(reply)
        return result

    def talk(self, message: dict):
        """
        :param message: {"db_id": database_name,
                        "query": user_query,
                        "extracted_schema": None if no preprocessed result found}
        :return: extracted database schema {"desc_str": extracted_db_schema}
        """
        if message['send_to'] != self.name: return
        self._message = message
        query, tables, db_id = message.get('query'), message.get('tables'), message.get('db_id')
        db_schema = self._get_db_desc_str(tables)

        # # without processor
        # message['old_schema'] = db_schema
        # message['send_to'] = COMPOSER_NAME
        # return

        try:
            result = self._process(db_id=db_id ,query=query, db_schema=db_schema)
        except Exception as e:
            print(e)
            result = {
                "filtered_schema": {},
                "new_schema": db_schema,
                "augmented_explanation": "",
                "query_difficulty": "0",
            }
        print(f"query: {message['query']}\n")
        message['old_schema'] = db_schema
        message['filtered_schema'] = result["filtered_schema"]
        message['new_schema'] = result['new_schema']
        message['augmented_explanation'] = result['augmented_explanation']
        message['query_difficulty'] = result['query_difficulty']
        message['send_to'] = COMPOSER_NAME


class Composer(BaseAgent):
    """
    Decompose the question and solve them using CoT
    """
    name = COMPOSER_NAME
    description = "Decompose the question and solve them using CoT"

    def __init__(self):
        super().__init__()
        self._message = {}

    def talk(self, message: dict):
        """
        :param self:
        :param message: {"query": user_query,
                        "new_schema": description of db schema
                        "query_difficulty": difficulty level of the query}
        :return: decompose question into sub ones and solve them in generated VQL
        """
        if message['send_to'] != self.name: return
        self._message = message

        query, schema_info, augmented_explanation = message.get('query'), message.get('new_schema'), message.get(
            'augmented_explanation')
        query_difficulty = message.get('query_difficulty', "0")
        if query_difficulty == "SINGLE":
            prompt_template = single_template
            schema_info = message.get('old_schema')
            prompt = prompt_template.format(query=query, desc_str = schema_info)
        else:
            prompt_template = multiple_template
            prompt = prompt_template.format(query=query, desc_str=schema_info, augmented_explanation=augmented_explanation)

        # # without processor
        # query, schema_info = message.get('query'), message.get('old_schema')
        # prompt = single_template.format(query=query,desc_str=schema_info)

        # # without composer
        # prompt = without_composer_template.format(query=query, desc_str=schema_info, augmented_explanation=augmented_explanation)

        warning = message.get('warning', False)
        if warning == True:
            prompt = "\n【WARNING】YOU MUST SELECT COLUMNS MORE THAN OR EQUAL TO 2 IN SQL!!!" + prompt \
                     + "\n【REMENBER】YOU MUST SELECT COLUMNS MORE THAN OR EQUAL TO 2 IN SQL!!!" \
                     + "\n【SOLUTION】You can add a column using aggregate functions for existing column"
        world_info = extract_world_info(self._message)
        reply = LLM_API_FUC(prompt, **world_info).strip()

        res = ''
        qa_pairs = reply

        try:
            res = parse_vql_from_string(reply)
            if res is None:
                print("No VQL found in the response")
        except Exception as e:
            res = f'error!: {str(e)}'
            print(res)
            time.sleep(1)

        vql = res
        sql_match = re.search(r'SELECT\s+.+', vql, re.IGNORECASE | re.DOTALL)
        try:
            sql = sql_match.group(0)
            sql = re.sub(r'\s+BIN\s+.*?BY\s+\w+', '', sql)
            parsed_sql = sqlglot.parse_one(sql)
            select = parsed_sql.find(sqlglot.exp.Select)
            select_exprs = select.expressions
            if len(select_exprs) < 2 and warning == False:
                message['send_to'] = COMPOSER_NAME
                message['warning'] = True
                return
        except Exception as e:
            print(e)

        message['final_vql'] = res
        message['qa_pairs'] = qa_pairs
        message['fixed'] = False
        message['improved'] = False
        message['send_to'] = VALIDATOR_NAME


class Validator(BaseAgent):
    # Translate VQL to python language using library{matplotlib, seaborn}
    # Execute python to plot and perform validation
    name = VALIDATOR_NAME
    description = "Translate VQL to python language using library{matplotlib, seaborn},and execute to perform validation"

    def __init__(self, data_path: str):
        super().__init__()
        self.data_path = data_path  # path to all databases
        self._message = {}

    def _translate_plus(self, db_path: str, vql: str,
                        library="matplotlib"):  # translate vql to python code, with stacked bar chart
        try:
            
            vis_match = re.search(r'Visualize\s+([\w\s]+)\s+SELECT', vql, re.IGNORECASE)
            if vis_match:
                vis_type = vis_match.group(1).upper().strip()
            else:
                raise ValueError("Visualization type not found in VQL")

            
            bin_clause = re.search(r'BIN\s+(.*?)\s+BY\s+(\w+)', vql, re.IGNORECASE)

            
            sql_match = re.search(r'SELECT\s+.+', vql, re.IGNORECASE | re.DOTALL)
            if sql_match:
                sql = sql_match.group(0)
            else:
                raise ValueError("SQL query not found in VQL")

            sql = re.sub(r'\s+BIN\s+.*?BY\s+\w+', '', sql)
            parsed_sql = sqlglot.parse_one(sql)

            
            select = parsed_sql.find(sqlglot.exp.Select)
            if not select:
                raise ValueError("SELECT statement not found in SQL")

            select_exprs = select.expressions
            x_col = select_exprs[0].alias_or_name
            group_col = select_exprs[2].alias_or_name
            y_col = select_exprs[1].alias_or_name

            
            if isinstance(select_exprs[-1], sqlglot.exp.AggFunc):
                agg_func = select_exprs[-1].key.lower()
                agg_arg = select_exprs[-1].this.name if hasattr(select_exprs[-1].this, 'name') else str(
                    select_exprs[-1].this)
                group_col = f"{agg_func}_{agg_arg}"
                select_exprs[-1] = select_exprs[-1].as_(group_col)
            if isinstance(select_exprs[1], sqlglot.exp.AggFunc):
                agg_func = select_exprs[1].key.lower()
                agg_arg = select_exprs[1].this.name if hasattr(select_exprs[1].this, 'name') else str(
                    select_exprs[1].this)
                y_col = f"{agg_func}_{agg_arg}"
                select_exprs[1] = select_exprs[1].as_(y_col)

            
            sql = parsed_sql.sql()
            sql = add_group_by(sql, group_col)

            bin_code = ''
            if bin_clause:
                bin_col, bin_type = bin_clause.groups()
                x_col = bin_col
                
                sql = add_group_by(sql, bin_col)

                bin_code += "# Apply binning operation\n"
                bin_code += "flag = True\n"  

                

                if bin_type.upper() == 'YEAR':
                    bin_code += f"""
is_datetime = pd.api.types.is_datetime64_any_dtype(df['{bin_col}'])
if is_datetime:
    df['{x_col}'] = df['{x_col}'].dt.year
else:
    df['{x_col}'] = df['{x_col}'].astype(int)
    flag = False
"""
                elif bin_type.upper() == 'MONTH':
                    bin_code += f"df['{bin_col}'] = pd.to_datetime(df['{bin_col}'])\n"
                    bin_code += f"df['{x_col}'] = df['{x_col}'].dt.strftime('%B')\n"
                elif bin_type.upper() == 'DAY':
                    bin_code += f"df['{bin_col}'] = pd.to_datetime(df['{bin_col}'])\n"
                    bin_code += f"df['{x_col}'] = df['{x_col}'].dt.date()\n"
                elif bin_type.upper() == 'WEEKDAY':
                    bin_code += f"df['{bin_col}'] = pd.to_datetime(df['{bin_col}'])\n"
                    bin_code += f"df['{x_col}'] = df['{x_col}'].dt.day_name()\n"

                
                parsed_sql = sqlglot.parse_one(sql)
                select_expr = parsed_sql.find(sqlglot.exp.Select)
                
                agg_func = 'size'
                
                for expr in select_expr.expressions:
                    if isinstance(expr, sqlglot.exp.Alias) and expr.alias == y_col:
                        if isinstance(expr.this, sqlglot.exp.Count):
                            agg_func = 'size'
                        elif isinstance(expr.this, sqlglot.exp.Sum):
                            agg_func = 'sum'
                        elif isinstance(expr.this, sqlglot.exp.Avg):
                            agg_func = 'mean'

                
                if agg_func == 'size':
                    bin_code += f"""
# Group by and calculate count
if flag:
    df = df.groupby(['{x_col}', '{group_col}']).sum().reset_index()
"""
                if agg_func == 'sum':
                    bin_code += f"""
# Group by and calculate sum
if flag:
    df = df.groupby(['{x_col}', '{group_col}']).sum().reset_index()
"""
                if agg_func == 'mean':
                    bin_code += f"""
# Group by and calculate avg
if flag:
    df = df.groupby(['{x_col}', '{group_col}']).mean().reset_index()
"""

                
                if bin_type.upper() == 'WEEKDAY':
                    bin_code += f"""
# Ensure all seven days of the week are included
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

all_combinations = pd.MultiIndex.from_product([weekday_order, df['{group_col}'].unique()], 
                                              names=['{x_col}', '{group_col}'])

df = df.set_index(['{x_col}', '{group_col}'])
df = df.reindex(all_combinations, fill_value=0).reset_index()
df['{x_col}'] = pd.Categorical(df['{x_col}'], categories=weekday_order, ordered=True)
"""
                elif bin_type.upper() == 'MONTH':
                    bin_code += f"""
# Sort months in chronological order, but only include existing months
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                'July', 'August', 'September', 'October', 'November', 'December']
existing_months = df['{x_col}'].unique()
ordered_existing_months = [month for month in month_order if month in existing_months]
df['{x_col}'] = pd.Categorical(df['{x_col}'], categories=ordered_existing_months, ordered=True)
"""
               
                order_by = parsed_sql.find(sqlglot.exp.Order)
                if order_by:
                    sort_columns = []
                    sort_ascending = []
                    for expr in order_by.expressions:
                        
                        original_col_name = expr.this.name if isinstance(expr.this, sqlglot.exp.Column) else str(
                            expr.this)
                        
                        if original_col_name == bin_col:
                            col_name = x_col
                        elif isinstance(expr.this, sqlglot.exp.AggFunc):
                            
                            agg_func = expr.this.key.lower()
                            agg_arg = expr.this.this.name if hasattr(expr.this.this, 'name') else str(expr.this.this)
                            col_name = f"{agg_func}_{agg_arg}"
                        else:
                            col_name = original_col_name
                        sort_columns.append(col_name)
                        
                        sort_ascending.append(expr.args.get('desc', False) == False)

                    sort_columns_str = ", ".join([f"'{col}'" for col in sort_columns])
                    sort_ascending_str = ", ".join(map(str, sort_ascending))

                    bin_code += f"""
# Ensure sorting columns exist in the DataFrame
sort_columns = [{sort_columns_str}]
sort_columns = [col for col in sort_columns if col in df.columns]
if sort_columns:
    df = df.sort_values(sort_columns, ascending=[{sort_ascending_str}])
else:
    print("Warning: Specified sorting columns not found in the DataFrame. No sorting applied.")
    df = df.sort_values(['{group_col}', '{x_col}'])
"""
                else:
                   
                    bin_code += f"df = df.sort_values(['{group_col}', '{x_col}'])\n"

            
            pivot = False
            vis_code = ""
            if "BAR" in vis_type:
                if library == 'matplotlib':
                    pivot = True
                    
                    vis_code += f"""
fig,ax = plt.subplots(1,1,figsize=(10,4))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
df_pivot = df.pivot(index='{x_col}', columns='{group_col}', values='{y_col}')
df_pivot.plot(kind='bar', stacked=True, ax=ax, alpha=0.8)
ax.set_xlabel('{x_col}')
ax.set_ylabel('{y_col}')
ax.set_title('Stacked Bar Chart of {y_col} by {x_col} and {group_col}')
ax.legend(title='{group_col}', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
"""
                elif library == 'seaborn':
                    vis_code = f"""
fig,ax = plt.subplots(1,1,figsize=(10,4))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
sns.barplot(x='{x_col}', y='{y_col}', hue='{group_col}', data=df, ax=ax, alpha=0.8)
ax.set_xlabel('{x_col}')
ax.set_ylabel('{y_col}')
ax.set_title('Stacked Bar Chart of {y_col} by {x_col} and {group_col}')
ax.legend(title='{group_col}', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
"""
            elif "LINE" in vis_type:
                if library == 'matplotlib':
                    vis_code = f"""
fig,ax = plt.subplots(1,1,figsize=(10,4))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for group in df['{group_col}'].unique():
    group_data = df[df['{group_col}'] == group]
    ax.plot(group_data['{x_col}'], group_data['{y_col}'], label=group, marker='o', alpha=0.7)
ax.set_xlabel('{x_col}')
ax.set_ylabel('{y_col}')
ax.set_title('Grouped Line Chart of {y_col} by {x_col} and {group_col}')
ax.legend(title='{group_col}', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
"""
                elif library == 'seaborn':
                    vis_code = f"""
fig,ax = plt.subplots(1,1,figsize=(10,4))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
sns.lineplot(x='{x_col}', y='{y_col}', hue='{group_col}', data=df, marker='o', ax=ax, alpha=0.7)
ax.set_xlabel('{x_col}')
ax.set_ylabel('{y_col}')
ax.set_title('Grouped Line Chart of {y_col} by {x_col} and {group_col}')
ax.legend(title='{group_col}', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
"""
            elif "SCATTER" in vis_type:
                if library == 'matplotlib':
                    vis_code = f"""
fig,ax = plt.subplots(1,1,figsize=(10,4))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for group in df['{group_col}'].unique():
    group_data = df[df['{group_col}'] == group]
    ax.scatter(group_data['{x_col}'], group_data['{y_col}'], label=group, alpha=0.6, s=80, edgecolor='k')
ax.set_xlabel('{x_col}')
ax.set_ylabel('{y_col}')
ax.set_title('Grouped Scatter Plot of {y_col} vs {x_col} by {group_col}')
ax.legend(title='{group_col}', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
"""
                elif library == 'seaborn':
                    vis_code = f"""
fig,ax = plt.subplots(1,1,figsize=(10,4))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
sns.scatterplot(x='{x_col}', y='{y_col}', hue='{group_col}', data=df, ax=ax, alpha=0.6, s=80, edgecolor='k')
ax.set_xlabel('{x_col}')
ax.set_ylabel('{y_col}')
ax.set_title('Grouped Scatter Plot of {y_col} vs {x_col} by {group_col}')
ax.legend(title='{group_col}', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
"""
           
            python_code = ''
            if library == "seaborn":
                python_code += "import seaborn as sns\n"
            python_code += f"""
import matplotlib.pyplot as plt
import pandas as pd
import os
import duckdb


data_folder = '{db_path}'


con = duckdb.connect(database=':memory:')


csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
for file in csv_files:
    table_name = os.path.splitext(file)[0]
    con.execute(f"CREATE VIEW {{table_name}} AS SELECT * FROM read_csv_auto('{{os.path.join(data_folder, file)}}')")


sql = f'''
{sql}
'''
df = con.execute(sql).fetchdf()


df.columns = ['{x_col}', '{y_col}', '{group_col}']


# print("Columns in the dataframe:", df.columns)


{bin_code}


{vis_code}
"""
            if pivot:
                python_code += f"""
# Print data
x_data = [df.index.tolist()]
y_data = [df[col].tolist() for col in df.columns]
print("x_data:", x_data)
print("y_data:", y_data)
print("groups:", df.columns.tolist())
"""
            else:
                python_code += f"""
# Print data
print("x_data (unique):", df['{x_col}'].unique().tolist())
print("group_data (unique):", df['{group_col}'].unique().tolist())
print("y_data (sum for each group):", df.groupby(['{x_col}', '{group_col}'])['{y_col}'].sum().to_dict())
"""
            return python_code

        except Exception as e:
            print(f"Error in _translate function: {e}")
            print(f"VQL: {vql}")
            print(f"Parsed SQL: {sql}")
            
            return f"Error occurred while processing the query: {str(e)}"

    def _translate_normal(self, db_path: str, vql: str,
                          library="matplotlib"):  # translate vql to python code using matplotlib or seaborn
        try:
            
            vis_type = re.search(r'Visualize\s+(\w+)', vql, re.IGNORECASE).group(1)
            
            bin_clause = re.search(r'BIN\s+(.*?)\s+BY\s+(\w+)', vql, re.IGNORECASE)

            
            sql = re.sub(r'Visualize\s+\w+\s+', '', vql)
            sql = re.sub(r'\s+BIN\s+.*?BY\s+\w+', '', sql)
            parsed_sql = sqlglot.parse_one(sql)  

            
            select = parsed_sql.find(sqlglot.exp.Select)
            if not select:
                raise ValueError("SELECT statement not found in SQL")

            select_exprs = select.expressions
            if len(select_exprs) < 2:
                raise ValueError(
                    f"Not enough expressions in SELECT statement. Found {len(select_exprs)}, expected at least 2")
            # print(select_exprs)
            x_col = select_exprs[0].alias_or_name
            y_col = select_exprs[1].alias_or_name

            
            if isinstance(select_exprs[1], sqlglot.exp.AggFunc):
                agg_func = select_exprs[1].key.lower()
                agg_arg = select_exprs[1].this.name if hasattr(select_exprs[1].this, 'name') else str(
                    select_exprs[1].this)
                y_col = f"{agg_func}_{agg_arg}"
                select_exprs[1] = select_exprs[1].as_(y_col)

            
            sql = parsed_sql.sql()

            bin_code = ''
            if bin_clause:
                bin_col, bin_type = bin_clause.groups()
                x_col = bin_col
               
                sql = add_group_by(sql, bin_col)

                bin_code += "# Apply binning operation\n"
                bin_code += "flag = True\n"

            
                if bin_type.upper() == 'YEAR':
                    bin_code += f"""
is_datetime = pd.api.types.is_datetime64_any_dtype(df['{bin_col}'])
if is_datetime:
    df['{x_col}'] = df['{x_col}'].dt.year
else:
    df['{x_col}'] = df['{x_col}'].astype(int)
    flag = False
"""
                elif bin_type.upper() == 'MONTH':
                    bin_code += f"df['{bin_col}'] = pd.to_datetime(df['{bin_col}'])\n"
                    bin_code += f"df['{x_col}'] = df['{x_col}'].dt.strftime('%B')\n"
                elif bin_type.upper() == 'DAY':
                    bin_code += f"df['{bin_col}'] = pd.to_datetime(df['{bin_col}'])\n"
                    bin_code += f"df['{x_col}'] = df['{x_col}'].dt.date\n"
                elif bin_type.upper() == 'WEEKDAY':
                    bin_code += f"df['{bin_col}'] = pd.to_datetime(df['{bin_col}'])\n"
                    bin_code += f"df['{x_col}'] = df['{x_col}'].dt.day_name()\n"

               
                parsed_sql = sqlglot.parse_one(sql)
                select_expr = parsed_sql.find(sqlglot.exp.Select)
                
                agg_func = 'size'
               
                for expr in select_expr.expressions:
                    if isinstance(expr, sqlglot.exp.Alias) and expr.alias == y_col:
                        if isinstance(expr.this, sqlglot.exp.Count):
                            agg_func = 'size'
                        elif isinstance(expr.this, sqlglot.exp.Sum):
                            agg_func = 'sum'
                        elif isinstance(expr.this, sqlglot.exp.Avg):
                            agg_func = 'mean'

                
                if agg_func == 'size':
                    bin_code += f"""
# Group by and calculate count
if flag:
    df = df.groupby('{x_col}').sum().reset_index()
"""
                if agg_func == 'sum':
                    bin_code += f"""
# Group by and calculate sum
if flag:
    df = df.groupby('{x_col}').sum().reset_index()
"""
                if agg_func == 'mean':
                    bin_code += f"""
# Group by and calculate avg
if flag:
    df = df.groupby('{x_col}').mean().reset_index()
"""
                
                if bin_type.upper() == 'WEEKDAY':
                    bin_code += f"""
# Ensure all seven days of the week are included
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df = df.set_index('{x_col}').reindex(weekday_order, fill_value=0).reset_index()
df['{x_col}'] = pd.Categorical(df['{x_col}'], categories=weekday_order, ordered=True)
"""

                order_by = parsed_sql.find(sqlglot.exp.Order)
                if order_by:
                    sort_columns = []
                    sort_ascending = []
                    for expr in order_by.expressions:
                        
                        original_col_name = expr.this.name if isinstance(expr.this, sqlglot.exp.Column) else str(
                            expr.this)
                        
                        if original_col_name == bin_col:
                            col_name = x_col
                        elif isinstance(expr.this, sqlglot.exp.AggFunc):
                            
                            agg_func = expr.this.key.lower()
                            agg_arg = expr.this.this.name if hasattr(expr.this.this, 'name') else str(expr.this.this)
                            col_name = f"{agg_func}_{agg_arg}"
                        else:
                            col_name = original_col_name
                        sort_columns.append(col_name)
                       
                        sort_ascending.append(expr.args.get('desc', False) == False)

                    sort_columns_str = ", ".join([f"'{col}'" for col in sort_columns])
                    sort_ascending_str = ", ".join(map(str, sort_ascending))

                    bin_code += f"""
# Ensure sorting columns exist in the DataFrame
sort_columns = [{sort_columns_str}]
sort_columns = [col for col in sort_columns if col in df.columns]
if sort_columns:
    df = df.sort_values(sort_columns, ascending=[{sort_ascending_str}])
else:
    print("Warning: Specified sorting columns not found in the DataFrame. No sorting applied.")
    df = df.sort_values('{x_col}')
"""
                else:
                    
                    if bin_type.upper() == 'MONTH':
                        bin_code += f"""
# Sort months in chronological order
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                'July', 'August', 'September', 'October', 'November', 'December']
df['{x_col}'] = pd.Categorical(df['{x_col}'], categories=month_order, ordered=True)
df = df.sort_values('{x_col}')
"""
                    elif bin_type.upper() == 'WEEKDAY':
                        bin_code += f"df = df.sort_values('{x_col}')\n"
                    else:
                        bin_code += f"df = df.sort_values('{x_col}')\n"

           
            vis_code = ""
            if library == 'matplotlib':
                vis_code = f"""
fig,ax = plt.subplots(1,1,figsize=(10,4))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
{"ax.bar(df['" + x_col + "'], df['" + y_col + "'])" if vis_type == 'BAR' else ""}
{"ax.plot(df['" + x_col + "'], df['" + y_col + "'])" if vis_type == 'LINE' else ""}
{"ax.scatter(df['" + x_col + "'], df['" + y_col + "'])" if vis_type == 'SCATTER' else ""}
{"ax.pie(df['" + y_col + "'], labels=df['" + x_col + "'], autopct='%1.1f%%')" if vis_type == 'PIE' else ""}
ax.set_xlabel('{x_col}')
ax.set_ylabel('{y_col}')
ax.set_title(f'{vis_type} Chart of {y_col} by {x_col}')
{"plt.xticks(rotation=45)" if vis_type != 'PIE' else ""}
plt.tight_layout()
"""

            elif library == 'seaborn':
                vis_code = f"""
fig,ax = plt.subplots(1,1,figsize=(10,4))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
{"sns.barplot(x='" + x_col + "', y='" + y_col + "', data=df, ax=ax)" if vis_type == 'BAR' else ""}
{"sns.lineplot(x='" + x_col + "', y='" + y_col + "', data=df, ax=ax)" if vis_type == 'LINE' else ""}
{"sns.scatterplot(x='" + x_col + "', y='" + y_col + "', data=df, ax=ax)" if vis_type == 'SCATTER' else ""}
{"ax.pie(df['" + y_col + "'], labels=df['" + x_col + "'], autopct='%1.1f%%')" if vis_type == 'PIE' else ""}
ax.set_xlabel('{x_col}')
ax.set_ylabel('{y_col}')
ax.set_title('{vis_type} Chart of {y_col} by {x_col}')
{"plt.xticks(rotation=45)" if vis_type != 'PIE' else ""}
sns.despine()
plt.tight_layout()
"""
            
            python_code = ''
            if library == "seaborn":
                python_code += "import seaborn as sns\n"
            python_code += f"""
import matplotlib.pyplot as plt
import pandas as pd
import os
import duckdb


data_folder = '{db_path}'


con = duckdb.connect(database=':memory:')


csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
for file in csv_files:
    table_name = os.path.splitext(file)[0]
    con.execute(f"CREATE VIEW {{table_name}} AS SELECT * FROM read_csv_auto('{{os.path.join(data_folder, file)}}')")


sql = f'''
{sql}
'''
df = con.execute(sql).fetchdf()
con.close()

# rename columns
df.columns = ['{x_col}','{y_col}']


# print("Columns in the dataframe:", df.columns)


{bin_code}


{vis_code}

# Print data
print("x_data:", df['{x_col}'].tolist())
print("y_data:", df['{y_col}'].tolist())
"""
            return python_code

        except Exception as e:
            print(f"Error in _translate function: {e}")
            print(f"VQL: {vql}")
            print(f"Parsed SQL: {sql}")
            
            return f"Error occurred while processing the query: {str(e)}"

    def _translate(self, db_path: str, vql: str, library="matplotlib"):
        try:
            match = re.search(r'Visualize\s+([\w\s]+)\s+SELECT\s+(.*?)\s+FROM', vql, re.IGNORECASE | re.DOTALL)
            if not match:
                return False

            vis_type = match.group(1).upper().strip()
            select_columns = [col.strip() for col in match.group(2).split(',')]

            if vis_type in ["STACKED BAR", "GROUPED LINE", "GROUPED SCATTER"] or len(select_columns) == 3:
                return self._translate_plus(db_path, vql, library)
            else:
                return self._translate_normal(db_path, vql, library)
        except Exception as e:
            print("error in translate:", e)

    def _execute_python_code(self, code):
        
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

      
        result = {
            'output': '',
            'error': ''
        }

        
        exec_globals = {}

        original_show = plt.show
        def dummy_show(*args, **kwargs):
            pass
        plt.show = dummy_show

        try:
            
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                
                exec(code, exec_globals)
           
            result['output'] = stdout_capture.getvalue()


        except Exception as e:
           
            result['error'] = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"

        finally:
           
            plt.show = original_show

        
        result['error'] += stderr_capture.getvalue()

        return result

    def _is_need_refine(self, exec_result: dict): 
        flag = False
        if exec_result['error']:
            flag = True
            if "UserWarning: set_ticklabels()" in exec_result['error']:
                flag = False
            if "RuntimeWarning" in exec_result['error']:
                flag = False
            if "UserWarning: Tight layout not applied" in exec_result['error']:
                flag = False
            if "FutureWarning" in exec_result['error']:
                flag = False
        return flag

    def _refine_vql(self, nl_query: str, vql: str, db_info, exec_result: dict):
        error = exec_result['error']
        prompt = refiner_vql_template.format(query=nl_query, db_info=db_info, vql=vql, error=error)
        world_info = extract_world_info(self._message)
        reply = LLM_API_FUC(prompt, **world_info)
        new_vql = parse_vql_from_string(reply)
        return new_vql

    def _refine_python(self, nl_query: str, code: str, db_info, exec_result: dict):
        error = exec_result['error']
        prompt = refiner_python_template.format(query=nl_query, db_info=db_info, code=code, error=error)
        world_info = extract_world_info(self._message)
        reply = LLM_API_FUC(prompt, **world_info)
        new_code = parse_code_from_string(reply)
        return new_code

    def talk(self, message: dict):
        if message['send_to'] != self.name: return
        self._message = message
        db_id, vql, query= message.get('db_id'), message.get('final_vql'), message.get('query')
        db_info = message.get('new_schema')

        db_path = f"{self.data_path}/{db_id}"
        library = message.get('library', 'matplotlib')
        code = self._translate(db_path, vql, library)

        # code = message.get('pred', self._translate(db_path, vql, library))

        # # without validator
        # message['pred'] = code
        # message['send_to'] = SYSTEM_NAME
        # return

        # print(code)
        # do not fix vql containing "error" string
        if 'error!' in vql:
            message['try_times'] = message.get('try_times', 0) + 1
            message['pred'] = code
            message['send_to'] = SYSTEM_NAME
            return

        is_timeout = False
        try:
            exec_result = self._execute_python_code(code)
            print(exec_result)
        except Exception as e:
            # print(e)
            exec_result = {'output': '', 'error': ''}
            is_timeout = True
        except FunctionTimedOut as fto:
            # print(fto)
            exec_result = {'output': '', 'error': ''}
            is_timeout = True

        is_need_refine = self._is_need_refine(exec_result)

        # # refine python code
        # if is_timeout:
        #     message['try_times'] = message.get('try_times', 0) + 1
        #     message['pred'] = code
        #     message['send_to'] = SYSTEM_NAME
        # elif not is_need_refine:
        #     message['try_times'] = message.get('try_times', 0) + 1
        #     message['pred'] = code
        #     message['send_to'] = SYSTEM_NAME
        # else:
        #     new_code = self._refine_python(query, code, db_info, exec_result)
        #     message['try_times'] = message.get('try_times', 0) + 1
        #     message['pred'] = new_code
        #     message['fixed'] = True
        #     message['send_to'] = VALIDATOR_NAME  # Send back to Refiner for another try


        if not is_need_refine:
            if not validate_select_order(vql):
                is_need_refine = True
                exec_result[
                    'error'] = "Incorrect select column numbers! If there are 3 columns, please use STACKED BAR, GROUPED LINE, or GROUPED SCATTER"
        if is_timeout:
            message['try_times'] = message.get('try_times', 0) + 1
            message['pred'] = code
            message['send_to'] = SYSTEM_NAME
        elif not is_need_refine:
            message['try_times'] = message.get('try_times', 0) + 1
            message['pred'] = code
            message['send_to'] = SYSTEM_NAME
        else:
            new_vql = self._refine_vql(query, vql, db_info, exec_result)
            message['try_times'] = message.get('try_times', 0) + 1
            message['final_vql'] = new_vql
            message['pred'] = code
            message['fixed'] = True
            message['send_to'] = VALIDATOR_NAME  # Send back to Refiner for another try
        return


if __name__ == "__main__":
    m = 0
