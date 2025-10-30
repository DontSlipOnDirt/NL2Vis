# -*- coding: utf-8 -*-
from core.agents import Processor, Composer, Validator
from core.const import MAX_ROUND, SYSTEM_NAME, PROVIDER_NAME
from core.utils import show_svg
from core.api_config import USE_LOCAL_VLLM
from viseval.dataset import Dataset
import matplotlib.pyplot as plt
import traceback

INIT_LOG__PATH_FUNC = None
LLM_API_FUC = None

if USE_LOCAL_VLLM:
    from core import vllm_client
    LLM_API_FUC = vllm_client.safe_call_llm
    INIT_LOG__PATH_FUNC = vllm_client.init_log_path
    print(f"Use vLLM from core.vllm_client in chat_manager.py")
else:
    try:
        from core import api
        LLM_API_FUC = api.safe_call_llm
        INIT_LOG__PATH_FUNC = api.init_log_path
        print(f"Use Azure OpenAI from core.api in chat_manager.py")
    except:
        from core import llm
        LLM_API_FUC = llm.safe_call_llm
        INIT_LOG__PATH_FUNC = llm.init_log_path
        print(f"Use Azure OpenAI from core.llm in chat_manager.py")

import time
from pprint import pprint
from typing import Optional
from attr import dataclass
from pathlib import Path

@dataclass
class ChartExecutionResult:
    """Response from a visualization execution"""

    # True if successful, False otherwise
    status: bool
    # Generate svg string if status is True
    svg_string: Optional[str] = None
    # Error message if status is False
    error_msg: Optional[str] = None

class ChatManager(object):
    def __init__(self, data_path: str, log_path: str):
        self.data_path = data_path + "/databases"
        self.log_path = log_path
        self.ping_network()
        self.chat_group = [
            Processor(),
            Composer(),
            Validator(data_path=self.data_path)
        ]
        INIT_LOG__PATH_FUNC(log_path)

    def ping_network(self):
        # check network status
        print("Checking network status...", flush=True)
        try:
            _ = LLM_API_FUC("Hello world!")
            print("Network is available", flush=True)
        except Exception as e:
            raise Exception(f"Network is not available: {e}")

    def _chat_single_round(self, message: dict):
        # we use `dict` type so value can be changed in the function
        for agent in self.chat_group:  # check each agent in the group
            if message['send_to'] == agent.name:
                agent.talk(message)

    def start(self, user_message: dict):
        start_time = time.time()
        if user_message['send_to'] == SYSTEM_NAME:  # in the first round, pass message to prune
            user_message['send_to'] = PROVIDER_NAME
        for _ in range(MAX_ROUND):  # start chat in group
            self._chat_single_round(user_message)
            if user_message['send_to'] == SYSTEM_NAME:  # should terminate chat
                break
        end_time = time.time()
        exec_time = end_time - start_time
        print(f"\033[0;34mExecute {exec_time} seconds\033[0m", flush=True)
        code = user_message.get('pred', "import matplotlib.pyplot as plt")
        code += "\nplt.show()"
        return code

    def execute_to_svg(self, code ,log_name:str = None ):
        global_env = {
            "svg_string": None,
            "show_svg": show_svg,
            "svg_name": log_name
        }

       
        original_show = plt.show

        code += "\nsvg_string = show_svg(plt,svg_name)"
        try:
            
            def dummy_show(*args, **kwargs):
                pass
               
            plt.show = dummy_show
            exec(code, global_env)
            svg_string = global_env["svg_string"]
            return ChartExecutionResult(status= True, svg_string=svg_string)
        except Exception as exception_error:
            exception_info = traceback.format_exception_only(type(exception_error), exception_error)
            return ChartExecutionResult(status= False, error_msg= exception_info)

        finally:
            
            plt.show = original_show


if __name__ == "__main__":
    folder = "E:/visEval_dataset"
    query = "How many documents are stored? Bin the store date by weekday in a bar chart."
    test_manager = ChatManager(data_path=folder,
                               log_path="./test_logs.txt")
    dataset = Dataset(Path(folder))
    for instance in dataset.benchmark:
        nl_queries = instance["nl_queries"]
        for nl_query in nl_queries:
            if nl_query == query:
                db_id = instance['db_id']
                tables = instance['tables']
    msg = {
        'db_id': db_id,
        'query': query,
        'tables': tables,
        'send_to': SYSTEM_NAME,
        'library': 'matplotlib'
    }
    code = test_manager.start(msg)
    pprint(msg)
    print(code)
    print(test_manager.execute_to_svg(code))
