# -*- coding: utf-8 -*-
from core.agents import Provider, Generator, Corrector
from core.const import MAX_ROUND, SYSTEM_NAME, PROVIDER_NAME
from core.utils import show_svg
from core.api_config import USE_LOCAL_VLLM
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
    from core import llm
    LLM_API_FUC = llm.safe_call_llm
    INIT_LOG__PATH_FUNC = llm.init_log_path
    print(f"Use Azure OpenAI from core.llm in chat_manager.py")

import time
from pprint import pprint
from typing import Optional,List
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
    def __init__(self, csv_files: List[str], log_path: str):
        self.csv_files = csv_files
        self.log_path = log_path
        self.ping_network()
        self.chat_group = [
            Provider(),
            Generator(),
            Corrector(csv_files=self.csv_files)
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

    def start(self, nl_query: str):
        start_time = time.time()
        user_message = {
            'query': nl_query,
            'tables': self.csv_files,
            'send_to': PROVIDER_NAME
        }
        for _ in range(MAX_ROUND):
            self._chat_single_round(user_message)
            if user_message['send_to'] == SYSTEM_NAME:
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
    csv_files = [r"E:\visEval_dataset\databases\activity_1\Faculty.csv",r"E:\visEval_dataset\databases\activity_1\Activity.csv"]
    query = "A pie chart showing the number of faculty members for each rank."
    test_manager = ChatManager(csv_files=csv_files, log_path="./test_logs.txt")

    code = test_manager.start(query)
    print(code)
    print(test_manager.execute_to_svg(code))
