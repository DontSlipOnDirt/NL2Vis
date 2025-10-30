"""
vLLM Client Module - Local LLM Interface
Provides a drop-in replacement for the Azure OpenAI API using vLLM server
"""
import sys
import json
import time
import os
from typing import Tuple
from openai import OpenAI

# Global state for logging and token tracking
world_dict = {}
log_path = None
api_trace_json_path = None
total_prompt_tokens = 0
total_response_tokens = 0

# vLLM Configuration
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL_NAME = os.environ.get("VLLM_MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

# Global vLLM client instance
_vllm_client = None


def get_vllm_client():
    """Get or create the global vLLM client instance"""
    global _vllm_client
    if _vllm_client is None:
        _vllm_client = OpenAI(
            base_url=VLLM_BASE_URL,
            api_key="dummy"  # vLLM doesn't require a real API key
        )
    return _vllm_client


def init_log_path(my_log_path):
    """Initialize logging paths and reset token counters"""
    global total_prompt_tokens
    global total_response_tokens
    global log_path
    global api_trace_json_path
    
    log_path = my_log_path
    total_prompt_tokens = 0
    total_response_tokens = 0
    dir_name = os.path.dirname(log_path)
    os.makedirs(dir_name, exist_ok=True)
    
    api_trace_json_path = os.path.join(dir_name, 'api_trace.json')


def api_func(prompt: str) -> Tuple[str, int, int]:
    """
    Call vLLM server with the given prompt
    
    Args:
        prompt: The input prompt to send to the model
        
    Returns:
        Tuple of (response_text, prompt_tokens, completion_tokens)
    """
    global VLLM_MODEL_NAME
    
    client = get_vllm_client()
    
    response = client.chat.completions.create(
        model=VLLM_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=4096,  # Adjust based on your model's context length
        # Optional: Add stop sequences if needed
        # stop=["```\n\n", "======"]
    )
    
    text = response.choices[0].message.content.strip()
    prompt_token = response.usage.prompt_tokens
    response_token = response.usage.completion_tokens
    
    return text, prompt_token, response_token


def safe_call_llm(input_prompt, **kwargs) -> str:
    """
    Safely call the vLLM with retry logic and logging
    
    Args:
        input_prompt: The prompt to send to the model
        **kwargs: Additional metadata for logging
        
    Returns:
        The model's response as a string
    """
    global VLLM_MODEL_NAME
    global log_path
    global api_trace_json_path
    global total_prompt_tokens
    global total_response_tokens
    global world_dict
    
    max_retries = 5
    
    for i in range(max_retries):
        try:
            if log_path is None:
                # No logging mode
                sys_response, prompt_token, response_token = api_func(input_prompt)
                print(f"\nsys_response: \n{sys_response}")
                print(f'\n prompt_token, response_token: {prompt_token} {response_token}\n')
            else:
                # Logging mode
                if (log_path is None) or (api_trace_json_path is None):
                    raise FileExistsError('log_path or api_trace_json_path is None, init_log_path first!')
                    
                with open(log_path, 'a+', encoding='utf8') as log_fp, \
                     open(api_trace_json_path, 'a+', encoding='utf8') as trace_json_fp:
                    
                    print('\n' + f'*' * 20 + '\n', file=log_fp)
                    print(input_prompt, file=log_fp)
                    print('\n' + f'=' * 20 + '\n', file=log_fp)
                    
                    sys_response, prompt_token, response_token = api_func(input_prompt)
                    
                    print(sys_response, file=log_fp)
                    print(f'\n prompt_token, response_token: {prompt_token} {response_token}\n', file=log_fp)
                    print(f'\n prompt_token, response_token: {prompt_token} {response_token}\n')
                    
                    # Reset world_dict if not empty
                    if len(world_dict) > 0:
                        world_dict = {}
                    
                    # Add kwargs to world_dict
                    if len(kwargs) > 0:
                        world_dict = {}
                        for k, v in kwargs.items():
                            world_dict[k] = v
                    
                    # Add response and prompt to world_dict
                    world_dict['response'] = '\n' + sys_response.strip() + '\n'
                    world_dict['input_prompt'] = input_prompt.strip() + '\n'
                    world_dict['prompt_token'] = prompt_token
                    world_dict['response_token'] = response_token
                    
                    total_prompt_tokens += prompt_token
                    total_response_tokens += response_token
                    
                    world_dict['cur_total_prompt_tokens'] = total_prompt_tokens
                    world_dict['cur_total_response_tokens'] = total_response_tokens
                    
                    # Write to JSON trace file
                    world_json_str = json.dumps(world_dict, ensure_ascii=False)
                    print(world_json_str, file=trace_json_fp)
                    
                    world_dict = {}
                    world_json_str = ''
                    
                    print(f'\n total_prompt_tokens, total_response_tokens: {total_prompt_tokens} {total_response_tokens}\n', file=log_fp)
                    print(f'\n total_prompt_tokens, total_response_tokens: {total_prompt_tokens} {total_response_tokens}\n')
            
            return sys_response
            
        except Exception as ex:
            print(f"vLLM request failed: {ex}")
            print(f'Request to {VLLM_MODEL_NAME} failed. Attempt {i+1}/{max_retries}. Retrying in 5 seconds...')
            time.sleep(5)
    
    raise ValueError(f'safe_call_llm error! Failed after {max_retries} attempts. Check if vLLM server is running at {VLLM_BASE_URL}')


if __name__ == "__main__":
    # Test the vLLM client
    print("Testing vLLM client...")
    print(f"Connecting to: {VLLM_BASE_URL}")
    print(f"Model: {VLLM_MODEL_NAME}")
    
    try:
        res = safe_call_llm('Hello! Please respond with a short greeting.')
        print(f"\nSuccess! Response: {res}")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure vLLM server is running. Start it with:")
        print(f"python -m vllm.entrypoints.openai.api_server --model {VLLM_MODEL_NAME} --host 0.0.0.0 --port 8000")
