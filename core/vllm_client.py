# core/vllm_client.py
"""
vLLM client wrapper compatible with existing llm.py interface.
"""

import json
import time
from typing import Optional, Tuple
from openai import OpenAI
from core.config import (
    VLLM_BASE_URL,
    VLLM_MODEL_NAME,
    VLLM_MAX_TOKENS,
    VLLM_MAX_MODEL_LEN,
    VLLM_TEMPERATURE
)
import os

# Global state (mirrors llm.py)
world_dict = {}
log_path = None
api_trace_json_path = None
total_prompt_tokens = 0
total_response_tokens = 0
call_idx = 0

# Singleton client
_vllm_client: Optional[OpenAI] = None


def get_vllm_client() -> OpenAI:
    """Get or create vLLM OpenAI-compatible client."""
    global _vllm_client
    if _vllm_client is None:
        _vllm_client = OpenAI(
            base_url=VLLM_BASE_URL,
            api_key="dummy-key"  # vLLM doesn't require authentication
        )
    return _vllm_client


def init_log_path(my_log_path: str):
    """Initialize logging paths (compatible with llm.py)."""
    global total_prompt_tokens
    global total_response_tokens
    global log_path
    global api_trace_json_path
    global call_idx
    
    log_path = my_log_path
    total_prompt_tokens = 0
    total_response_tokens = 0
    call_idx = 0
    dir_name = os.path.dirname(log_path)
    os.makedirs(dir_name, exist_ok=True)
    
    api_trace_json_path = os.path.join(dir_name, 'api_trace.json')


def api_func(prompt: str) -> Tuple[str, int, int]:
    """
    Call vLLM via OpenAI-compatible API.
    Returns: (response_text, prompt_tokens, completion_tokens)
    """
    client = get_vllm_client()
    
    try:
        response = client.chat.completions.create(
            model=VLLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=VLLM_TEMPERATURE,
            max_tokens=VLLM_MAX_TOKENS
        )
        
        text = response.choices[0].message.content.strip()
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        
        # Check if response was truncated
        finish_reason = response.choices[0].finish_reason
        if finish_reason == "length":
            print(f"\n⚠️  WARNING: Response truncated due to max_tokens limit!")
            print(f"   Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}")
            print(f"   Consider increasing VLLM_MAX_TOKENS (currently {VLLM_MAX_TOKENS})")
            print(f"   Or check if prompt is too long.\n")
        
        return text, prompt_tokens, completion_tokens
    
    except Exception as e:
        # Check for context length errors
        error_msg = str(e)
        if "maximum context length" in error_msg.lower() or "too many tokens" in error_msg.lower():
            print(f"\n❌ ERROR: Input too long for model context window!")
            print(f"   VLLM_MAX_MODEL_LEN: {VLLM_MAX_MODEL_LEN}")
            print(f"   Error: {error_msg}\n")
        raise


def safe_call_llm(input_prompt: str, **kwargs) -> str:
    """
    Safely call vLLM with retry logic and logging.
    Drop-in replacement for llm.safe_call_llm().
    """
    global log_path
    global api_trace_json_path
    global total_prompt_tokens
    global total_response_tokens
    global world_dict
    global call_idx
    
    MAX_RETRIES = 5
    
    for attempt in range(MAX_RETRIES):
        try:
            if log_path is None:
                # Simple mode without logging
                sys_response, prompt_token, response_token = api_func(input_prompt)
                print(f"\nsys_response: \n{sys_response}")
                print(f'\nprompt_token, response_token: {prompt_token} {response_token}\n')
            else:
                # Full logging mode
                if (log_path is None) or (api_trace_json_path is None):
                    raise FileExistsError('log_path or api_trace_json_path is None, init_log_path first!')
                
                with open(log_path, 'a+', encoding='utf8') as log_fp, \
                     open(api_trace_json_path, 'a+', encoding='utf8') as trace_json_fp:
                    
                    print('\n' + f'*' * 20 + '\n', file=log_fp)
                    print(input_prompt, file=log_fp)
                    print('\n' + f'=' * 20 + '\n', file=log_fp)
                    
                    sys_response, prompt_token, response_token = api_func(input_prompt)
                    
                    print(sys_response, file=log_fp)
                    print(f'\nprompt_token, response_token: {prompt_token} {response_token}\n', file=log_fp)
                    print(f'\nprompt_token, response_token: {prompt_token} {response_token}\n')
                    
                    # Reset world_dict
                    if len(world_dict) > 0:
                        world_dict = {}
                    
                    # Add kwargs to world_dict
                    if len(kwargs) > 0:
                        world_dict = {}
                        for k, v in kwargs.items():
                            world_dict[k] = v
                    
                    # Increment and add call index
                    call_idx += 1
                    world_dict['idx'] = call_idx
                    
                    # Add response data
                    world_dict['response'] = '\n' + sys_response.strip() + '\n'
                    world_dict['input_prompt'] = input_prompt.strip() + '\n'
                    world_dict['prompt_token'] = prompt_token
                    world_dict['response_token'] = response_token
                    
                    # Update totals
                    total_prompt_tokens += prompt_token
                    total_response_tokens += response_token
                    
                    world_dict['cur_total_prompt_tokens'] = total_prompt_tokens
                    world_dict['cur_total_response_tokens'] = total_response_tokens
                    
                    # Write JSON trace
                    world_json_str = json.dumps(world_dict, ensure_ascii=False)
                    print(world_json_str, file=trace_json_fp)
                    
                    world_dict = {}
                    world_json_str = ''
                    
                    print(f'\ntotal_prompt_tokens, total_response_tokens: {total_prompt_tokens} {total_response_tokens}\n', file=log_fp)
                    print(f'\ntotal_prompt_tokens, total_response_tokens: {total_prompt_tokens} {total_response_tokens}\n')
            
            return sys_response
            
        except Exception as ex:
            print(f"vLLM request failed (attempt {attempt + 1}/{MAX_RETRIES}): {ex}")
            if attempt < MAX_RETRIES - 1:
                print(f"Retrying in 5 seconds...")
                time.sleep(5)
            else:
                raise ValueError(f'safe_call_llm error after {MAX_RETRIES} attempts: {ex}')


if __name__ == "__main__":
    # Test the vLLM client
    print("Testing vLLM client...")
    try:
        response = safe_call_llm("Hello! Please respond with 'vLLM is working'")
        print(f"\nSuccess! Response: {response}")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure vLLM server is running!")
