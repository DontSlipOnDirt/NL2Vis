# core/openai_vision_client.py
"""
Vision model client using OpenAI API via LangChain.
"""
import json
import os
from datetime import datetime
from typing import Any, Optional, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from core.config import (
    OPENAI_API_KEY,
    OPENAI_VISION_MODEL_NAME,
    OPENAI_VISION_TEMPERATURE,
    OPENAI_VISION_MAX_TOKENS
)

# Global state for logging
world_dict = {}
log_path = None
api_trace_json_path = None
total_prompt_tokens = 0
total_response_tokens = 0
call_idx = 0

def init_log_path(my_log_path: str):
    """Initialize logging paths."""
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
    
    api_trace_json_path = os.path.join(dir_name, 'openai_vision_trace.json')

def _serialize_content(content):
    """Helper to handle mixed text/image content in messages"""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        serialized = []
        for item in content:
            if isinstance(item, dict):
                if item.get('type') == 'image_url':
                    # Truncate base64 for logging readability
                    url = item['image_url']['url']
                    if url.startswith('data:image'):
                        serialized.append({
                            "type": "image_url",
                            "image_url": {"url": url[:50] + "...[TRUNCATED]..."}
                        })
                    else:
                        serialized.append(item)
                else:
                    serialized.append(item)
            else:
                serialized.append(str(item))
        return serialized
    return str(content)

class LoggedChatOpenAI(ChatOpenAI):
    def invoke(self, input: Any, config: Optional[dict] = None, **kwargs: Any) -> BaseMessage:
        global call_idx, total_prompt_tokens, total_response_tokens, api_trace_json_path
        
        response = super().invoke(input, config, **kwargs)
        
        if api_trace_json_path:
            call_idx += 1
            
            # Extract usage
            usage = response.response_metadata.get('token_usage', {})
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            total_tokens = usage.get('total_tokens', 0)
            
            total_prompt_tokens += prompt_tokens
            total_response_tokens += completion_tokens
            
            # Prepare log entry
            messages = input if isinstance(input, list) else [input]
            
            log_entry = {
                "idx": call_idx,
                "timestamp": datetime.now().isoformat(),
                "model": self.model_name,
                "messages": [
                    {
                        "role": m.type if hasattr(m, 'type') else "unknown",
                        "content": _serialize_content(m.content)
                    } for m in messages
                ],
                "response": response.content,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                },
                "cumulative_usage": {
                    "total_prompt_tokens": total_prompt_tokens,
                    "total_response_tokens": total_response_tokens
                }
            }
            
            with open(api_trace_json_path, 'a+', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                
        return response

def get_vision_model():
    """Get OpenAI vision model client."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
        
    return LoggedChatOpenAI(
        model=OPENAI_VISION_MODEL_NAME,
        temperature=OPENAI_VISION_TEMPERATURE,
        max_tokens=OPENAI_VISION_MAX_TOKENS,
        api_key=OPENAI_API_KEY
    )