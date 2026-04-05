# core/vision_vllm_client.py
"""
Vision model client wrapper compatible with langchain BaseChatModel interface.
Implements the same interface as ChatOpenAI for seamless integration.
"""

import base64
from typing import List, Optional
from openai import OpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration

from core.config import (
    VISION_VLLM_BASE_URL,
    VISION_VLLM_MODEL_NAME,
    VISION_VLLM_MAX_TOKENS,
    VISION_VLLM_TEMPERATURE
)


class VisionVLLMClient(BaseChatModel):
    """
    Vision model client using vLLM OpenAI-compatible API.
    Compatible with langchain's BaseChatModel interface.
    """
    
    client: Optional[OpenAI] = None
    model_name: str = VISION_VLLM_MODEL_NAME
    temperature: float = VISION_VLLM_TEMPERATURE
    max_tokens: int = VISION_VLLM_MAX_TOKENS
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.client is None:
            self.client = OpenAI(
                base_url=VISION_VLLM_BASE_URL,
                api_key="dummy-key"  # vLLM doesn't require authentication
            )
    
    @property
    def _llm_type(self) -> str:
        """Return type of language model."""
        return "vision-vllm"
    
    def _convert_messages_to_openai_format(self, messages: List[BaseMessage]) -> List[dict]:
        """
        Convert langchain messages to OpenAI API format.
        Handles both text and image content.
        """
        openai_messages = []
        
        for message in messages:
            if isinstance(message, SystemMessage):
                openai_messages.append({
                    "role": "system",
                    "content": message.content
                })
            elif isinstance(message, HumanMessage):
                # Check if content is a list (multimodal) or string (text-only)
                if isinstance(message.content, list):
                    # Multimodal message with text and images
                    content_parts = []
                    for part in message.content:
                        if part["type"] == "text":
                            content_parts.append({
                                "type": "text",
                                "text": part["text"]
                            })
                        elif part["type"] == "image_url":
                            # Extract base64 image data
                            image_url = part["image_url"]["url"]
                            content_parts.append({
                                "type": "image_url",
                                "image_url": {"url": image_url}
                            })
                    
                    openai_messages.append({
                        "role": "user",
                        "content": content_parts
                    })
                else:
                    # Text-only message
                    openai_messages.append({
                        "role": "user",
                        "content": message.content
                    })
            elif isinstance(message, AIMessage):
                openai_messages.append({
                    "role": "assistant",
                    "content": message.content
                })
        
        return openai_messages
    
    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> ChatResult:
        """
        Generate a response from the vision model.
        Called by langchain's invoke() method.
        """
        openai_messages = self._convert_messages_to_openai_format(messages)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=stop
            )
            
            content = response.choices[0].message.content
            
            # Create ChatGeneration compatible with langchain
            message = AIMessage(content=content)
            generation = ChatGeneration(message=message)
            
            return ChatResult(generations=[generation])
            
        except Exception as e:
            raise RuntimeError(f"Vision vLLM inference failed: {e}")
    
    def invoke(self, messages: List[BaseMessage], **kwargs):
        """
        Main method called by evaluation code.
        Returns an AIMessage with the response content.
        """
        result = self._generate(messages, **kwargs)
        return result.generations[0].message


# Helper function for easy instantiation
def get_vision_model() -> VisionVLLMClient:
    """Get or create vision vLLM client."""
    return VisionVLLMClient()


if __name__ == "__main__":
    # Test the vision client
    import sys
    
    print("Testing Vision vLLM Client...")
    print(f"Model: {VISION_VLLM_MODEL_NAME}")
    print(f"Server: {VISION_VLLM_BASE_URL}")
    
    try:
        client = get_vision_model()
        
        # Test with text-only message
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hello! Please respond with 'Vision vLLM is working'")
        ]
        
        response = client.invoke(messages)
        print(f"\n✅ Success! Response: {response.content}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure vision vLLM server is running on port 8001!")
        print("Start it with: python -m core.vision_vllm_server")
        sys.exit(1)
