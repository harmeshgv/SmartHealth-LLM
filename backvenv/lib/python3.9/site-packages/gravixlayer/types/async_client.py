import os
import httpx
import logging
import asyncio
import json
from typing import Optional, Dict, Any, List, Union, AsyncIterator
from ..types.chat import ChatCompletion, ChatCompletionChoice, ChatCompletionMessage, ChatCompletionUsage, ChatCompletionDelta, FunctionCall, ToolCall
from ..types.exceptions import (
    GravixLayerError,
    GravixLayerAuthenticationError,
    GravixLayerRateLimitError,
    GravixLayerServerError,
    GravixLayerBadRequestError,
    GravixLayerConnectionError
)
from ..resources.async_embeddings import AsyncEmbeddings
from ..resources.async_completions import AsyncCompletions

class AsyncChatResource:
    def __init__(self, client):
        self.client = client
        self.completions = AsyncChatCompletions(client)

class AsyncChatCompletions:
    def __init__(self, client):
        self.client = client
    
    def create(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs
    ) -> Union[ChatCompletion, AsyncIterator[ChatCompletion]]:
        # Convert message objects to dictionaries if needed
        serialized_messages = []
        for msg in messages:
            if hasattr(msg, '__dict__'):
                # Convert dataclass to dict
                msg_dict = {
                    "role": msg.role,
                    "content": msg.content
                }
                if hasattr(msg, 'name') and msg.name:
                    msg_dict["name"] = msg.name
                if hasattr(msg, 'tool_call_id') and msg.tool_call_id:
                    msg_dict["tool_call_id"] = msg.tool_call_id
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    msg_dict["tool_calls"] = []
                    for tool_call in msg.tool_calls:
                        tool_call_dict = {
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        }
                        msg_dict["tool_calls"].append(tool_call_dict)
                serialized_messages.append(msg_dict)
            else:
                # Already a dictionary
                serialized_messages.append(msg)
        
        data = {
            "model": model,
            "messages": serialized_messages,
            "stream": stream
        }
        if temperature is not None:
            data["temperature"] = temperature
        if max_tokens is not None:
            data["max_tokens"] = max_tokens
        if top_p is not None:
            data["top_p"] = top_p
        if frequency_penalty is not None:
            data["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            data["presence_penalty"] = presence_penalty
        if stop is not None:
            data["stop"] = stop
        if tools is not None:
            data["tools"] = tools
        if tool_choice is not None:
            data["tool_choice"] = tool_choice
        data.update(kwargs)
        
        # Fix: Return the async generator directly, don't await it here
        if stream:
            return self._create_stream(data)
        else:
            # For non-streaming, return the coroutine to be awaited
            return self._create_non_stream(data)

    async def _create_non_stream(self, data: Dict[str, Any]) -> ChatCompletion:
        resp = await self.client._make_request("POST", "chat/completions", data)
        return self._parse_response(resp.json())

    async def _create_stream(self, data: Dict[str, Any]) -> AsyncIterator[ChatCompletion]:
        """Async generator for streaming responses"""
        resp = await self.client._make_request("POST", "chat/completions", data, stream=True)
        
        async for line in resp.aiter_lines():
            if not line:
                continue
            line = line.strip()
            
            # Handle SSE format
            if line.startswith("data: "):
                line = line[6:]  # Remove "data: " prefix
            
            # Skip empty lines and [DONE] marker
            if not line or line == "[DONE]":
                continue
            
            try:
                chunk_data = json.loads(line)
                parsed_chunk = self._parse_response(chunk_data, is_stream=True)
                
                # Only yield if we have valid choices
                if parsed_chunk.choices:
                    yield parsed_chunk
                    
            except json.JSONDecodeError:
                # Skip malformed JSON
                continue
            except Exception:
                # Skip other errors
                continue

    def _parse_response(self, resp_data: Dict[str, Any], is_stream: bool = False) -> ChatCompletion:
        choices = []
        
        # Handle different response formats
        if "choices" in resp_data and resp_data["choices"]:
            for choice_data in resp_data["choices"]:
                if is_stream:
                    # For streaming, create delta object
                    delta_content = None
                    delta_role = None
                    delta_tool_calls = None
                    
                    if "delta" in choice_data:
                        delta = choice_data["delta"]
                        delta_content = delta.get("content")
                        delta_role = delta.get("role")
                        
                        # Parse tool calls in delta
                        if "tool_calls" in delta and delta["tool_calls"]:
                            delta_tool_calls = []
                            for tool_call_data in delta["tool_calls"]:
                                function_data = tool_call_data.get("function", {})
                                function_call = FunctionCall(
                                    name=function_data.get("name", ""),
                                    arguments=function_data.get("arguments", "{}")
                                )
                                tool_call = ToolCall(
                                    id=tool_call_data.get("id", ""),
                                    type=tool_call_data.get("type", "function"),
                                    function=function_call
                                )
                                delta_tool_calls.append(tool_call)
                                
                    elif "message" in choice_data:
                        # Fallback: treat message as delta
                        message = choice_data["message"]
                        delta_content = message.get("content")
                        delta_role = message.get("role")
                    
                    # Create delta object
                    delta_obj = ChatCompletionDelta(
                        role=delta_role,
                        content=delta_content,
                        tool_calls=delta_tool_calls
                    )
                    
                    # Create message object (for compatibility)
                    msg = ChatCompletionMessage(
                        role=delta_role or "assistant",
                        content=delta_content or "",
                        tool_calls=delta_tool_calls
                    )
                    
                    choices.append(ChatCompletionChoice(
                        index=choice_data.get("index", 0),
                        message=msg,
                        delta=delta_obj,
                        finish_reason=choice_data.get("finish_reason")
                    ))
                else:
                    # For non-streaming, use message object
                    message_data = choice_data.get("message", {})
                    
                    # Parse tool calls if present
                    tool_calls = None
                    if "tool_calls" in message_data and message_data["tool_calls"]:
                        tool_calls = []
                        for tool_call_data in message_data["tool_calls"]:
                            function_data = tool_call_data.get("function", {})
                            function_call = FunctionCall(
                                name=function_data.get("name", ""),
                                arguments=function_data.get("arguments", "{}")
                            )
                            tool_call = ToolCall(
                                id=tool_call_data.get("id", ""),
                                type=tool_call_data.get("type", "function"),
                                function=function_call
                            )
                            tool_calls.append(tool_call)
                    
                    msg = ChatCompletionMessage(
                        role=message_data.get("role", "assistant"),
                        content=message_data.get("content"),
                        tool_calls=tool_calls,
                        tool_call_id=message_data.get("tool_call_id")
                    )
                    choices.append(ChatCompletionChoice(
                        index=choice_data.get("index", 0),
                        message=msg,
                        finish_reason=choice_data.get("finish_reason")
                    ))
        
        # Fallback: create a single choice if no choices found
        if not choices:
            content = ""
            if isinstance(resp_data, str):
                content = resp_data
            elif "content" in resp_data:
                content = resp_data["content"]
            
            if is_stream:
                delta_obj = ChatCompletionDelta(content=content)
                msg = ChatCompletionMessage(role="assistant", content=content)
                choices = [ChatCompletionChoice(
                    index=0, 
                    message=msg, 
                    delta=delta_obj,
                    finish_reason=None
                )]
            else:
                msg = ChatCompletionMessage(role="assistant", content=content)
                choices = [ChatCompletionChoice(
                    index=0, 
                    message=msg, 
                    finish_reason="stop"
                )]

        # Parse usage if available
        usage = None
        if "usage" in resp_data:
            usage = ChatCompletionUsage(
                prompt_tokens=resp_data["usage"].get("prompt_tokens", 0),
                completion_tokens=resp_data["usage"].get("completion_tokens", 0),
                total_tokens=resp_data["usage"].get("total_tokens", 0),
            )
        
        import time
        return ChatCompletion(
            id=resp_data.get("id", f"chatcmpl-{hash(str(resp_data))}"),
            object="chat.completion" if not is_stream else "chat.completion.chunk",
            created=resp_data.get("created", int(time.time())),
            model=resp_data.get("model", "unknown"),
            choices=choices,
            usage=usage,
        )

class AsyncGravixLayer:
    """
    Async client for GravixLayer
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        headers: Optional[Dict[str, str]] = None,
        logger: Optional[logging.Logger] = None,
        user_agent: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("GRAVIXLAYER_API_KEY")
        self.base_url = base_url or os.environ.get("GRAVIXLAYER_BASE_URL", "https://api.gravixlayer.com/v1/inference")
        
        # Validate URL scheme - support both HTTP and HTTPS
        if not (self.base_url.startswith("http://") or self.base_url.startswith("https://")):
            raise ValueError("Base URL must start with http:// or https://")
        self.timeout = timeout
        self.max_retries = max_retries
        self.custom_headers = headers or {}
        self.logger = logger or logging.getLogger("gravixlayer-async")
        self.user_agent = user_agent or f"gravixlayer-python/0.0.22"
        if not self.api_key:
            raise ValueError("API key must be provided via argument or GRAVIXLAYER_API_KEY environment variable")
        
        # Create the proper chat resource structure
        self.chat = AsyncChatResource(self)
        self.embeddings = AsyncEmbeddings(self)
        self.completions = AsyncCompletions(self)

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        **kwargs
    ) -> httpx.Response:
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}" if endpoint else self.base_url
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": self.user_agent,
            **self.custom_headers,
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(self.max_retries + 1):
                try:
                    resp = await client.request(
                        method=method,
                        url=url,
                        headers=headers,
                        json=data,
                        **kwargs,
                    )
                    
                    if resp.status_code == 200:
                        return resp
                    elif resp.status_code == 401:
                        raise GravixLayerAuthenticationError("Authentication failed.")
                    elif resp.status_code == 429:
                        if attempt < self.max_retries:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        raise GravixLayerRateLimitError(resp.text)
                    elif resp.status_code in [502, 503, 504] and attempt < self.max_retries:
                        self.logger.warning(f"Server error: {resp.status_code}. Retrying...")
                        await asyncio.sleep(2 ** attempt)
                        continue
                    elif 400 <= resp.status_code < 500:
                        raise GravixLayerBadRequestError(resp.text)
                    elif 500 <= resp.status_code < 600:
                        raise GravixLayerServerError(resp.text)
                    else:
                        resp.raise_for_status()
                        
                except httpx.RequestError as e:
                    if attempt == self.max_retries:
                        raise GravixLayerConnectionError(str(e)) from e
                    await asyncio.sleep(2 ** attempt)
        
        raise GravixLayerError("Failed async request")