from typing import Dict, Any, List, Optional, Union, Iterator
import json
from ...types.chat import ChatCompletion, ChatCompletionChoice, ChatCompletionMessage, ChatCompletionUsage, ChatCompletionDelta, FunctionCall, ToolCall

class ChatCompletions:
    """Chat completions resource"""

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
    ) -> Union[ChatCompletion, Iterator[ChatCompletion]]:
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
        return self._create_stream(data) if stream else self._create_non_stream(data)

    def _create_non_stream(self, data: Dict[str, Any]) -> ChatCompletion:
        resp = self.client._make_request("POST", "chat/completions", data)
        return self._parse_response(resp.json())

    def _create_stream(self, data: Dict[str, Any]) -> Iterator[ChatCompletion]:
        resp = self.client._make_request("POST", "chat/completions", data, stream=True)
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8").strip()
            
            # Handle SSE format
            if line.startswith("data: "):
                line = line[6:]  # Remove "data: " prefix
            
            # Skip empty lines and [DONE] marker
            if not line or line == "[DONE]":
                continue
            
            try:
                chunk_data = json.loads(line)
                
                # Validate chunk_data is not None and is a dict
                if chunk_data is None or not isinstance(chunk_data, dict):
                    continue
                    
                parsed_chunk = self._parse_response(chunk_data, is_stream=True)
                
                # Only yield if we have valid choices and parsed_chunk is not None
                if parsed_chunk and parsed_chunk.choices:
                    yield parsed_chunk
                    
            except json.JSONDecodeError as e:
                # Log the problematic line for debugging
                print(f"Failed to parse JSON: {line[:100]}...")
                continue
            except Exception as e:
                print(f"Error processing chunk: {e}")
                continue

    def _parse_response(self, resp_data: Dict[str, Any], is_stream: bool = False) -> ChatCompletion:
        # Validate input
        if not resp_data or not isinstance(resp_data, dict):
            return None
            
        choices = []
        
        # Handle different response formats
        if "choices" in resp_data and resp_data["choices"]:
            for choice_data in resp_data["choices"]:
                # Validate choice_data
                if not choice_data or not isinstance(choice_data, dict):
                    continue
                    
                if is_stream:
                    # For streaming, create delta object
                    delta_content = None
                    delta_role = None
                    delta_tool_calls = None
                    
                    if "delta" in choice_data and choice_data["delta"]:
                        delta = choice_data["delta"]
                        if isinstance(delta, dict):
                            delta_content = delta.get("content")
                            delta_role = delta.get("role")
                            
                            # Parse tool calls in delta
                            if "tool_calls" in delta and delta["tool_calls"]:
                                delta_tool_calls = []
                                for tool_call_data in delta["tool_calls"]:
                                    if tool_call_data and isinstance(tool_call_data, dict):
                                        function_data = tool_call_data.get("function", {})
                                        if isinstance(function_data, dict):
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
                                            
                    elif "message" in choice_data and choice_data["message"]:
                        # Fallback: treat message as delta
                        message = choice_data["message"]
                        if isinstance(message, dict):
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
                    if not isinstance(message_data, dict):
                        message_data = {}
                    
                    # Parse tool calls if present
                    tool_calls = None
                    if "tool_calls" in message_data and message_data["tool_calls"]:
                        tool_calls = []
                        for tool_call_data in message_data["tool_calls"]:
                            if tool_call_data and isinstance(tool_call_data, dict):
                                function_data = tool_call_data.get("function", {})
                                if isinstance(function_data, dict):
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
        if "usage" in resp_data and resp_data["usage"] and isinstance(resp_data["usage"], dict):
            usage_data = resp_data["usage"]
            usage = ChatCompletionUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
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
