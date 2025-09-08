from typing import Optional, List, Dict, Any
from dataclasses import dataclass

@dataclass
class FunctionCall:
    """Function call object"""
    name: str
    arguments: str

@dataclass
class ToolCall:
    """Tool call object for function calling"""
    id: str
    type: str = "function"
    function: Optional[FunctionCall] = None

@dataclass
class ChatCompletionMessage:
    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None

@dataclass
class ChatCompletionDelta:
    """Delta object for streaming responses"""
    role: Optional[str] = None
    content: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[ToolCall]] = None

@dataclass
class ChatCompletionChoice:
    index: int
    message: ChatCompletionMessage
    finish_reason: Optional[str] = None
    delta: Optional[ChatCompletionDelta] = None

@dataclass
class ChatCompletionUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

@dataclass
class ChatCompletion:
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[ChatCompletionUsage] = None
