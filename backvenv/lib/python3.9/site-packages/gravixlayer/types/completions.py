"""
Completions types for GravixLayer SDK
"""
from typing import List, Optional, Union, Dict, Any
from dataclasses import dataclass


@dataclass
class CompletionChoice:
    """A single completion choice"""
    text: str
    index: int
    logprobs: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None


@dataclass
class CompletionUsage:
    """Usage statistics for completion"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class Completion:
    """Completion response"""
    id: str
    object: str
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Optional[CompletionUsage] = None