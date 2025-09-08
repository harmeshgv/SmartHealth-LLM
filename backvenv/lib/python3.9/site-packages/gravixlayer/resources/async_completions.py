"""
Async Completions resource for GravixLayer SDK
"""
from typing import Dict, Any, List, Optional, Union, AsyncIterator
import json
from ..types.completions import Completion, CompletionChoice, CompletionUsage


class AsyncCompletions:
    """Async Completions resource for prompt-based completions"""

    def __init__(self, client):
        self.client = client

    def create(
        self,
        model: str,
        prompt: Union[str, List[str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        n: Optional[int] = None,
        stream: bool = False,
        logprobs: Optional[int] = None,
        echo: bool = False,
        stop: Optional[Union[str, List[str]]] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        best_of: Optional[int] = None,
        logit_bias: Optional[Dict[str, float]] = None,
        user: Optional[str] = None,
        **kwargs
    ) -> Union[Completion, AsyncIterator[Completion]]:
        """
        Create a completion for the provided prompt and parameters asynchronously.
        
        Args:
            model: ID of the model to use
            prompt: The prompt(s) to generate completions for
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0 to 2)
            top_p: Nucleus sampling parameter
            n: Number of completions to generate
            stream: Whether to stream back partial progress
            logprobs: Include the log probabilities on the logprobs most likely tokens
            echo: Echo back the prompt in addition to the completion
            stop: Up to 4 sequences where the API will stop generating further tokens
            presence_penalty: Penalty for new tokens based on whether they appear in the text so far
            frequency_penalty: Penalty for new tokens based on their existing frequency in the text so far
            best_of: Generates best_of completions server-side and returns the "best"
            logit_bias: Modify the likelihood of specified tokens appearing in the completion
            user: A unique identifier representing your end-user
            **kwargs: Additional parameters
            
        Returns:
            Completion or AsyncIterator[Completion]: The completion response
        """
        data = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }
        
        if max_tokens is not None:
            data["max_tokens"] = max_tokens
        if temperature is not None:
            data["temperature"] = temperature
        if top_p is not None:
            data["top_p"] = top_p
        if n is not None:
            data["n"] = n
        if logprobs is not None:
            data["logprobs"] = logprobs
        if echo is not None:
            data["echo"] = echo
        if stop is not None:
            data["stop"] = stop
        if presence_penalty is not None:
            data["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            data["frequency_penalty"] = frequency_penalty
        if best_of is not None:
            data["best_of"] = best_of
        if logit_bias is not None:
            data["logit_bias"] = logit_bias
        if user is not None:
            data["user"] = user
        
        data.update(kwargs)
        
        return self._create_stream(data) if stream else self._create_non_stream(data)

    async def _create_non_stream(self, data: Dict[str, Any]) -> Completion:
        """Create non-streaming completion"""
        resp = await self.client._make_request("POST", "completions", data)
        return self._parse_response(resp.json())

    async def _create_stream(self, data: Dict[str, Any]) -> AsyncIterator[Completion]:
        """Create streaming completion"""
        resp = await self.client._make_request("POST", "completions", data, stream=True)
        
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

    def _parse_response(self, resp_data: Dict[str, Any], is_stream: bool = False) -> Completion:
        """Parse API response into Completion object"""
        choices = []
        
        # Handle different response formats
        if "choices" in resp_data and resp_data["choices"]:
            for choice_data in resp_data["choices"]:
                text = ""
                
                if is_stream:
                    # For streaming, get text from delta or text field
                    if "delta" in choice_data:
                        text = choice_data["delta"].get("content", "") or choice_data["delta"].get("text", "")
                    elif "text" in choice_data:
                        text = choice_data["text"]
                else:
                    # For non-streaming, get text directly
                    text = choice_data.get("text", "")
                
                choice = CompletionChoice(
                    text=text,
                    index=choice_data.get("index", 0),
                    logprobs=choice_data.get("logprobs"),
                    finish_reason=choice_data.get("finish_reason")
                )
                choices.append(choice)
        
        # Fallback: create a single choice if no choices found
        if not choices:
            text = ""
            if isinstance(resp_data, str):
                text = resp_data
            elif "text" in resp_data:
                text = resp_data["text"]
            elif "content" in resp_data:
                text = resp_data["content"]
            
            choices = [CompletionChoice(
                text=text,
                index=0,
                finish_reason="stop" if not is_stream else None
            )]

        # Parse usage if available
        usage = None
        if "usage" in resp_data:
            usage = CompletionUsage(
                prompt_tokens=resp_data["usage"].get("prompt_tokens", 0),
                completion_tokens=resp_data["usage"].get("completion_tokens", 0),
                total_tokens=resp_data["usage"].get("total_tokens", 0),
            )
        
        import time
        return Completion(
            id=resp_data.get("id", f"cmpl-{hash(str(resp_data))}"),
            object="text_completion" if not is_stream else "text_completion.chunk",
            created=resp_data.get("created", int(time.time())),
            model=resp_data.get("model", "unknown"),
            choices=choices,
            usage=usage,
        )