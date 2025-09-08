import os
import requests
import logging
from typing import Optional, Dict, Any, Type
from .resources.chat.completions import ChatCompletions
from .resources.embeddings import Embeddings
from .resources.completions import Completions
from .resources.deployments import Deployments
from .resources.accelerators import Accelerators
from .types.exceptions import (
    GravixLayerError,
    GravixLayerAuthenticationError,
    GravixLayerRateLimitError,
    GravixLayerServerError,
    GravixLayerBadRequestError,
    GravixLayerConnectionError
)

class GravixLayer:
    """
    Main GravixLayer client - OpenAI compatible.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        headers: Optional[Dict[str, str]] = None,
        logger: Optional[Type[logging.Logger]] = None,
        user_agent: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("GRAVIXLAYER_API_KEY")
        self.base_url = base_url or os.environ.get("GRAVIXLAYER_BASE_URL", "https://api.gravixlayer.com/v1/inference")
        
        # Validate URL scheme - support both HTTP and HTTPS
        if not (self.base_url.startswith("http://") or self.base_url.startswith("https://")):
            raise ValueError("Base URL must use HTTP or HTTPS protocol")

        # Allow both http and https; require explicit scheme for clarity
        if not (self.base_url.startswith("http://") or self.base_url.startswith("https://")):
            raise ValueError("Base URL must start with http:// or https://")
        self.timeout = timeout
        self.max_retries = max_retries
        self.custom_headers = headers or {}
        self.logger = logger or logging.getLogger("gravixlayer")
        self.logger.setLevel(logging.INFO)
        if not self.logger.hasHandlers():
            logging.basicConfig(level=logging.INFO)
        self.user_agent = user_agent or f"gravixlayer-python/0.0.22"
        if not self.api_key:
            raise ValueError("API key must be provided via argument or GRAVIXLAYER_API_KEY environment variable")
        self.chat = ChatResource(self)
        self.embeddings = Embeddings(self)
        self.completions = Completions(self)
        self.deployments = Deployments(self)
        self.accelerators = Accelerators(self)

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        **kwargs
    ) -> requests.Response:
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}" if endpoint else self.base_url
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": self.user_agent,
            **self.custom_headers,
        }
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=data,
                    timeout=self.timeout,
                    stream=stream,
                    **kwargs
                )
                if resp.status_code == 200:
                    return resp
                elif resp.status_code == 401:
                    raise GravixLayerAuthenticationError("Authentication failed.")
                elif resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    self.logger.warning(f"Rate limit exceeded. Retrying in {retry_after or 2**attempt}s...")
                    if attempt < self.max_retries:
                        import time
                        time.sleep(float(retry_after) if retry_after else (2 ** attempt))
                        continue
                    raise GravixLayerRateLimitError(resp.text)
                elif resp.status_code in [502, 503, 504] and attempt < self.max_retries:
                    self.logger.warning(f"Server error: {resp.status_code}. Retrying...")
                    import time
                    time.sleep(2 ** attempt)
                    continue
                elif 400 <= resp.status_code < 500:
                    raise GravixLayerBadRequestError(resp.text)
                elif 500 <= resp.status_code < 600:
                    raise GravixLayerServerError(resp.text)
                else:
                    resp.raise_for_status()
            except requests.RequestException as e:
                if attempt == self.max_retries:
                    raise GravixLayerConnectionError(str(e)) from e
                self.logger.warning("Transient connection error, retrying...")
                import time
                time.sleep(2 ** attempt)
        raise GravixLayerError("Failed to complete request.")

class ChatResource:
    def __init__(self, client: GravixLayer):
        self.client = client
        self.completions = ChatCompletions(client)

OpenAI = GravixLayer
