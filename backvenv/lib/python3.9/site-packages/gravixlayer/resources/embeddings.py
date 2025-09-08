from typing import Union, List
from ..types.embeddings import EmbeddingResponse, EmbeddingObject, EmbeddingUsage

class Embeddings:
    """Embeddings resource"""

    def __init__(self, client):
        self.client = client

    def create(
        self,
        model: str,
        input: Union[str, List[str]],
        encoding_format: str = "float",
        dimensions: int = None,
        user: str = None,
        **kwargs
    ) -> EmbeddingResponse:
        """
        Create embeddings for the given input.
        
        Args:
            model: ID of the model to use
            input: Input text to embed, encoded as a string or array of strings
            encoding_format: The format to return the embeddings in
            dimensions: The number of dimensions the resulting output embeddings should have
            user: A unique identifier representing your end-user
            **kwargs: Additional parameters
            
        Returns:
            EmbeddingResponse: The embedding response
        """
        # Prepare the request data
        data = {
            "model": model,
            "input": input,
        }
        
        if encoding_format:
            data["encoding_format"] = encoding_format
        if dimensions:
            data["dimensions"] = dimensions
        if user:
            data["user"] = user
        
        # Add any additional kwargs
        data.update(kwargs)
        
        # Make the API request
        resp = self.client._make_request("POST", "embeddings", data)
        response_data = resp.json()
        
        # Parse the response
        return self._parse_response(response_data)

    def _parse_response(self, resp_data: dict) -> EmbeddingResponse:
        """Parse the API response into an EmbeddingResponse object."""
        
        # Parse embeddings data
        embeddings = []
        if "data" in resp_data:
            for i, item in enumerate(resp_data["data"]):
                embedding = EmbeddingObject(
                    object=item.get("object", "embedding"),
                    embedding=item.get("embedding", []),
                    index=item.get("index", i)
                )
                embeddings.append(embedding)
        
        # Parse usage information
        usage = None
        if "usage" in resp_data:
            usage = EmbeddingUsage(
                prompt_tokens=resp_data["usage"].get("prompt_tokens", 0),
                total_tokens=resp_data["usage"].get("total_tokens", 0)
            )
        
        return EmbeddingResponse(
            object=resp_data.get("object", "list"),
            data=embeddings,
            model=resp_data.get("model", ""),
            usage=usage
        )