from typing import List, Dict, Any
from ..types.accelerators import Accelerator, AcceleratorList

class Accelerators:
    def __init__(self, client):
        self.client = client

    def list(self) -> List[Accelerator]:
        """List all available accelerators/GPUs"""
        # Use the accelerators endpoint
        original_base_url = self.client.base_url
        self.client.base_url = self.client.base_url.replace("/v1/inference", "/v1")
        
        try:
            response = self.client._make_request("GET", "accelerators")
            accelerators_data = response.json()
            
            # Handle different response formats
            if isinstance(accelerators_data, list):
                return [Accelerator(**accelerator) for accelerator in accelerators_data]
            elif isinstance(accelerators_data, dict) and 'accelerators' in accelerators_data:
                return [Accelerator(**accelerator) for accelerator in accelerators_data['accelerators']]
            elif isinstance(accelerators_data, dict) and not accelerators_data:
                # Empty dict response means no accelerators
                return []
            else:
                # If it's a different format, return empty list and log the issue
                print(f"Unexpected response format: {type(accelerators_data)}, content: {accelerators_data}")
                return []
        finally:
            self.client.base_url = original_base_url