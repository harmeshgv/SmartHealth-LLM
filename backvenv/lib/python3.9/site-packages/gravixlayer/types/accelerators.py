from typing import List, Optional
from pydantic import BaseModel, Field, computed_field

class Accelerator(BaseModel):
    """Represents a GPU/accelerator specification"""
    accelerator_id: str
    pricing: float
    hw_model: str
    hw_link: str
    hw_memory: int
    provider: str
    status: str
    updated_at: str
    
    @computed_field
    @property
    def name(self) -> str:
        """Generate a friendly name from the accelerator_id"""
        return self.accelerator_id.replace("_", " ")
    
    @computed_field
    @property
    def hardware_string(self) -> str:
        """Generate hardware string in the expected format"""
        # Convert to lowercase and format like: nvidia-t4-16gb-pcie_1
        provider_lower = self.provider.lower()
        model_lower = self.hw_model.lower()
        memory_str = f"{self.hw_memory}gb"
        link_lower = self.hw_link.lower()
        return f"{provider_lower}-{model_lower}-{memory_str}-{link_lower}_1"
    
    @computed_field
    @property
    def memory(self) -> str:
        """Format memory as string"""
        return f"{self.hw_memory}GB"
    
    @computed_field
    @property
    def gpu_type(self) -> str:
        """Get GPU type (model)"""
        return self.hw_model.lower()
    
    @computed_field
    @property
    def use_case(self) -> str:
        """Determine use case based on memory and model"""
        if self.hw_memory <= 16:
            return "Small models, development"
        elif self.hw_memory <= 32:
            return "Medium models"
        elif self.hw_memory <= 24 and "rtx" in self.hw_model.lower():
            return "Development, small production"
        else:
            return "Large models, production"

class AcceleratorList(BaseModel):
    """Response model for accelerator list"""
    accelerators: List[Accelerator]