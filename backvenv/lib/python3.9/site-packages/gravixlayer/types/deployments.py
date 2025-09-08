from typing import List, Optional, Literal
from pydantic import BaseModel
from datetime import datetime

class DeploymentCreate(BaseModel):
    deployment_name: str
    hw_type: Literal["dedicated"] = "dedicated"
    hardware: str
    min_replicas: int = 1
    model_name: str

class Deployment(BaseModel):
    deployment_id: str
    user_email: str
    model_name: str
    deployment_name: str
    status: str
    created_at: str
    hardware: str
    min_replicas: int
    hw_type: str

class DeploymentList(BaseModel):
    deployments: List[Deployment]

class DeploymentResponse(BaseModel):
    deployment_id: str
    message: str
    status: str