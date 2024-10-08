from pydantic import BaseModel
from typing import List

class InputData(BaseModel):
    features: List[float]

class Prediction(BaseModel):
    label: int
    confidence: float
