from typing_extensions import TypedDict

class ExtractText(TypedDict):
    weight: str
    muscleMass: str
    fatMass: str
    bodyFatPercentage: str
    bmi: str
    
class InBodyData(TypedDict):   
    id: int
    date: str
    weight: float
    muscleMass: float
    fatMass: float
    bodyFatPercentage: float
    bmi: float
    status: str
    
class AnalysisRequest(TypedDict):
    previous: InBodyData
    current: InBodyData