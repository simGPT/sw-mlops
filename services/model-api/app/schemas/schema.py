from pydantic import BaseModel
from typing import Any

class RequestSchema(BaseModel):
    type: str   # 모델 타입
    data: dict  # 요청 데이터

class ResponseSchema(BaseModel):
    success: bool   # 성공 여부
    model: str  # 모델 타입(추후 모델 추가하거나 변경할때)
    result: Any  # 응답 결과
    metadata: Any
    error: Any  
