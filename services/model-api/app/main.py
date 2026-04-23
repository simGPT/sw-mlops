from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.schemas.schema import RequestSchema, ResponseSchema
from app.services.service_factory import get_service


app = FastAPI(title="Model API")
# Fastapi의 모든 요청과 응답을 자동으로 추적하고 메트릭 수집 + /metrics 앤드포인트를 자동으로 추가 해줌
# prometheus가 /metrics 앤드포인트로 접근해서 매트릭을 가져갈 수 있음
Instrumentator().instrument(app).expose(app) 

# API 헬스 체크용
@app.get("/health")
def health():
    return {"status": "ok"}

# MNIST 데이터 모델 예측 API
@app.post("/mnist_predict", response_model=ResponseSchema)
def predict(request: RequestSchema):
    try:
        service = get_service(request.type) # 모델에 맞는 service 지정
        output = service.predict(request.data)  # 예측 결과

        return ResponseSchema(
            success=True,
            model=request.type,
            result=output["result"],
            metadata=output["metadata"],
            error=None,
        )
    except Exception as e:
        return ResponseSchema(
            success=False,
            model=request.type,
            result=None,
            metadata=None,
            error=str(e),
        )
