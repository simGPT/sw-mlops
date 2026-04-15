from app.services import mnist_service


_services = {
    "mnist": mnist_service,
}

# 모델의 종류에 맞게 service 파일 불러오는 라우터
def get_service(model_type: str):
    service = _services.get(model_type)
    if service is None:
        raise ValueError(f"지원하지 않는 모델 타입: {model_type} (가능한 타입: {list(_services.keys())})")
    return service
