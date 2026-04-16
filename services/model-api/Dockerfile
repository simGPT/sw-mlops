FROM python:3.11-slim

# 컨테이너 안에 app 폴더에서 작업
WORKDIR /app

# 의존성 먼저 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY app/ app/
COPY artifacts/ artifacts/

# 포트 열기
EXPOSE 8000

# 컨테이너 켜지면 자동으로 서버 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
