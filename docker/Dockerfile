# 베이스 이미지로 PyTorch와 필요한 라이브러리가 포함된 이미지 사용
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 패키지 설치
RUN apt-get update && \
    apt-get install -y python3-pip && \
    pip install --upgrade pip

# 프로젝트 파일 복사
COPY requirements.txt ./

# 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt
# cuda 지원 버전 torch 설치
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 애플리케이션 코드 복사
COPY ./src /app/src

# FastAPI 앱 실행
CMD ["uvicorn", "src.run:app", "--host", "0.0.0.0", "--port", "8000"]