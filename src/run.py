import sys
import argparse
import uvicorn
from fastapi import FastAPI

from src.utils.logger import logger
from src.model.train import main as train_main
from src.model.predict import predict
from src.api.routers import router

app = FastAPI()
# 라우터 등록
app.include_router(router)

def main():
    # argparse를 사용하여 명령어와 옵션들을 정의합니다.
    parser = argparse.ArgumentParser(description="Smart Customer Analysis System")
    # 명령어 정의 (train, predict)
    parser.add_argument('command', choices=['train', 'predict'], help="Command to run")
    # train 옵션
    parser.add_argument('--epochs', type=int, default=3, help="Number of epochs for training")
    # predict 옵션
    parser.add_argument('--text', type=str, help="Text to predict")

    args = parser.parse_args()

    if args.command == 'train':
        logger.info("모델 학습을 시작합니다...")
        train_main(epochs=args.epochs)

    elif args.command == 'predict':
        if not args.text:
            logger.info("Error: --text 인자를 입력하세요.")
            sys.exit(1)
        label = predict(args.text)
        logger.info(f"Predicted label: {label}")

    elif args.command == 'serve':
        logger.info("서버가 시작됩니다...")
        uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
