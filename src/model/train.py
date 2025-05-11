import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from src.utils.logger import logger
from src.data.data_loader import load_data_pandas
from src.data.dataset import TextDataset
from src.model.model import BertClassifier
from src.utils.config import BASE_DIR

def get_optimizer(model, learning_rate=2e-5):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    return optimizer

def get_model(num_labels=2):
    model = BertClassifier(num_classes=num_labels)
    return model

def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0

    # 실시간 상태 확인
    loop = tqdm(train_loader, desc="Training", leave=True)

    for batch in loop:
        optimizer.zero_grad()  # 기울기 초기화
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 모델 예측과 loss 계산
        loss, _ = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)  # loss를 받음
        
        loss.backward()  # 역전파 계산
        optimizer.step()  # 가중치 업데이트
        
        total_loss += loss.item()  # 전체 손실을 누적

        # tqdm description 업데이트
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)  # 평균 손실
    return avg_loss

def evaluate_training(model, val_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    loop = tqdm(val_loader, desc="Evaluating", leave=True)
    
    with torch.no_grad():
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            _, logits = model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          labels=labels)
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            # 현재까지 예측 개수 표시
            loop.set_postfix(batch=len(all_preds))
    
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

def main(epochs=1, batch_size=8, learning_rate=1e-4, max_length=16):
    # epochs=3, batch_size=32, learning_rate=2e-5, max_length=128

    logger.info("데이터 로드를 시작합니다.")
    df = load_data_pandas()
    logger.info("데이터를 로드했습니다.")

    train_dataset = TextDataset(df, tokenizer_name="bert-base-uncased", max_length=max_length)
    logger.info("데이터셋을 생성했습니다.")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    logger.info("데이터로더를 생성했습니다.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"디바이스 설정 완료: {device}")

    model = get_model(num_labels=2).to(device)
    logger.info("모델을 초기화했습니다.")

    optimizer = get_optimizer(model, learning_rate)
    logger.info("옵티마이저를 초기화했습니다.")

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        avg_loss = train(model, train_loader, optimizer, device)
        logger.info(f"훈련 손실 값: {avg_loss:.4f}")

        accuracy = evaluate_training(model, train_loader, device)
        logger.info(f"훈련 정확도: {accuracy:.4f}")

    # 모델 저장
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'save_models', 'text_classification_model.pt')
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    logger.info(f"모델이 다음 경로에 저장되었습니다: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
