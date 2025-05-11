import os
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model.model import BertClassifier
from src.data.data_loader import load_data_pandas
from src.data.dataset import TextDataset
from src.utils.config import BASE_DIR
from src.utils.logger import logger

MODEL_PATH = os.path.join(BASE_DIR, 'save_models', 'text_classification_model.pt')

def load_model(device):
    """모델 로드"""
    model = BertClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    logger.info(f"모델을 {MODEL_PATH}에서 로드했습니다.")
    return model

def evaluate_final(model, data_loader, device):
    """모델 평가"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", unit="batch"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    logger.info(f"정확도: {accuracy:.4f}")
    logger.info(f"정밀도: {precision:.4f}")
    logger.info(f"재현율: {recall:.4f}")
    logger.info(f"F1 점수: {f1:.4f}")
    logger.info("\n" + classification_report(all_labels, all_preds))
    logger.info(f"Confusion Matrix:\n{confusion_matrix(all_labels, all_preds)}")

def main():
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"디바이스 설정 완료: {device}")

    # 테스트 데이터 경로
    test_data_path = os.path.join(BASE_DIR, 'data', 'test.ft.txt')
    
    # 테스트 데이터 로드
    df = load_data_pandas(file_path=test_data_path)
    test_dataset = TextDataset(df, tokenizer_name="bert-base-uncased", max_length=64)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    logger.info("테스트 데이터를 로드했습니다.")

    # 모델 로드
    model = load_model(device)

    # 평가
    logger.info("모델 평가를 시작합니다.")
    evaluate_final(model, test_loader, device)

if __name__ == "__main__":
    main()
