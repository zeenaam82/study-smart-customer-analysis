import torch
import os

from src.model.model import BertClassifier
from transformers import BertTokenizer
from src.utils.config import BASE_DIR

MODEL_PATH = os.path.join(BASE_DIR, 'save_models', 'text_classification_model.pt')
TOKENIZER_NAME = 'bert-base-uncased'

def load_model(device):
    # 모델과 토크나이저를 로드합니다.
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)
    model = BertClassifier()
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, tokenizer

def predict(text: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = load_model(device)

    # 입력 텍스트 처리
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # 예측
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predicted_class = torch.argmax(outputs, dim=1).item()

    return predicted_class
