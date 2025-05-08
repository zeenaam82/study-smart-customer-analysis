import torch
import os

from src.model.model import BertClassifier
from transformers import BertTokenizer
from src.utils.config import BASE_DIR

MODEL_PATH = os.path.join(BASE_DIR, 'save_models', 'text_classification_model.pt')
TOKENIZER_NAME = 'bert-base-uncased'

def predict(text: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")

    # 호출 시점에 토크나이저와 모델 로드
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)
    model = BertClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predicted_class = torch.argmax(outputs, dim=1).item()

    return predicted_class