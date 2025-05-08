import pytest
import dask.bag as db
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

# 데이터 로딩 및 전처리
@pytest.fixture
def load_data():
    bag = db.read_text('../data/test.ft.txt')
    
    def parse_line(line):
        label, text = line.split(' ', 1)  # 첫 번째 공백 기준으로만
        label = int(label.replace('__label__', ''))  # __label__1 → 1, __label__2 → 2
        return {'label': label, 'text': text}
    
    parsed = bag.map(parse_line)
    df = parsed.to_dataframe()
    return df

# 데이터셋을 텍스트와 레이블로 분리한 후, PyTorch 텐서로 변환
class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.texts = dataframe['text'].tolist()
        self.labels = dataframe['label'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 텍스트를 토큰화하고, 텐서로 변환
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 데이터 로딩 테스트
def test_loading(load_data):
    df = load_data
    print(df.columns)
    assert df.shape[1] == 2  # 두 개의 컬럼이 있는지
    assert 'label' in df.columns  # 'label' 컬럼이 있는지
    assert 'text' in df.columns  # 'text' 컬럼이 있는지

# 모델 훈련을 위한 데이터셋 준비
def test_data_preparation(load_data):
    df = load_data
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    dataset = TextDataset(df, tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # 데이터 배치 확인
    batch = next(iter(dataloader))
    print(batch)
    assert 'input_ids' in batch
    assert 'attention_mask' in batch
    assert 'labels' in batch

# 모델 학습 함수 (PyTorch 모델에 맞는 텍스트 분류 모델을 정의 후 학습)
def train_model(dataloader, model, optimizer, criterion, device):
    model.train()
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        
        loss.backward()
        optimizer.step()

# 텍스트 분류 모델 정의 (예: BERT 기반 모델)
from transformers import BertForSequenceClassification, AdamW

def create_model(num_labels=2):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    return model

# 모델, 옵티마이저, 손실 함수 설정
def test_model_training(load_data):
    df = load_data
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = TextDataset(df, tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = create_model(num_labels=2)  # 예시: 2개의 클래스
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    optimizer = AdamW(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 모델 학습
    train_model(dataloader, model, optimizer, criterion, device='cuda' if torch.cuda.is_available() else 'cpu')
