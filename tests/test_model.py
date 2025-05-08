import pytest
import torch
import pandas as pd
from transformers import BertTokenizer
from src.model.model import create_model
from src.data.dataset import TextDataset

@pytest.fixture
def tokenizer():
    return BertTokenizer.from_pretrained('bert-base-uncased')

@pytest.fixture
def dataset(tokenizer):
    # 예시 데이터
    data = pd.DataFrame({'text': ['This is a test.', 'Another test sentence.'],
                         'label': [0, 1]})
    return TextDataset(data, tokenizer)

# 모델 생성 테스트
def test_model_creation():
    model = create_model()
    assert model is not None
    assert isinstance(model, torch.nn.Module)

# 데이터셋 샘플 테스트
def test_dataset_length(dataset):
    assert len(dataset) == 2

def test_dataset_output(dataset):
    sample = dataset[0]
    assert 'input_ids' in sample
    assert 'attention_mask' in sample
    assert 'labels' in sample