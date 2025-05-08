import pytest
import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader

from src.data.data_loader import load_data_pandas
from src.data.dataset import TextDataset
from src.model.model import BertClassifier

@pytest.fixture
def dataset_and_loader():
    df = load_data_pandas()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = TextDataset(df, tokenizer, max_length=64)
    dataloader = DataLoader(dataset, batch_size=4)
    return dataset, dataloader

def test_dataset_length(dataset_and_loader):
    dataset, _ = dataset_and_loader
    assert len(dataset) > 0

def test_batch_structure(dataset_and_loader):
    _, dataloader = dataset_and_loader
    batch = next(iter(dataloader))
    assert 'input_ids' in batch
    assert 'attention_mask' in batch
    assert 'labels' in batch
    assert batch['input_ids'].shape[0] == 4  # batch size

def test_model_forward_pass(dataset_and_loader):
    _, dataloader = dataset_and_loader
    batch = next(iter(dataloader))

    model = BertClassifier(num_classes=2)
    model.eval()
    
    with torch.no_grad():
        outputs = model(batch['input_ids'], batch['attention_mask'])

    assert outputs.shape[0] == 4  # batch size
    assert outputs.shape[1] == 2  # num_classes