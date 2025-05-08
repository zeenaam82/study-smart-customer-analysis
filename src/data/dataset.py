from torch.utils.data import Dataset
from transformers import BertTokenizer

class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer_name='bert-base-uncased', max_length=128):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.texts = dataframe['text'].tolist()
        self.labels = dataframe['label'].apply(lambda x: x - 1).tolist()  # 1을 0으로, 2를 1로 변경
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=self.max_length)
        input_ids = inputs["input_ids"].squeeze(0)  # 텐서 차원 축소
        attention_mask = inputs["attention_mask"].squeeze(0)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": label}