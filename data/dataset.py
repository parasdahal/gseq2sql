import os
import json
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer

class SpiderDataset(Dataset):

  def __init__(self, json_file):
    questions = []
    queries = []

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    with open(json_file) as f:
      data = json.load(f)
      for item in data:
        questions.append(item['question'])
        queries.append(item['query'])

    tokenized_questions = tokenizer(questions, truncation=True, padding=True, add_special_tokens=True)
    tokenized_queries = tokenizer(queries, truncation=True, padding=True, add_special_tokens=True)
    self.input_ids = tokenized_questions['input_ids']
    self.att_mask = tokenized_questions['attention_mask']
    self.queries = tokenized_queries['input_ids']

  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, i):
    return torch.tensor(self.input_ids[i]), torch.tensor(self.att_mask[i]), torch.tensor(self.queries[i])

def create_dataloader(json_file, batch_size=1):
  dataset = SpiderDataset(json_file)
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  return dataloader

# train_dataloader = create_dataloader('spider/train_spider.json')
