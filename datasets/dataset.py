import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

from transformers import BertTokenizer

class SpiderDataset(Dataset):

  def __init__(self, dataset_path, json_file):
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Read questions, queries and db_ids from the train/dev json file
    questions = []
    queries = []
    self.dbs = []
    
    with open(os.path.join(dataset_path, json_file)) as f:
      data = json.load(f)
      for item in data:
        questions.append(item['question'])
        queries.append(item['query'])
        self.dbs.append(item['db_id'])

    # Read schema information from table.json
    self.schema_info = {}
    with open(os.path.join(dataset_path, 'tables.json')) as f:
      data = json.load(f)
      for item in data:
        db_schema = defaultdict(list)
        table_names = item['table_names']
        column_names = item['column_names']
        for column_inf in column_names:
          table_idx = column_inf[0]
          if table_idx < 0:
            continue
          table_name, column_name = table_names[table_idx], column_inf[1]
          db_schema[table_name].append(column_name)
        self.schema_info[item['db_id']] = db_schema
        
    # Tokenize questions and queries
    tokenized_questions = self.tokenizer(questions, truncation=True, padding=True, add_special_tokens=True)
    tokenized_queries = self.tokenizer(queries, truncation=True, padding=True, add_special_tokens=True)
    self.input_ids = tokenized_questions['input_ids']
    self.att_mask = tokenized_questions['attention_mask']
    self.queries = tokenized_queries['input_ids']

  def get_vocab_size(self):
    return self.tokenizer.vocab_size

  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, i):
    db_id = self.dbs[i]
    db_schema = self.schema_info[db_id]
    return torch.tensor(self.input_ids[i]), torch.tensor(self.att_mask[i]), torch.tensor(self.queries[i]), self.dbs[i]

def create_dataloader(dataset_path, json_file, batch_size=1):
  dataset = SpiderDataset(dataset_path, json_file)
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  return dataloader

# train_dataloader = create_dataloader('spider/train_spider.json')
train_dataset = SpiderDataset('./datasets/spider', 'train_spider.json')

train_dataset.__getitem__(0)