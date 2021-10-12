import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import itertools
from datasets.schema_info import SchemaInfo

from transformers import BertTokenizer

class SpiderDataset(Dataset):

  def __init__(self, dataset_path, json_file, use_schema=False):
    self.dataset_path = dataset_path
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

    if use_schema:
      # Read schema information from table.json
      self.schema_info = SchemaInfo(os.path.join(self.dataset_path, 'tables.json'))
      self.tokenizer.add_tokens(['[T]', '[C]'])
      added_tokens = self.tokenizer.add_tokens(self.schema_info.get_tokens())
      self.num_added_tokens = added_tokens
      print(f'Added {added_tokens} schema tokens to vocabulary')

      # Add schema info to questions
      questions = self.add_schema_info(questions)

    # Tokenize questions and queries
    tokenized_questions = self.tokenizer(questions, truncation=True, padding=True, add_special_tokens=True)
    tokenized_queries = self.tokenizer(queries, truncation=True, padding=True, add_special_tokens=True)
    self.input_ids = tokenized_questions['input_ids']
    self.att_mask = tokenized_questions['attention_mask']
    self.queries = tokenized_queries['input_ids']

  def get_vocab_size(self):
    return len(self.tokenizer)

  def add_schema_info(self, questions):
    # Append correct schema to each question
    return [question + ' ' + self.schema_info.get_schema_string(self.dbs[i]) for i, question in enumerate(questions)]

  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, i):
    return torch.tensor(self.input_ids[i]), torch.tensor(self.att_mask[i]), torch.tensor(self.queries[i]), self.dbs[i]

def create_dataloader(dataset_path, json_file, batch_size=1):
  dataset = SpiderDataset(dataset_path, json_file)
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  return dataloader

# train_dataloader = create_dataloader('spider/train_spider.json')
# train_dataset = SpiderDataset('./datasets/spider', 'train_spider.json')

# train_dataset.__getitem__(0)
