import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from datasets.schema_info import SchemaInfo

from transformers import BertTokenizer

class SpiderDataset(Dataset):

  def __init__(self, questions, queries, dbs, schema_file=None):
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    self.dbs = dbs

    if schema_file:
      # Read schema information from table.json
      self.schema_info = SchemaInfo(schema_file)
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

def create_splits(dataset_path, json_files, seed=None, use_schema=False):
  questions = []
  queries = []
  dbs = []

  if type(json_files) is str:
    json_files = [json_files]

  for json_file in json_files:
    with open(os.path.join(dataset_path, json_file)) as f:
      data = json.load(f)
      for item in data:
        questions.append(item['question'])
        queries.append(item['query'])
        dbs.append(item['db_id'])

  train_questions, val_questions, train_queries, val_queries, train_dbs, val_dbs = train_test_split(questions, queries, dbs, test_size=0.2, random_state=seed, stratify=dbs)

  if use_schema:
    schema_file = os.path.join(dataset_path, 'tables.json')
  else:
    schema_file = None

  train_dataset = SpiderDataset(train_questions, train_queries, train_dbs, schema_file=schema_file)
  val_dataset = SpiderDataset(val_questions, val_queries, val_dbs, schema_file=schema_file)

  return train_dataset, val_dataset

if __name__ == '__main__':
  # test
  train_dataset, val_dataset = create_splits('./data/spider', ['train_spider.json', 'dev.json'])
  print(len(train_dataset), len(val_dataset))


