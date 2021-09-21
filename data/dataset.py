import os
import json
from torch.utils.data import Dataset, DataLoader

class SpiderDataset(Dataset):

  def __init__(self, json_file):
    self.items = []

    with open(json_file) as f:
      data = json.load(f)
      for item in data:
        self.items.append((item['question'], item['query']))

  def __len__(self):
    return len(self.items)

  def __getitem__(self, i):
    return self.items[i]

def create_dataloader(json_file, batch_size=1):
  dataset = SpiderDataset(json_file)
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  return dataloader

# train_dataloader = create_dataloader('spider/train_spider.json')
