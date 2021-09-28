import torch
import numpy as np
from torch.optim import Adam
from torch.utils.data.sampler import RandomSampler
from models.query_encoder.bert import BertEncoder
from data.dataset import SpiderDataset
from torch.utils.data import DataLoader, RandomSampler


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


def main():


    # setup data_loader instances
    train_dataset = SpiderDataset('spider/train_spider.json')
    valid_dataset = SpiderDataset('spider/dev.json')
    # train_dataset, valid_dataset = random_split(dataset, [0.8, 0.2])

    train_dataloader = DataLoader(train_dataset,
                        sampler=RandomSampler(train_dataset),
                        batch_size=32) # TODO change batch size?
    valid_dataloader = DataLoader(valid_dataset,
                        sampler=RandomSampler(valid_dataset),
                        batch_size=32) 

    model = BertEncoder()
    model = model.to(device)

    optimizer = Adam(model.params(),lr=0.001)
    
    loss_fn = ...

    for epoch in range(epochs):
        model.train()

        for i, batch in enumerate(train_dataloader):
            input_ids, attention_masks, labels = batch
            input_ids.to(device); attention_masks.to(device); labels.to(device)

            model.zero_grad()
            preds = model(input_ids, attention_masks)

            loss = loss_fn(preds, labels)

            loss.backward()
            optimizer.step()

        val_loss = evaluation(model, valid_dataloader)


def evaluation(model, valid_dataloader):
    model.eval()

    for batch in valid_dataloader:
        input_ids, attention_masks, labels = batch
        input_ids.to(device); attention_masks.to(device); labels.to(device)

        with torch.no_grad():
            preds = model(input_ids, attention_masks)

        loss = loss_fn(preds, labels)
        ...

        return loss



if __name__ == '__main__':
    main()