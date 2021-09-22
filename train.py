import argparse
import torch
import collections
import numpy as np
from torch.optim import Adam
from models.query_encoder.bert import BertEncoder

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
    train_data_loader = ...
    valid_data_loader = ...
    test_data_loader = ...    

    model = BertEncoder()
    model = model.to(device)

    optimizer = Adam(model.params(),lr=0.001)
    
    loss_fn = ...

    for epoch in range(epochs):
        model.train()

        for i, batch in enumerate(train_data_loader):
            input_ids, attention_masks, labels = batch
            input_ids.to(device); attention_masks.to(device); labels.to(device)

            model.zero_grad()
            preds = model(input_ids, attention_masks)

            loss = loss_fn(preds, labels)

            loss.backward()
            optimizer.step()

        val_loss = evaluation(model, valid_data_loader)


def evaluation(model, valid_data_loader):
    model.eval()

    for batch in valid_data_loader:
        input_ids, attention_masks, labels = batch
        input_ids.to(device); attention_masks.to(device); labels.to(device)

        with torch.no_grad():
            preds = model(input_ids, attention_masks)

        loss = loss_fn(preds, labels)
        ...

        return loss



if __name__ == '__main__':
    main()