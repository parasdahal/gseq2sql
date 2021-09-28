import torch
import numpy as np
from torch.optim import Adam
from torch.utils.data.sampler import RandomSampler
from models.query_encoder.bert import BertEncoder
from data.dataset import SpiderDataset
from torch.utils.data import DataLoader, RandomSampler
from models.seq2seq import decoder

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


def train_step(input, target, loss_fn, decoder_optimizer):
    
    decoder_optimizer.zero_grad();
    
    input_length = input.size(0)
    target_length = target.size(0)
    
    bert_outputs = None # Generate bert outputs here...
    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = bert_output
    loss = 0
    # Generate token and compute loss in each timestep.
    for i in range(target_length):
        decoder_output, decoder_hidden, attetion_weights = decoder(decoder_input, decoder_hidden, bert_outputs)
        decoder_input = decoder_output
        loss += loss_fn(decoder_output, target[i])
        if decoder_input.item() == EOS_token:
            break
    loss.backward()
    decoder_optimizer()
    return loss.item() / target_length
        
    
def train():


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

    bert_optimizer = Adam(model.params(),lr=0.001)
    decoder_optimizer = Adam(model.params(),lr=0.001)
    
    loss_fn = nn.NLLLoss()

    for epoch in range(epochs):
        model.train()

        for i, batch in enumerate(train_dataloader):
            input_ids, attention_masks, labels = batch
            input_ids.to(device); attention_masks.to(device); labels.to(device)
            
            train_loss = train_step(input_ids, labels, loss_fn, decoder_optimizer)
            

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
    train()