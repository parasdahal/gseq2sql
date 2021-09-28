import torch
import numpy as np
from torch.optim import Adam
from torch.utils.data.sampler import RandomSampler
from models.query_encoder.bert import BertEncoder
from data.dataset import SpiderDataset
from torch.utils.data import DataLoader, RandomSampler
from models.seq2seq import decoder

SOS_TOKEN = 101
EOS_TOKEN = 102

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


def train_step(input, attention_masks, target, loss_fn, bert, decoder, 
               bert_optimizer, decoder_optimizer):
    
    decoder_optimizer.zero_grad();
    
    input_length = input.size(0)
    target_length = target.size(0)
    
    bert_outputs = bert(input, attention_masks) # Generate bert outputs here...
    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = bert_outputs
    loss = 0
    # Generate token and compute loss in each timestep.
    for i in range(target_length):
        decoder_output, decoder_hidden, attetion_weights = decoder(
            decoder_input, decoder_hidden, bert_outputs)
        decoder_input = decoder_output
        loss += loss_fn(decoder_output, target[i])
        if decoder_input.item() == EOS_token:
            break
    loss.backward()
    
    bert_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / target_length, attetion_weights
        
    
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

    bert = BertEncoder()
    decoder = decoder.Decoder(hidden_size=512, output_size=300)
    
    bert = bert.to(device)
    decoder = decoder.to(device)

    bert_optimizer = Adam(bert.params(),lr=0.001)
    decoder_optimizer = Adam(decoder.params(),lr=0.001)
    
    loss_fn = nn.NLLLoss()

    for epoch in range(epochs):
        bert.train(); decoder.train()

        for i, batch in enumerate(train_dataloader):
            input_ids, attention_masks, labels = batch
            input_ids.to(device); attention_masks.to(device);labels.to(device)
            
            train_loss, attetion_weights = train_step(input_ids, attention_masks, labels, 
                    loss_fn, bert, decoder, bert_optimizer, decoder_optimizer)
            

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