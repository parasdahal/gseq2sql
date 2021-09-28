import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data.sampler import RandomSampler
from models.query_encoder.bert import BertEncoder
from data.dataset import SpiderDataset
from torch.utils.data import DataLoader, RandomSampler
from models.seq2seq.decoder import Decoder
from utils import parse_args

SOS_TOKEN = 101
EOS_TOKEN = 102

def train_step(input, attention_masks, target, loss_fn, bert, decoder, 
               bert_optimizer, decoder_optimizer):
    
    decoder_optimizer.zero_grad();
    
    input_length = input.size(0)
    target_length = target.size(0)
    
    bert_outputs = bert(input, attention_masks) # Generate bert outputs here...
    decoder_input = torch.tensor([[SOS_TOKEN]], device=device)
    decoder_hidden = bert_outputs
    loss = 0
    # Generate token and compute loss in each timestep.
    for i in range(target_length):
        decoder_output, decoder_hidden, attetion_weights = decoder(
            decoder_input, decoder_hidden, bert_outputs)
        _, vocab_id = decoder_output.topk(1)
        decoder_input = vocab_id.squeeze().detach()
        
        loss += loss_fn(decoder_output, target[i])
        if decoder_input.item() == EOS_TOKEN:
            break
    loss.backward()
    
    bert_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / target_length, attetion_weights
        
    
def train(args):

    # Setup data_loader instances.
    train_dataset = SpiderDataset('data/spider/train_spider.json')
    valid_dataset = SpiderDataset('data/spider/dev.json')
    # train_dataset, valid_dataset = random_split(dataset, [0.8, 0.2])

    train_dataloader = DataLoader(train_dataset,
                        sampler=RandomSampler(train_dataset),
                        batch_size=args.batch_size) # TODO change batch size?
    valid_dataloader = DataLoader(valid_dataset,
                        sampler=RandomSampler(valid_dataset),
                        batch_size=args.batch_size) 

    args.vocab_size = train_dataset.get_vocab_size()

    bert = BertEncoder()
    decoder = Decoder(hidden_size=args.dec_hidden_dim, output_size=args.vocab_size)
    
    bert = bert.to(device)
    decoder = decoder.to(device)

    bert_optimizer = Adam(bert.parameters(),lr=args.lr)
    decoder_optimizer = Adam(decoder.parameters(),lr=args.lr)
    
    loss_fn = nn.NLLLoss()

    for epoch in range(args.epochs):
        print('Training epoch: ', epoch)
        bert.train(); decoder.train()

        for i, batch in enumerate(train_dataloader):
            input_ids, attention_masks, labels = batch
            input_ids, attention_masks, labels = input_ids.to(device), \
                attention_masks.to(device), labels.to(device)
            
            train_loss, attetion_weights = train_step(input_ids, attention_masks, labels, 
                    loss_fn, bert, decoder, bert_optimizer, decoder_optimizer)
            print('Batch loss: ', train_loss)

        # val_loss = evaluation(model, valid_dataloader)


def evaluation(model, valid_dataloader):
    return NotImplementedError()
    # model.eval()

    # for batch in valid_dataloader:
    #     input_ids, attention_masks, labels = batch
    #     input_ids.to(device); attention_masks.to(device); labels.to(device)

    #     with torch.no_grad():
    #         preds = model(input_ids, attention_masks)

    #     loss = loss_fn(preds, labels)
    #     ...

    #     return loss



if __name__ == '__main__':
    args = parse_args()
    
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print('There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    
    train(args)
