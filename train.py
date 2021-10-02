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

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def train_step(input, attention_masks, target, loss_fn, bert, decoder, 
               bert_optimizer, decoder_optimizer, device):
    
    bert_optimizer.zero_grad();
    decoder_optimizer.zero_grad();
    
    bert_outputs = bert(input, attention_masks)
    batch_size, hidden_dim = bert_outputs.size()
    target_size = target.size(0)
    
    loss = 0
    for batch_i in range(batch_size):
        # Size = [1, 1]
        decoder_input = torch.tensor([[SOS_TOKEN]], device=device)
        # Size = [1, 1, hidden_dim]
        h0 = bert_outputs[batch_i].unsqueeze(0).unsqueeze(0)
        c0 = torch.zeros(1, 1, hidden_dim).to(device)
        
        decoder_hidden = (h0, c0)
        
        # Generate token and compute loss in each timestep.
        loss_ = 0
        for target_i in range(target_size):

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, bert_outputs)
            
            _, vocab_id = decoder_output.topk(1)
            decoder_input = vocab_id.squeeze().detach()
            expected_target = torch.tensor([target[batch_i][target_i]], device=device)
            loss_ += loss_fn(decoder_output, expected_target)
            if decoder_input.item() == EOS_TOKEN:
                break
        loss += loss_ / target_size
    loss.backward()
    bert_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / batch_size
    
def train(args):
    
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print('There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    
    # Setup data_loader instances.
    train_dataset = SpiderDataset(args.dataset_path,'train_spider.json')
    valid_dataset = SpiderDataset(args.dataset_path,'dev.json')
    # train_dataset, valid_dataset = random_split(dataset, [0.8, 0.2])

    train_dataloader = DataLoader(train_dataset,
                        sampler=RandomSampler(train_dataset),
                        batch_size=args.batch_size) # TODO change batch size?
    valid_dataloader = DataLoader(valid_dataset,
                        sampler=RandomSampler(valid_dataset),
                        batch_size=args.batch_size) 

    args.vocab_size = train_dataset.get_vocab_size()

    bert = BertEncoder()
    decoder = Decoder(hidden_size=args.hidden_dim, output_size=args.vocab_size)
    
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
            
            train_loss = train_step(input_ids, attention_masks, labels, 
                    loss_fn, bert, decoder, bert_optimizer, decoder_optimizer, device)
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
    train(args)
