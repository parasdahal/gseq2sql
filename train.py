import torch, csv, pickle
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data.sampler import RandomSampler
from models.query_encoder.bert import BertEncoder
from datasets.dataset import SpiderDataset
from torch.utils.data import DataLoader, RandomSampler
from models.seq2seq.decoder import Decoder
from utils import parse_args, EarlyStopping

SOS_TOKEN = 101
EOS_TOKEN = 102

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enable = False

def train_step(input, attention_masks, target, loss_fn, bert, decoder, 
               bert_optimizer, decoder_optimizer, device):
    
    bert_optimizer.zero_grad();
    decoder_optimizer.zero_grad();
    
    bert_outputs = bert(input, attention_masks)
    batch_size, hidden_dim = bert_outputs.size()
    #target_size = target.size(0)
    
    loss = 0
    for batch_i in range(batch_size):
        # Size = [1, 1]
        decoder_input = torch.tensor([[SOS_TOKEN]], device=device)
        # Size = [1, 1, hidden_dim]
        h0 = bert_outputs[batch_i].unsqueeze(0).unsqueeze(0)
        c0 = torch.zeros(1, 1, hidden_dim).to(device)
        
        decoder_hidden = (h0, c0)

        try:
            target_size = list(target[batch_i]).index(0)
        except:
            target_size = target[batch_i].size(0)
        
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
    early_stopping = EarlyStopping()

    bert = BertEncoder()
    decoder = Decoder(hidden_size=args.hidden_dim, output_size=args.vocab_size)
    
    bert = bert.to(device)
    decoder = decoder.to(device)

    bert_optimizer = Adam(bert.parameters(),lr=args.lr)
    decoder_optimizer = Adam(decoder.parameters(),lr=args.lr)
    
    loss_fn = nn.NLLLoss()
    sum_loss = 0

    for epoch in range(args.epochs):
        print('Training epoch: ', epoch)
        bert.train(); decoder.train()

        sum_loss = 0
        for i, batch in enumerate(train_dataloader):
            input_ids, attention_masks, labels, db_id = batch
            input_ids, attention_masks, labels = input_ids.to(device), \
                attention_masks.to(device), labels.to(device)
            
            train_loss = train_step(input_ids, attention_masks, labels, 
                    loss_fn, bert, decoder, bert_optimizer, decoder_optimizer, device)
            sum_loss += train_loss
            print('Batch loss: ', train_loss)
        
        epoch_loss = sum_loss/i
        print("Epoch loss: ", epoch_loss)
        if early_stopping(epoch_loss):
            break

    evaluation(bert, decoder, loss_fn, valid_dataloader)
    
    torch.save(bert.state_dict(), './checkpoints/bert-state-dict')
    torch.save(decoder.state_dict(), './checkpoints/decoder-state-dict')


def valid_step(input, attention_masks, target, loss_fn, bert, decoder, device):

    bert.eval(); decoder.eval()
    batch_outputs = []; batch_expected = []

    with torch.no_grad():
        bert_outputs = bert(input, attention_masks)
        batch_size, hidden_dim = bert_outputs.size()
        #target_size = target.size(0)
    
        loss = 0
        for batch_i in range(batch_size):
            # Size = [1, 1]
            decoder_input = torch.tensor([[SOS_TOKEN]], device=device)
            # Size = [1, 1, hidden_dim]
            h0 = bert_outputs[batch_i].unsqueeze(0).unsqueeze(0)
            c0 = torch.zeros(1, 1, hidden_dim).to(device)
        
            decoder_hidden = (h0, c0)

            try:
                target_size = list(target[batch_i]).index(0)
            except:
                target_size = target[batch_i].size(0)

            # Generate token and compute loss in each timestep.
            loss_ = 0; gen_output = []; expected_output= []
            for target_i in range(target_size):

                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, bert_outputs)
                
                _, vocab_id = decoder_output.topk(1)
                gen_output.extend([vocab_id.item()])

                decoder_input = vocab_id.squeeze().detach()
                expected_target = torch.tensor([target[batch_i][target_i]], device=device)
                expected_output.extend(expected_target)

                loss_ += loss_fn(decoder_output, expected_target)
                if decoder_input.item() == EOS_TOKEN:
                    break
            loss += loss_ / target_size
            batch_outputs.append(gen_output)
            batch_expected.append(expected_output)

    return loss.item() / batch_size, batch_outputs, batch_expected




def evaluation(bert, decoder, loss_fn, valid_dataloader):
    
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print('There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    
    print('Starting validation')
    bert.eval(); decoder.eval()

    total_generated = []; total_expected = []; total_dbid = []

    for i, batch in enumerate(valid_dataloader):
        input_ids, attention_masks, labels, db_id = batch
        input_ids, attention_masks, labels = input_ids.to(device), \
            attention_masks.to(device), labels.to(device)
        
        train_loss, generated, expected = valid_step(input_ids, attention_masks, labels, 
                loss_fn, bert, decoder, device)
        print('Batch loss: ', train_loss)
        total_generated.append(generated); total_expected.append(expected); total_dbid.append(db_id)

    create_csv(total_generated, total_expected, total_dbid)

def create_csv(generated, expected, dbid):

    #import pdb; pdb.set_trace()
    with open('generated.pkl', 'wb') as f:
      pickle.dump(generated, f)
    with open('expected.pkl', 'wb') as f:
      pickle.dump(expected, f)
    with open('dbid.pkl', 'wb') as f:
      pickle.dump(dbid, f)

    #import pdb; pdb.set_trace()
    with open('outputs.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for (gen, exp, dbid) in zip(generated, expected, dbid):
            for (g, e, db) in zip(gen, exp, dbid):
                writer.writerow([g, e, db])



if __name__ == '__main__':
    args = parse_args()
    train(args)
