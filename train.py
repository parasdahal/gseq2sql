import torch, csv, pickle
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data.sampler import RandomSampler
from models.query_encoder.bert import BertEncoder
from models.query_decoder.lstm import LSTMDecoder
from datasets.dataset import create_splits
from torch.utils.data import DataLoader, RandomSampler
from utils import parse_args, EarlyStopping, plot_losses
from eval import ids_to_string
import os

SOS_TOKEN = 101
EOS_TOKEN = 102

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enable = False

def train_step(iter, input, attention_masks, target, loss_fn, bert, decoder, dataset_size,
               bert_optimizer, decoder_optimizer, batch_size, effective_batch_size,
               teacher_forcing, device, verbose=False):
    
    
    bert_outputs = bert(input, attention_masks)
    batch_size_, hidden_dim = bert_outputs.size()

    accum_iter = effective_batch_size / batch_size
    #target_size = target.size(0)
    
    batch_outputs = []; batch_expected = []
    
    loss = 0
    # Iterate over samples in the batch.
    for batch_i in range(batch_size_):
        # Size = [1, 1]
        decoder_input = torch.tensor([[SOS_TOKEN]], device=device)
        # Size = [1, 1, hidden_dim]
        h0 = bert_outputs[batch_i].unsqueeze(0).unsqueeze(0)
        # c0 = torch.zeros(1, 1, hidden_dim).to(device)
        c0 = bert_outputs[batch_i].unsqueeze(0).unsqueeze(0)
        
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
            expected_target = torch.tensor([target[batch_i][target_i]], device=device)
            _, vocab_id = decoder_output.topk(1)
            if not teacher_forcing:
                decoder_input = vocab_id.squeeze().detach()
            else:
                decoder_input = expected_target
            loss_ += loss_fn(decoder_output, expected_target)
            
            gen_output.append(vocab_id.item())
            expected_output.append(expected_target.item())
            
            if decoder_input.item() == EOS_TOKEN:
                break
        loss += loss_ / target_size
        
        if verbose and batch_i < 5:
            batch_outputs.append(ids_to_string(gen_output))
            batch_expected.append(ids_to_string(expected_output))
    
    # Print generated and expected strings.
    if verbose:
        for gen, exp in zip(batch_outputs,batch_expected):
            print('-'*80)
            print('Expected: ', exp)
            print('Generated: ', gen)
        print('-'*80)
    loss = loss / accum_iter
    loss.backward()

    if ((iter + 1) % accum_iter == 0) or (iter + 1 == dataset_size):
        print('optimize model')
        bert_optimizer.step()
        decoder_optimizer.step()
        bert_optimizer.zero_grad();
        decoder_optimizer.zero_grad();

    return loss.item() / batch_size * accum_iter
    

def train(args):
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    
    # Setup data_loader instances.
    train_dataset, valid_dataset = create_splits(args.dataset_path, ['train_spider.json', 'dev.json'], use_schema=args.use_schema, seed=args.seed)

    train_dataloader = DataLoader(train_dataset,
                        sampler=RandomSampler(train_dataset),
                        batch_size=args.batch_size)
    valid_dataloader = DataLoader(valid_dataset,
                        sampler=RandomSampler(valid_dataset),
                        batch_size=args.batch_size)

    args.vocab_size = train_dataset.get_vocab_size()
    early_stopping = EarlyStopping()

    bert = BertEncoder(args.vocab_size)
    decoder = LSTMDecoder(hidden_size=args.hidden_dim, output_size=args.vocab_size)
    
    bert = bert.to(device)
    decoder = decoder.to(device)

    bert_optimizer = Adam(bert.parameters(),lr=args.lr)
    decoder_optimizer = Adam(decoder.parameters(),lr=args.lr)
    
    loss_fn = nn.NLLLoss()
    sum_loss = 0
    
    if(args.teacher_forcing): print('Using teacher forcing for training...')

    train_losses, valid_losses = [], []

    for epoch in range(args.epochs):
        print('Training epoch: ', epoch)
        bert.train(); decoder.train()

        sum_loss = 0
        for i, batch in enumerate(train_dataloader):
            input_ids, attention_masks, labels, db_id = batch
            input_ids, attention_masks, labels = input_ids.to(device), \
                attention_masks.to(device), labels.to(device)
            
            train_loss = train_step(i, input_ids, attention_masks, labels, 
                    loss_fn, bert, decoder, len(train_dataloader), bert_optimizer, decoder_optimizer, 
                    args.batch_size, args.effective_batch_size, args.teacher_forcing, device, args.verbose)
            sum_loss += train_loss
            print(f'Batch {i}/{len(train_dataloader)} loss: {train_loss}')
        
        epoch_loss = sum_loss/i
        print(f"Epoch {epoch} loss: {epoch_loss}")
        print("="*80)

        valid_loss = evaluation(bert, decoder, loss_fn, valid_dataloader)

        train_losses.append(epoch_loss)
        valid_losses.append(valid_loss)
        plot_losses(args.log_dir, train_losses, valid_losses)

        if early_stopping(epoch_loss):
            break


    
    print('Training completed. Saving the model...')
    if not os.path.exists('./checkpoints/'):
        os.mkdir('./checkpoints/')
    torch.save(bert.state_dict(), './checkpoints/bert-state-dict.pth')
    torch.save(decoder.state_dict(), './checkpoints/decoder-state-dict.pth')


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
                
                decoder_input = vocab_id.squeeze().detach()
                expected_target = torch.tensor([target[batch_i][target_i]], device=device)
                
                gen_output.extend([vocab_id.item()])
                expected_output.extend([expected_target.item()])

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
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    
    print('Starting validation')
    bert.eval(); decoder.eval()

    total_generated = []; total_expected = []; total_dbid = []

    valid_losses = []

    for i, batch in enumerate(valid_dataloader):
        input_ids, attention_masks, labels, db_id = batch
        input_ids, attention_masks, labels = input_ids.to(device), \
            attention_masks.to(device), labels.to(device)
        
        valid_loss, generated, expected = valid_step(input_ids, attention_masks, labels, 
                loss_fn, bert, decoder, device)
        print('Batch loss: ', valid_loss)
        total_generated.append(generated); total_expected.append(expected); total_dbid.append(db_id)
        valid_losses.append(valid_loss)
    create_csv(total_generated, total_expected, total_dbid)

    return np.mean(valid_losses)

def create_csv(generated, expected, dbid):

    #import pdb; pdb.set_trace()
    with open('generated.pkl', 'wb') as f:
      pickle.dump(generated, f)
    with open('expected.pkl', 'wb') as f:
      pickle.dump(expected, f)
    with open('dbid.pkl', 'wb') as f:
      pickle.dump(dbid, f)

    generated_strings = [[ids_to_string(id) for id in batch] for batch in generated]
    expected_strings = [[ids_to_string(id) for id in batch] for batch in expected]

    #import pdb; pdb.set_trace()
    with open('outputs.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for (gen, exp, dbid, gen_s, exp_s) in zip(generated, expected, dbid, generated_strings, expected_strings):
            for (g, e, db, gs, es) in zip(gen, exp, dbid, gen_s, exp_s):
                writer.writerow([g, e, db, gs, es])



if __name__ == '__main__':
    args = parse_args()
    train(args)
