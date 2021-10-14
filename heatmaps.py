import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data.sampler import RandomSampler
from models.query_encoder.bert import BertEncoder
from models.query_decoder.lstm import LSTMDecoder
from datasets.dataset import create_splits
from torch.utils.data import DataLoader, RandomSampler
from eval import ids_to_string
import Levenshtein
from att_visualization import generate_heatmap

SOS_TOKEN = 101
EOS_TOKEN = 102

def train_step(iter, input, attention_masks, target, loss_fn, bert, decoder, dataset_size,
               bert_optimizer, decoder_optimizer, batch_size, effective_batch_size,
               teacher_forcing, device, verbose=False):
    
    bert.load_state_dict(torch.load("checkpoints-kevin/bert-state-dict.pth", map_location='cpu'))
    bert.eval()

    decoder.load_state_dict(torch.load("checkpoints-kevin/decoder-state-dict.pth", map_location='cpu'))
    decoder.eval()

    bert.eval()
    decoder.eval()

    
    bert_outputs, bert_all = bert(input, attention_masks)
    
    batch_outputs = []; batch_expected = []
    
    loss = 0
    # Iterate over samples in the batch.
    for batch_i in range(4):
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
            decoder_output, decoder_hidden, attn_weights = decoder(
                decoder_input, decoder_hidden, bert_all[batch_i])

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
    
        inputtokens = input[batch_i]
        inputstr = ids_to_string(inputtokens)
        targettokens = target[batch_i]
        targetstr = ids_to_string(targettokens)
        predictedtokens = expected_output
        predictedstr = ids_to_string(predictedtokens)

        import torch.nn.functional as F

        similarity = Levenshtein.distance(targetstr, predictedstr)
        tokenized_input = inputstr.split()
        lenn = len(tokenized_input)
        weights = attn_weights[0][0].tolist()[:lenn]
        # rescaled = np.asarray([i*10000 for i in weights])
        # toone = rescaled/ rescaled.sum()
        generate_heatmap(tokenized_input, weights, "latex"+str(batch_i)+".tex")
        continue

    import sys; sys.exit(0)


def train(args):
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enable = False
    
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

    args.vocab_size = train_dataset.get_vocab_size()

    bert = BertEncoder(args.vocab_size)
    decoder = LSTMDecoder(hidden_size=args.hidden_dim, output_size=args.vocab_size,
                          use_attention=args.use_attention)
    
    bert = bert.to(device)
    decoder = decoder.to(device)

    bert_optimizer = Adam(bert.parameters(),lr=args.lr)
    decoder_optimizer = Adam(decoder.parameters(),lr=args.lr)
    
    loss_fn = nn.NLLLoss()
    
    if(args.teacher_forcing): print('Using teacher forcing for training...')

    for epoch in range(args.epochs):
        print('Training epoch: ', epoch)
        bert.train(); decoder.train()

        for i, batch in enumerate(train_dataloader):
            input_ids, attention_masks, labels, db_id, or_queries = batch
            input_ids, attention_masks, labels = input_ids.to(device), \
                attention_masks.to(device), labels.to(device)
            
            train_loss = train_step(i, input_ids, attention_masks, labels, 
                    loss_fn, bert, decoder, len(train_dataloader), bert_optimizer, decoder_optimizer, 
                    args.batch_size, args.effective_batch_size, args.teacher_forcing, device, args.verbose)


if __name__ == '__main__':
    args = parse_args()
    train(args)
