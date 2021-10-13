import torch
from models.query_encoder.bert import BertEncoder
from models.query_decoder.lstm import LSTMDecoder
from torch.utils.data import DataLoader, RandomSampler
from datasets.dataset import create_splits
from utils import parse_args


def load_weights(args):
    bert = BertEncoder(args.vocab_size)
    decoder = LSTMDecoder(hidden_size=args.hidden_dim, output_size=args.vocab_size,
                        use_attention=args.use_attention)
        
    device = 'cpu'
    bert = bert.to(device)
    decoder = decoder.to(device)

    bert.load_state_dict(torch.load("checkpoints/bert-state-dict.pth"))
    bert.eval()

    decoder.load_state_dict(torch.load("checkpoints/decoder-state-dict.pth"))
    decoder.eval()

    train_dataset, valid_dataset = create_splits(args.dataset_path, ['train_spider.json', 'dev.json'], use_schema=args.use_schema, seed=args.seed)
        
    train_dataloader = DataLoader(train_dataset,
                        sampler=RandomSampler(train_dataset),
                        batch_size=args.batch_size)

    for i, batch in enumerate(train_dataloader):
            input_ids, attention_masks, labels, db_id = batch
            input_ids, attention_masks, labels = input_ids.to(device), \
                attention_masks.to(device), labels.to(device)
            
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
    



if __name__ == '__main__':
    args = parse_args()
    load_weights(args)