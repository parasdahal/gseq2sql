from transformers import BertModel
import torch.nn as nn

class BertEncoder(nn.Module):
    def __init__(self, vocab_size):
        super(BertEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased" # TODO: discuss which type we need
        )
        self.bert.resize_token_embeddings(vocab_size)

    def forward(self, input_ids, attention_masks):
        # import pdb; pdb.set_trace()
        outputs = self.bert(input_ids=input_ids, 
                            attention_mask=attention_masks)

        # As done in https://skimai.com/fine-tuning-bert-for-sentiment-analysis/
        last_hidden_state_cls = outputs[0][:, 0, :]

        return last_hidden_state_cls
