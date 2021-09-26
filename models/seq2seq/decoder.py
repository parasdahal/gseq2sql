import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Attention(nn.module):
  def __init__(self, hidden_size, max_length):
    super(Decoder, self).__init__()
    self.attn = nn.Linear(hidden_size * 2, max_length)
    self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
    
  def forward(self, embedded, hidden, encoder_outputs):
    attn_weights = F.softmax(
    self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
    attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                              encoder_outputs.unsqueeze(0))

    output = torch.cat((embedded[0], attn_applied[0]), 1)
    output = self.attn_combine(output).unsqueeze(0)
    return output, attn_weights
    
class Decoder(nn.Module):
  def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=500):
    super(Decoder, self).__init__()
    self.embedding = nn.Embedding(output_size, hidden_size)
    self.dropout = nn.Dropout(dropout_p)
    self.attention = Attention(hidden_size, max_length)
    self.rnn = nn.LSTM(hidden_size, hidden_size, bidirectional=True)
    self.out = nn.Linear(hidden_size, output_size)

  def forward(self, input, hidden, encoder_outputs):
    embedded = self.embedding(input).view(1, 1, -1)
    output = self.dropout(embedded)
    
    output, attn_weights = self.attention(output, hidden, encoder_outputs);
    
    output = F.relu(output)
    output, hidden = self.rnn(output, hidden)

    output = F.log_softmax(self.out(output[0]), dim=1)
    return output, hidden, attn_weights

  def init_hidden(self):
    return torch.zeros(1, 1, self.hidden_size, device=device)