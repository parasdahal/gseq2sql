import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Attention(nn.Module):
  def __init__(self, hidden_size, max_length):
    super(Attention, self).__init__()
    self.attn = nn.Linear(hidden_size * 2, max_length)
    self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
    
  def forward(self, embedded, hidden, encoder_outputs):
    
    attn_weights = F.softmax(
    self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1).unsqueeze(0)
    # print('attn wt', attn_weights.size())
    # print('bert ops ', encoder_outputs.size())
    attn_applied = torch.bmm(attn_weights,
                              encoder_outputs.unsqueeze(0))

    output = torch.cat((embedded[0], attn_applied[0]), 1)
    output = self.attn_combine(output).unsqueeze(0)
    return output, attn_weights
    
class LSTMDecoder(nn.Module):
  def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=512,
               use_attention=False, bidirectional=False, num_layers=1):
    super(LSTMDecoder, self).__init__()
    self.hidden_size = hidden_size
    self.use_attention = use_attention
    self.embedding = nn.Embedding(output_size, hidden_size)
    self.dropout = nn.Dropout(dropout_p)
    if use_attention:
      self.attention = Attention(hidden_size, max_length)
    self.rnn = nn.LSTM(hidden_size, hidden_size, 
                       num_layers=num_layers, 
                       bidirectional=bidirectional, 
                       batch_first=True)
    self.out = nn.Linear(hidden_size, output_size)

  def forward(self, input, hidden, encoder_outputs):
    embedded = self.embedding(input).view(1, 1, -1)
    output = self.dropout(embedded)
    
    if self.use_attention:
      # print(output.size())
      # print(hidden[0].size())
      # print(encoder_outputs.size())
      output, attn_weights = self.attention(output, hidden[0], encoder_outputs);

    output = F.relu(output)
    # print(hidden.size(), output.size())
    output, hidden = self.rnn(output, hidden)

    output = F.log_softmax(self.out(output[0]), dim=1)
    if self.use_attention : return output, hidden, attn_weights
    return output, hidden

  def init_hidden(self, device):
    if not device:
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.zeros(1, 1, self.hidden_size, device=device)