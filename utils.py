import os
import argparse
import matplotlib.pyplot as plt
import pickle as pkl

class EarlyStopping():
  # source: https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
  def __init__(self, patience=5, min_delta=0):
      self.patience = patience
      self.min_delta = min_delta
      self.counter = 0
      self.best_loss = None
      self.early_stop = False
      self.just_started = False

  def __call__(self, val_loss):
      if self.best_loss == None:
          self.best_loss = val_loss
      elif self.best_loss - val_loss > self.min_delta:
          self.best_loss = val_loss
      elif self.best_loss - val_loss < self.min_delta:
          self.counter += 1
          self.just_started = True
          print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
          if self.counter >= self.patience:
              print('INFO: Early stopping')
              self.early_stop = True
      return self.early_stop, self.just_started

def plot_losses(dir, train_losses, valid_losses):
    if not os.path.exists(dir):
        os.mkdir(dir)

    x = list(range(len(train_losses)))
    plt.plot(x, train_losses, label='train loss')
    plt.plot(x, valid_losses, label='valid loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(dir, 'losses.png'))
    plt.close()

    losses = {
        'train_losses': train_losses,
        'valid_losses': valid_losses
    }
    with open(os.path.join(dir, 'losses_log'), 'wb') as handle:
        pkl.dump(losses, handle, protocol=pkl.HIGHEST_PROTOCOL)

def t_or_f(arg):
    ua = str(arg).upper()
    if 'TRUE'.startswith(ua):
       return True
    elif 'FALSE'.startswith(ua):
       return False

def parse_args():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  # Model hyperparameters
  parser.add_argument('--hidden_dim',
                      default=768,
                      type=int,
                      help='Size of hidden state of decoder model')
  parser.add_argument('--vocab_size',
                      default=30522,
                      type=int,
                      help='Size of the vocabulary')
  parser.add_argument('--teacher_forcing',
                      default=True,
                      type=t_or_f,
                      help='Use teacher forcing training')
  parser.add_argument('--use_attention',
                      default=True,
                      type=t_or_f,
                      help='Use attention in decoder')
  # Dataset hyperparameters
  parser.add_argument('--dataset_path',
                      default='./datasets/spider',
                      type=str,
                      help='Path to the spider dataset')
  parser.add_argument('--use_schema',
                      default=False,
                      type=t_or_f,
                      help='Add schema information to the input questions.')
  # Optimizer hyperparameters
  parser.add_argument('--lr',
                      default=0.0001,
                      type=float,
                      help='Learning rate to use')
  parser.add_argument('--batch_size',
                      default=4,
                      type=int,
                      help='Minibatch size')
  parser.add_argument('--effective_batch_size',
                      default=128,
                      type=int,
                      help='Minibatch size')
  # Other hyperparameters
  parser.add_argument('--epochs',
                      default=20,
                      type=int,
                      help='Max number of epochs')
  parser.add_argument('--seed', default=42, type=int, help='Random seed')
  parser.add_argument('--limit_train_batches',
                      default=1.0,
                      type=float,
                      help='Percentage of data to use for training')
  parser.add_argument('--refresh_rate',
                      default=10,
                      type=int,
                      help='Progress bar refresh rate')
  parser.add_argument('--log_dir', default='./logs', type=str)
  parser.add_argument('--verbose',
                      default=True,
                      type=t_or_f,
                      help='Generate 10 samples of output and target after each batch.')
  return parser.parse_args()
