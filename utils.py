import argparse

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
  # Dataset hyperparameters
  parser.add_argument('--dataset_path',
                      default='./data/spider',
                      type=str,
                      help='Path to the spider dataset')
  # Optimizer hyperparameters
  parser.add_argument('--lr',
                      default=0.001,
                      type=float,
                      help='Learning rate to use')
  parser.add_argument('--batch_size',
                      default=128,
                      type=int,
                      help='Minibatch size')
  # Other hyperparameters
  parser.add_argument('--epochs',
                      default=5,
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
  return parser.parse_args()
