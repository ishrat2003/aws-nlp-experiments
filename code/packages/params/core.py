import argparse
import os
import logging
import json

class Core:

  def __init__(self):
    self.parser = argparse.ArgumentParser()
    self.parser.add_argument('--display_details', default=False, type=bool)
    self.parser.add_argument('--display_network', default=True, type=bool)
    self.parser.add_argument('--dataset_name', default='newsroom')
    self.parser.add_argument('--dataset_percentage', default=100, help = "Percentage of dataset for processing")
    self.parser.add_argument('--total_items', default=0, help = "Total items of data for processing")
    self.parser.add_argument('--data_directory', default='/content/drive/My Drive/Colab Notebooks/data')
    
    self.parser.add_argument('--source_max_sequence_length', default=64, type=int) # Content length
    self.parser.add_argument('--target_max_sequence_length', default=192, type=int) # Summary length
    
    self.parser.add_argument('--epochs', default=20, type=int)
    self.parser.add_argument('--buffer_size', default=20000, type=int)
    self.parser.add_argument('--batch_size', default=64, type=int)
    self.parser.add_argument('--dimensions', default=128, type=int)
    self.parser.add_argument('--num_layers', default=4, type=int)
    self.parser.add_argument('--num_heads', default=8, type=int)
    self.parser.add_argument('--d_model', default=128, type=int)
    self.parser.add_argument('--dff', default=512, type=int, help = "Positive integer, dimensionality of the output space for FF network")
    self.parser.add_argument('--dropout_rate', default=0.1, type=float)

    self.parser.add_argument('--plot_directory', default='/content/drive/My Drive/Colab Notebooks/data/plots')
    
    # CWR
    self.parser.add_argument('--position_contributing_factor', default=1.0, type=float)
    self.parser.add_argument('--occurance_contributing_factor', default=1.0, type=float)
    self.parser.add_argument('--type', default='title', help="title|summary|lda")
    self.parser.add_argument('--word',type=str)
    self.parser.add_argument('--mode', default='none', help="none|context|masked_context")
    
    return

  def get(self):
    return self.parser.parse_args()

  def save(self, path):
    if not os.path.exists(path):
      os.makedirs(path)

    params = json.dumps(vars(self.get()))
    path = os.path.join(path, self._getFileName())
    with open(path, 'w') as fileToProcess:
        fileToProcess.write(params)

    logging.info("# Params saved in " + path)
    return

  def load(self, path):
    if not os.path.isdir(path):
        path = os.path.dirname(path)

    path = os.path.join(path, self._getFileName())
    fileContent = open(path, 'r').read()
    flag2val = json.loads(fileContent)
    for flag, value in flag2val.items():
        self.parser.flag = value

  def _getFileName(self):
    return type(self).__name__ + "params"