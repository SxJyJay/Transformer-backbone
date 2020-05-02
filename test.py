import torch
import torch.nn as nn
import dill as pickle
from tensorboardX import SummaryWriter

data = pickle.load(open("bpe_vocab.pkl", 'rb'))
print(data)