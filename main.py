from model_loader import gpt2
from continous_train import processor
import torch
import tensorflow as tf



#t = handler()

#t.train_test()
gpt = gpt2()
#test = processor(gpt.model,[gpt])
print(gpt.inference('Hello'))