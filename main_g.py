import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import utility
from source.model_loader import gpt2,gemma2b
import torch
import tensorflow as tf
from source import folder_path
from source.train import handler
from source.data_processor import causal_data

torch.manual_seed(39)



#t = handler()

#t.train_test()

#test = processor(gpt.model,[gpt])
gpt = gemma2b(folder_path.output.gemma + "\\Train")
print(folder_path.output.gemma + "\\Train")
# print(gpt.inference_base('Hello',32))
# print(gpt.inference_lora('Hello',32))
# print(gpt.inference_base('Hello',32))
# print(gpt.inference_lora('Hello',32))
# print(gpt.inference_base('Hello',32))
num = 11
trainer = handler()
dp = causal_data(gpt.tokenizer)
for i in range(num):
    dp.add_data('Hello, I am a box.')
gpt2_trainer = trainer.train_gemma(gpt,train_dataset=dp.get_data(),epoch=1)
gpt.unload_soft()
print(gpt.inference_lora('Hello',32))
gpt.unload_soft()
for i in range(num):
    dp.add_data('Hello, I am a boss.')
gpt2_trainer = trainer.train_gemma(gpt,train_dataset=dp.get_data(),epoch=1)
gpt.unload_soft()
print(gpt.inference_lora('Hello',32))
gpt.unload_soft()
for i in range(num):
    dp.add_data('Hello, I am a bee.')
gpt2_trainer = trainer.train_gemma(gpt,train_dataset=dp.get_data(),epoch=1)
gpt.unload_soft()
print(gpt.inference_lora('Hello',32))
gpt.unload_soft()
for i in range(num):
    dp.add_data('Hello, I am a cat.')
gpt2_trainer = trainer.train_gemma(gpt,train_dataset=dp.get_data(),epoch=1)
gpt.unload_soft()
print(gpt.inference_lora('Hello',32))


# gemma = gemma2b()


# print(gemma.inference('Hello',64))