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

gpt = gemma2b(folder_path.output.gemma + "\\Train")
print(folder_path.output.gemma + "\\Train")
num = 5
token = 64
dp = causal_data(gpt.tokenizer)
trainer = handler()
while(token != -1):
    token = int(input())
    prompt = str(input())
    result = gpt.inference_lora(prompt,token)
    print(result)
    is_ok = str(input())
    if (is_ok == 'ok'):
        for i in range(num):
            dp.add_data(prompt+"\n"+result)
        gpt_trainer = trainer.train_gemma(gpt,train_dataset=dp.get_data(),epoch=1)
        gpt.unload_soft()
    elif (is_ok == 'no'):
        result = str(input())
        for i in range(num):
            dp.add_data(prompt+"\n"+result)
        gpt_trainer = trainer.train_gemma(gpt,train_dataset=dp.get_data(),epoch=1)
        gpt.unload_soft()
