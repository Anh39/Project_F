import os,json
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import tensorflow as tf
from sklearn.model_selection import train_test_split
from source import folder_path
from source.model_loader import causal_lora_model
from source.data_processor import causal_mmlu,mmlu_category,causal_data
from source.evaluator import causal_mmlu_eval
import utility,random

seed = 399884
# seed = random.randint(1,999999)
# with open('seed.txt','a') as file:
#     file.write( str(seed)+'\n'  )
# torch.manual_seed(seed)

model = causal_lora_model(folder_path.model.gemma,folder_path.output.gemma.path,lam=0.25,
                          data_threshold=2,seed=seed,max_data_remember=5)

causal_mmlu.custom_init()
ds = causal_mmlu.get_data(amount=1000,seed=seed)
train_data,test_data = train_test_split(ds,test_size=0.1,random_state=seed)
evaluator = causal_mmlu_eval(model,train_dataset=train_data,eval_dataset=test_data,step=90,
                             use_selector=True,fixed_adapter=None)
evaluator.lr = 0.0001
result = evaluator.evaluate_and_train(start_from=0,end_to=900)
print(len(model.loaded_loras))
