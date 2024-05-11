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

seed = 399884 # random.randint(1,999999)
# with open('seed.txt','a') as file:
#     file.write( str(seed)+'\n'  )
torch.manual_seed(seed)

model = causal_lora_model(folder_path.model.gemma,folder_path.output.gemma.path,lam=2,
                          data_threshold=4,seed=seed,max_data_remember=4)

dt_point = {
        'Category Content' : 'Which of the following is true :\nThis is not 0. This is false. This is 1. This is 2',
        'Question' : 'Which of the following is true :\nA. This is not 0. B. This is false. C. This is 1. D. This is 2.\nAnswer: ',
        'Train' : 'Which of the following is true :\nA. This is not 0. B. This is false. C. This is 1. D. This is 2.\nAnswer: A. This is not 0',
        'Answer' : 'A',
        'Category' : 'Question'
    }
ds = []
for i in range(100):
    ds.append(dt_point)
train_data,test_data = train_test_split(ds,test_size=0.1,random_state=seed)
evaluator = causal_mmlu_eval(model,train_dataset=train_data,eval_dataset=test_data,step=9,
                             use_selector=False,fixed_adapter='logic')
evaluator.lr = 0.001
result = evaluator.evaluate_and_train(start_from=0,end_to=180)
print(len(model.loaded_loras))
