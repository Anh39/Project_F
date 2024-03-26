import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import torch
import tensorflow as tf
torch.manual_seed(39)
from source import folder_path,data_processor,model_loader
from source.evaluator import causal_mmlu_eval
from source.data_processor import mmlu_category
import utility

# gpt = model_loader.gemma2b(folder_path.output.gemma+'\\Train')
gpt = model_loader.gemma2b(folder_path.output.gemma+'\\Train')
evaltor = causal_mmlu_eval(gpt)
# print(evaltor.dual_random_evaluate(save=True,num=10,sample_amount=2,ran_range=[1,1000000]))
print(evaltor.dual_random_evaluate_single_category(category=mmlu_category.abstract_algebra,save=True,num=10,sample_amount=2,ran_range=[1,1000000]))