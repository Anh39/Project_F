import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import torch
import tensorflow as tf
torch.manual_seed(39)
from source import folder_path,data_processor,model_loader
from source.evaluator import causal_mmlu_eval
from source.data_processor import mmlu_category,causal_mmlu
import utility
import shutil

# gpt = model_loader.gemma2b(folder_path.output.gemma+'\\Train')
gpt = model_loader.gemma2b(folder_path.output.gemma+'\\Train')
evaltor = causal_mmlu_eval(gpt)
# causal_mmlu.custom_init()
# res = causal_mmlu.get_data(mmlu_category.elementary_mathematics,1,2,39)
# for qa in res:
#     i = 0
#     for q in qa['Question']:
#         i+=1
#         print(f'Question {i} :\n{q}')
#     print(qa['Answer'])
# print(evaltor.dual_random_evaluate(save=True,num=10,sample_amount=2,ran_range=[1,1000000]))
# print(evaltor.dual_random_evaluate_single_category(category=mmlu_category.abstract_algebra,save=True,num=1,sample_amount=2,ran_range=[1,1000000]))
# shutil.rmtree(os.path.join(folder_path.output.gemma,'Train'))
# print(evaltor.dual_random_evaluate_category(category=None,save=True,num=1,sample_amount=2,ran_range=[1,1000000]))
# shutil.rmtree(os.path.join(folder_path.output.gemma,'Train'))
print(evaltor.dual_random_evaluate_category_separete(category=mmlu_category.moral_scenarios,save=True,num=500,sample_amount=2,seed=39))

