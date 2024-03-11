import os

_project_path = os.getcwd()
_model_folder = os.path.join(_project_path,'model')
_lora_folder = os.path.join(_project_path,'lora')
_data_folder = os.path.join(_project_path,'data\processed')
_raw_data_folder = os.path.join(_project_path,'data\original')
_output_folder = os.path.join(_project_path,'output')
class model:
    gpt2 = os.path.join(_model_folder,'gpt2')
    gemma_2b_q = os.path.join(_model_folder,'gemma_2b_q')

class lora:
    test = os.path.join(_lora_folder,'test')
    q_test = os.path.join(_lora_folder,'q_test')
        
class data:
    intents = os.path.join(_data_folder,'intents\data.txt')
    
class raw_data:
    intents = os.path.join(_raw_data_folder,'intents.json')
    
class output:
    q_model = os.path.join(_output_folder,'q_model')
    model = os.path.join(_output_folder,'model')
    test = os.path.join(_output_folder,'test')