import os

_project_path = os.getcwd()
_model_folder = os.path.join(_project_path,'model')
_lora_folder = os.path.join(_project_path,'lora')
_data = os.path.join(_project_path,'data')
_data_folder = os.path.join(_project_path,'data\processed')
_raw_data_folder = os.path.join(_project_path,'data\original')
_output_folder = os.path.join(_project_path,'output')
_continous_model_folder = os.path.join(_project_path,'continous_model')
class model:
    gpt2 = os.path.join(_model_folder,'gpt2')
    gemma_2b_q = os.path.join(_model_folder,'gemma_2b_q')

class lora:
    test = os.path.join(_lora_folder,'test')
    q_test = os.path.join(_lora_folder,'q_test')
        
class data:
    intents = os.path.join(_data_folder,'intents\data.txt')
    test_input = os.path.join(_data,'test_input\data.txt')
    
class raw_data:
    intents = os.path.join(_raw_data_folder,'intents.json')
    
class output:
    q_model = os.path.join(_output_folder,'q_model')
    model = os.path.join(_output_folder,'model')
    test = os.path.join(_output_folder,'test')
    
class continous:
    lora = os.path.join(_continous_model_folder,'lora')
    q_lora = os.path.join(_continous_model_folder,'q_lora')
    raw_data = os.path.join(_continous_model_folder,'cache\\raw_data.txt')
    data = os.path.join(_continous_model_folder,'cache\\data.txt')