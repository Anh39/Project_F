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
    gemma = os.path.join(_model_folder,'gemma2b')

class lora:
    # test = os.path.join(_lora_folder,'test')
    # q_test = os.path.join(_lora_folder,'q_test')
    gpt2 = os.path.join(_lora_folder,'gpt2')
    gemma = os.path.join(_lora_folder,'gemma')
        
class data:
    intents = os.path.join(_data_folder,'intents\data.txt')
    test_input = os.path.join(_data,'test_input\data.txt')
    
class raw_data:
    intents = os.path.join(_raw_data_folder,'intents.json')
    
class output:
    gpt2 = os.path.join(_output_folder,'gpt2')
    gemma = os.path.join(_output_folder,'gemma')
    # q_model = os.path.join(_output_folder,'q_model')
    # model = os.path.join(_output_folder,'model')
    # test = os.path.join(_output_folder,'test')
    
class continous:
    _continous_lora = os.path.join(_continous_model_folder,'lora')
    gp2 = os.path.join(_continous_lora,'q_lora')
    gemma = os.path.join(_continous_lora,'gemma')
    raw_data = os.path.join(_continous_model_folder,'cache\\raw_data.txt')
    data = os.path.join(_continous_model_folder,'cache\\data.txt')