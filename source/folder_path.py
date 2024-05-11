import os,shutil
import datetime

_project_path = os.getcwd()
_model_folder = os.path.join(_project_path,'model')
_data = os.path.join(_project_path,'data')
_output_folder = os.path.join(_project_path,'output')
_logs_folder = os.path.join(_project_path,'logs')
class model:
    gpt2 = os.path.join(_model_folder,'gpt2')
    gemma = os.path.join(_model_folder,'gemma2b')
    llama7b = os.path.join(_model_folder,'llama7b')
    
class data:
    mmlu = os.path.join(_data,'mmlu')
    mmlu_aux = os.path.join(mmlu,'auxiliary_train-00000-of-00001.parquet')
    mmlu_dv = os.path.join(mmlu,'dev-00000-of-00001.parquet')
    mmlu_test = os.path.join(mmlu,'test-00000-of-00001.parquet')
    mmlu_validate = os.path.join(mmlu,'validation-00000-of-00001.parquet')
    
class output:
    class gpt2:
        _name = 'gpt2'
        path = os.path.join(_output_folder,_name)
        current_train = os.path.join(path,'Train')
        info = os.path.join(path,'info')
        eval_log = os.path.join(path,'eval_log.json')
        data = os.path.join(path,'data.json')
        readme = os.path.join(path,'README.md')
        @classmethod
        def save_record(self):
            current_datetime = datetime.datetime.now()
            formated_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
            save_path = os.path.join(_logs_folder,self._name + formated_datetime)
            shutil.copytree(self.info,os.path.join(save_path,'info'))
            shutil.copytree(self.current_train,os.path.join(save_path,'Train'))
            shutil.copy(self.readme,os.path.join(save_path,'README.md'))
    class gemma:
        _name = 'gemma'
        path = os.path.join(_output_folder,_name)
        current_train = os.path.join(path,'Train')
        info = os.path.join(path,'info')
        eval_log = os.path.join(path,'eval_log.json')
        data = os.path.join(path,'data.json')
        readme = os.path.join(path,'README.md')
        @classmethod
        def save_record(self):
            current_datetime = datetime.datetime.now()
            formated_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
            save_path = os.path.join(_logs_folder,self._name + formated_datetime)
            shutil.copytree(self.info,os.path.join(save_path,'info'))
            shutil.copytree(self.current_train,os.path.join(save_path,'Train'))
            shutil.copy(self.readme,os.path.join(save_path,'README.md'))
    class llma7b:
        _name = 'llama7b'
        path = os.path.join(_output_folder,_name)
        current_train = os.path.join(path,'Train')
        info = os.path.join(path,'info')
        eval_log = os.path.join(path,'eval_log.json')
        data = os.path.join(path,'data.json')
        readme = os.path.join(path,'README.md')
        @classmethod
        def save_record(self):
            current_datetime = datetime.datetime.now()
            formated_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
            save_path = os.path.join(_logs_folder,self._name + formated_datetime)
            shutil.copytree(self.info,os.path.join(save_path,'info'))
            shutil.copytree(self.current_train,os.path.join(save_path,'Train'))
            shutil.copy(self.readme,os.path.join(save_path,'README.md'))

    
    