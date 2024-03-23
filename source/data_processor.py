from enum import Enum
import os,json
from source import folder_path
from transformers import AutoTokenizer,PreTrainedTokenizerBase

class data_tyle(Enum):
    intents = 'intents.json'
class data_processor:
    def __init__(self) -> None:
        pass
    def process(self,input_path) -> list:
        file_name = os.path.basename(input_path)
        if (file_name == 'intents.json'):
            return self._data_intents()
    def _data_intents(self) -> list:
        with open(folder_path.raw_data.intents,'r') as file:
            raw_data = json.load(file)
        data = []
        for intent in raw_data['intents']:
            for patternn in intent['patterns']:
                data.append(f'User: {patternn}\n')
                for response in intent['responses']:
                    data.append(f'Assistant: {response}\n')
        return data
        
class gpt2_data:
    def __init__(self) -> None:
        self.processor = data_processor()
    def _preprocess(self,input_path) -> list:
        return self.processor.process(input_path)
    def _save(self,data,output_path):
        with open(output_path,'w') as file:
            data = ''.join(data)
            file.write(data)
    def process(self,type : data_tyle) -> str:
        if (type == data_tyle.intents):
            data = self._preprocess(folder_path.raw_data.intents)
            self._save(data,folder_path.data.intents)
            return folder_path.data.intents
        
class causal_data:
    def __init__(self,tokenizer,block_size : int = 128) -> None:
        self.tokenizer : PreTrainedTokenizerBase = tokenizer
        self.block_size : int = block_size
        self.container : list = []
    def add_data(self,user : str,asisstant : str):
        self.container.append(user+'.'+asisstant)
    def add_data(self,text : str):
        self.container.append(text)
    def empy(self):
        self.container = []
    def get_data(self,empty : bool = True):
        result = []
        tokenized_container = []
        for ele in self.container:
            tokenized_container.append(self.tokenizer(ele))
        for ele in tokenized_container:
            if (len(ele)//self.block_size < 1):
                result.append(ele)
            else:
                it = 0
                while (len(ele)//self.block_size > it):
                    result.append(self.tokenizer(ele[it*self.block_size:(it+1)*self.block_size]))
                    it+=1
        # for i in range(len(result)):
        #     while (len(result[i]['input_ids']) < self.block_size):
        #         result[i]['input_ids'].append(self.tokenizer.eos_token_id)
        #     while (len(result[i]['attention_mask']) < self.block_size):
        #         result[i]['attention_mask'].append(1)
        if (empty):
            self.empy()
        return result