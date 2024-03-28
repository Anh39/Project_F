from enum import Enum
import os,json
from source import folder_path
from transformers import AutoTokenizer,PreTrainedTokenizerBase
import pandas as pd
import random
from sklearn.model_selection import train_test_split

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
    

class causal_mmlu:
    mapping = {
        0 : 'A',
        1 : 'B',
        2 : 'C',
        3 : 'D'
        }
    reversed_mapping = {
        'A' : 0,
        'B' : 1,
        'C' : 2,
        'D' : 3
    }
    raw_data_frame = pd.read_parquet(folder_path.data.mmlu_test)
    data_frame : pd.DataFrame = pd.DataFrame(columns=['Content','Context'])
    @classmethod
    def _process_row(self,data_row : pd.Series):
        context = data_row['subject']
        question = data_row['question']
        choices = data_row['choices']
        options = f'{self.mapping[0]}: {choices[0]} {self.mapping[1]}: {choices[1]} {self.mapping[2]}: {choices[2]} {self.mapping[3]}: {choices[3]}'
        answer = f'{self.mapping[data_row['answer']]}: {choices[data_row['answer']]}'
        result = f'Context: {context}\nQuestion: {question}\nOptions: {options}\nAnswer: {answer}'
        return [result,context]
    @classmethod
    def custom_init(self):
        for index,row in self.raw_data_frame.iterrows():
            self.data_frame.loc[index] = self._process_row(row)
    @classmethod
    def _get_single_data(self,dataframe : pd.DataFrame,k : int = 5, seed : int = 39):
        result = []
        sampled_data = dataframe.sample(n=k,random_state=seed)
        for index,row in sampled_data.iterrows():
            result.append(row['Content'])
        last_ele = result[len(result)-1]
        last_ele = last_ele.split('\nAnswer:')
        answer = last_ele[1][1:]
        last_ele = last_ele[0]
        result[len(result)-1] = last_ele
        return {
            'Question' : result,
            'Answer' : answer
        }
    @classmethod
    def get_single_data(self,context : str,k : int = 5,seed : int = 39):
        filtered_data = self.data_frame[self.data_frame['Context'] == context]
        result = self._get_single_data(filtered_data,k=k,seed=seed)
        return result
    @classmethod
    def get_data(self,context,amount : int = 10,k : int = 5,seed : int = 39):
        filtered_data = self.data_frame[self.data_frame['Context'] == context]
        dataframe = filtered_data
        result = []
        sampled_datas = dataframe.sample(n=amount*k,random_state=seed)
        total_it = 0
        while len(result)<amount:
            it = 0
            question = []
            while (it<k):
                total_it += 1
                it+=1
                question.append(sampled_datas.iloc[total_it-1]['Content'])
            last_ele = question[len(question)-1]
            last_ele = last_ele.split('\nAnswer:')
            answer = last_ele[1][1:]
            last_ele = last_ele[0]
            question[len(question)-1] = last_ele
            qa = {
                'Question' : question,
                'Answer' : answer,
                'Context' : context
            }
            result.append(qa)
        return result
    
class mmlu_category:
    abstract_algebra = "abstract_algebra"
    anatomy = "anatomy"
    astronomy = "astronomy"
    business_ethics = "business_ethics"
    clinical_knowledge = "clinical_knowledge"
    college_biology = "college_biology"
    college_chemistry = "college_chemistry"
    college_computer_science = "college_computer_science"
    college_mathematics = "college_mathematics"
    college_medicine = "college_medicine"
    college_physics = "college_physics"
    computer_security = "computer_security"
    conceptual_physics = "conceptual_physics"
    econometrics = "econometrics"
    electrical_engineering = "electrical_engineering"
    elementary_mathematics = "elementary_mathematics"
    formal_logic = "formal_logic"
    global_facts = "global_facts"
    high_school_biology = "high_school_biology"
    high_school_chemistry = "high_school_chemistry"
    high_school_computer_science = "high_school_computer_science"
    high_school_european_history = "high_school_european_history"
    high_school_geography = "high_school_geography"
    high_school_government_and_politics = "high_school_government_and_politics"
    high_school_macroeconomics = "high_school_macroeconomics"
    high_school_mathematics = "high_school_mathematics"
    high_school_microeconomics = "high_school_microeconomics"
    high_school_physics = "high_school_physics"
    high_school_psychology = "high_school_psychology"
    high_school_statistics = "high_school_statistics"
    high_school_us_history = "high_school_us_history"
    high_school_world_history = "high_school_world_history"
    human_aging = "human_aging"
    human_sexuality = "human_sexuality"
    international_law = "international_law"
    jurisprudence = "jurisprudence"
    logical_fallacies = "logical_fallacies"
    machine_learning = "machine_learning"
    management = "management"
    marketing = "marketing"
    medical_genetics = "medical_genetics"
    miscellaneous = "miscellaneous"
    moral_disputes = "moral_disputes"
    moral_scenarios = "moral_scenarios"
    nutrition = "nutrition"
    philosophy = "philosophy"
    prehistory = "prehistory"
    professional_accounting = "professional_accounting"
    professional_law = "professional_law"
    professional_medicine = "professional_medicine"
    professional_psychology = "professional_psychology",
    public_relations = "public_relations"
    security_studies = "security_studies"
    sociology = "sociology"
    us_foreign_policy = "us_foreign_policy"
    virology = "virology"
    world_religions = "world_religions"
    total = [abstract_algebra,anatomy,astronomy,business_ethics,clinical_knowledge,
             college_biology,college_chemistry,college_computer_science,college_mathematics,college_medicine,college_physics,
             computer_security,conceptual_physics,econometrics,electrical_engineering,elementary_mathematics,
             formal_logic,global_facts,
             high_school_biology,high_school_chemistry,high_school_computer_science,high_school_european_history,
             high_school_geography,high_school_government_and_politics,high_school_macroeconomics,high_school_mathematics,
             high_school_microeconomics,high_school_physics,high_school_statistics,high_school_us_history,high_school_world_history,
             human_aging,human_sexuality,jurisprudence,logical_fallacies,
             machine_learning,management,marketing,medical_genetics,moral_disputes,moral_scenarios,
             nutrition,philosophy,prehistory,professional_accounting,professional_law,professional_medicine,professional_psychology,
             public_relations,security_studies,sociology,us_foreign_policy,world_religions]
    @classmethod
    def get_random(self,seed = None):
        if (seed == None):
            random.seed()
        else:
            random.seed(seed)
        return random.choice(self.total)
        