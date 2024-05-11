from enum import Enum
import os,json
from source import folder_path
from transformers import AutoTokenizer,PreTrainedTokenizerBase
import pandas as pd
import random
from sklearn.model_selection import train_test_split
        
class causal_data:
    def __init__(self,tokenizer,block_size : int = 128,max_data_remember : int = 10) -> None:
        self.tokenizer : PreTrainedTokenizerBase = tokenizer
        self.block_size : int = block_size
        self.container : list = []
        self.max_remember = max_data_remember
    def add_data(self,text : str):
        while (len(self.container) == self.max_remember):
            self.container.pop(0)
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
    initialized = False
    raw_data_frame = pd.read_parquet(folder_path.data.mmlu_test)
    data_frame : pd.DataFrame = pd.DataFrame(columns=['Context','Question','Choices','Answer','Raw Choices'])
    @classmethod
    def _process_row(self,data_row : pd.Series):
        context = data_row['subject']
        question = data_row['question']
        choices = data_row['choices']
        options = f'{self.mapping[0]}: {choices[0]} {self.mapping[1]}: {choices[1]} {self.mapping[2]}: {choices[2]} {self.mapping[3]}: {choices[3]}'
        answer = f"{self.mapping[data_row['answer']]}: {choices[data_row['answer']]}"
        return [context,question,options,answer,choices]
    @classmethod
    def custom_init(self):
        if (self.initialized == True):
            return
        self.initialized = True
        for index,row in self.raw_data_frame.iterrows():
            self.data_frame.loc[index] = self._process_row(row)
    @classmethod
    def convert_to_data_point(self,row : pd.Series):
        result = {
            'Category Content' : None,
            'Question' : None,
            'Train' : None,
            'Answer' : None,
            'Category' : None,
        }
        result['Answer'] = row['Answer']
        result['Question'] = f'Question : ' + row['Question'] + '\nChoices : ' + row["Choices"] + '\nAnswer:'
        result['Category Content'] = row["Question"] + ' ' +'. '.join(row["Raw Choices"])
        result['Train'] = f'Question : ' + row['Question'] + '\nChoices : ' + row["Choices"] + '\nAnswer:' + row["Answer"]
        result['Category'] = row['Context']
        return result
    @classmethod
    def get_data(self,context = None,amount : int = 10,seed : int = 39):
        if (context != None):
            filtered_data = self.data_frame[self.data_frame['Context'] == context]
            dataframe = filtered_data
        else:
            dataframe = self.data_frame
        result = []
        sampled_datas = dataframe.sample(n=amount,random_state=seed)
        for i in range(amount):
            result.append(self.convert_to_data_point(sampled_datas.iloc[i]))
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
        