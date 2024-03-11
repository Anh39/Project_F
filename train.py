from transformers import AutoModelForCausalLM, OPTForCausalLM, AutoTokenizer,AutoModelForSeq2SeqLM,TrainingArguments,Trainer,DataCollatorForLanguageModeling,TextDataset
from peft import LoraConfig,TaskType,get_peft_model
from datasets import load_dataset
import json
import folder_path as path
import time
import os
from enum import Enum

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
        with open(path.raw_data.intents,'r') as file:
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
            file.write(','.join(data))
    def process(self,type : data_tyle):
        if (type == data_tyle.intents):
            data = self._preprocess(path.raw_data.intents)
            self._save(data,path.data.intents)
            return path.data.intents
class handler:
    def __init__(self) -> None:
        pass
    def get_lora_config(self,
                        task_type : TaskType = TaskType.CAUSAL_LM, inference_mode : bool = False,
                        r : int = 4, lora_alpha : int = 32, lora_dropout : float = 0.1
                        ) -> LoraConfig:
        return LoraConfig(
            task_type=task_type,
            inference_mode=inference_mode,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
    def get_training_args(self,
                          output_dir : int = path.output.model,learning_rate : float = 1e-3,
                          per_device_eval_batch_size : int = 32, per_device_train_batch_size : int = 32,
                          num_train_epochs : int = 16, weight_decay : float = 0.1,
                          evaluation_stragery = 'no', save_stragery = 'epoch'
                          ) -> TrainingArguments:
        return TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            per_device_eval_batch_size=per_device_eval_batch_size,
            per_device_train_batch_size=per_device_train_batch_size,
            num_train_epochs=num_train_epochs,
            weight_decay=weight_decay,
            evaluation_strategy=evaluation_stragery,
            save_strategy=save_stragery
        )
    def get_text_data_set(self,tokenizer,file_path,block_size,overwrite_cache : bool = False,cache_dir : str | None = None) -> TextDataset:
        return TextDataset(
            tokenizer=tokenizer,
            file_path=file_path,
            block_size=block_size,
            overwrite_cache=overwrite_cache,
            cache_dir=cache_dir
        )
    def get_trainer(self,
                    model,train_args,train_dataset,tokenizer,collator
                    ) -> Trainer:
        return Trainer(
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator = collator
        )
    def train_test(self):
        print(f'Start training')
        start_time = time.time()
        peft_config = self.get_lora_config()
        model = AutoModelForCausalLM.from_pretrained(path.model.gpt2)
        model = get_peft_model(model,peft_config)
        tokenizer = AutoTokenizer.from_pretrained(path.model.gpt2)
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)
        train_args = self.get_training_args()
        if (not os.path.exists(path.data.intents)):
            processor = gpt2_data()
            processor.process(type=data_tyle.intents)
        train_ds = self.get_text_data_set(tokenizer,path.data.intents,64)
        trainer = self.get_trainer(model,train_args,train_ds,tokenizer,collator)
        trainer.train()
        model.save_pretrained(path.output.model)
        end_time = time.time()
        print(f'Elapsed time : {end_time-start_time}s')
        print(f'Saved at {path.output.model}')


