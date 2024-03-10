from transformers import AutoModelForCausalLM, OPTForCausalLM, AutoTokenizer,AutoModelForSeq2SeqLM,TrainingArguments,Trainer,DataCollatorForLanguageModeling,TextDataset
from peft import LoraConfig,TaskType,get_peft_model
from datasets import load_dataset
import json
import folder_path as path
import time

class data_processor: # haven't refactored
    def __init__(self) -> None:
        pass
    def preprocess(input_path = 'intents.json'):
        with open(input_path,'r') as file:
            data = json.load(file)
        preprocess_data = []
        for intent in data['intents']:
            for patternn in intent['patterns']:
                preprocess_data.append(f'User: {patternn}\n')
                for response in intent['responses']:
                    preprocess_data.append(f'Assistant: {response}\n')
        return ''.join(preprocess_data)
    def save_preprocess(data,output_path = 'data/data.txt'):
        with open(output_path,'w') as file:
            file.write(data)
    def aaa(self):
        data = self.preprocess()
        self.save_preprocess(data)
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
        train_ds = self.get_text_data_set(tokenizer,path.data.intents,64)
        trainer = self.get_trainer(model,train_args,train_ds,tokenizer,collator)
        trainer.train()
        model.save_pretrained(path.output.model)
        end_time = time.time()
        print(f'Elapsed time : {end_time-start_time}s')


