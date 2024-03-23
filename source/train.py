from transformers import AutoModelForCausalLM, OPTForCausalLM, AutoTokenizer,AutoModelForSeq2SeqLM,TrainingArguments,Trainer,DataCollatorForLanguageModeling,TextDataset
from transformers import BitsAndBytesConfig
from peft import LoraConfig,TaskType,PeftModel,get_peft_model
from peft import prepare_model_for_kbit_training
import source.folder_path as path
import time,os,json
from source.model_loader import causal_model

class handler:
    def __init__(self) -> None:
        pass
    def get_trainer(self,model,train_args,train_dataset,tokenizer,collator):
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=collator
        )
        return trainer
    def train_gpt2(self,causal_model : causal_model,train_dataset,adapter_name = 'Train',
                    lr : float = 1e-3,epoch : int = 1,weight_decay : float = 0.01,
                    r : int = 4):
        collator = DataCollatorForLanguageModeling(tokenizer=causal_model.tokenizer,mlm=False)
        train_args = TrainingArguments(
            output_dir=path.output.gpt2,
            learning_rate=lr,
            num_train_epochs=epoch,
            weight_decay=weight_decay,
            evaluation_strategy="epoch",
            save_strategy="no"
        )
        trainer = Trainer(
            model = causal_model.model,
            tokenizer = causal_model.tokenizer,
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
            data_collator=collator
        )
        peft_config = LoraConfig(
            task_type = TaskType.CAUSAL_LM,
            inference_mode = False,
            r = r,
            lora_alpha = 32,
            lora_dropout = 0.1,
            target_modules = ["c_fc","c_attn","c_proj"]
        )
        
        causal_model.model = prepare_model_for_kbit_training(causal_model.model)
        causal_model.model.add_adapter(adapter_name,peft_config)
        causal_model.model.set_adapter(adapter_name)
        
        start_time = time.time()
        trainer.train()
        end_time = time.time()
        causal_model.model.save_pretrained(path.output.gpt2,selected_adapters=['Train'])
        print(f'Training time : {end_time-start_time}s')
        causal_model.model.unload()
    def train_gemma(self,causal_model : causal_model,train_dataset,adapter_name = 'Train',
                    lr : float = 1e-3,epoch : int = 1,weight_decay : float = 0.01,
                    r : int = 4):
        collator = DataCollatorForLanguageModeling(tokenizer=causal_model.tokenizer,mlm=False)
        train_args = TrainingArguments(
            output_dir=path.output.gemma,
            learning_rate=lr,
            num_train_epochs=epoch,
            weight_decay=weight_decay,
            evaluation_strategy="epoch",
            save_strategy="no"
        )
        trainer = Trainer(
            model = causal_model.model,
            tokenizer = causal_model.tokenizer,
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
            data_collator=collator
        )
        peft_config = LoraConfig(
            task_type = TaskType.CAUSAL_LM,
            inference_mode = False,
            r = r,
            lora_alpha = 32,
            lora_dropout = 0.1,
            target_modules = ["q_proj","k_proj","down_proj","gate_proj","o_proj","up_proj","v_proj"]
        )
        
        causal_model.model = prepare_model_for_kbit_training(causal_model.model)
        causal_model.model.add_adapter(adapter_name,peft_config)
        causal_model.model.set_adapter(adapter_name)
        
        start_time = time.time()
        trainer.train()
        end_time = time.time()
        causal_model.model.save_pretrained(path.output.gemma,selected_adapters=['Train'])
        print(f'Training time : {end_time-start_time}s')
        causal_model.model.unload()

