from source import folder_path
import time

from datasets import load_dataset
eli5 = load_dataset("eli5_category",split="train[:50]")

eli5 = eli5.train_test_split(test_size=0.2,shuffle=False)

#print(eli5["train"][0])

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(folder_path.model.gpt2)
eli5 = eli5.flatten()
#print(eli5["train"][0]['answers.text'])
def preprocess_function(input_datas):
    return tokenizer([" ".join(x) for x in input_datas['answers.text']])

tokenized_eli5 = eli5.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=eli5["train"].column_names
)

block_size = 128

def group_text(input_datas):
    concated_data = {k : sum(input_datas[k],[]) for k in input_datas.keys()}
    total_length = len(concated_data[list(input_datas.keys())[0]])
    print(total_length)
    if (total_length >= block_size):
        total_length = (total_length // block_size) * block_size
    result = {
        k : [t[i : i + block_size] for i in range(0,total_length,block_size)]
        for k,t in concated_data.items()
    }
    result['labels'] = result['input_ids'].copy()
    return result

#print(tokenized_eli5['train'][0])        
lm_dataset = tokenized_eli5.map(
    group_text,
    batched=True,
    num_proc=1
)

from transformers import DataCollatorForLanguageModeling

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)

from transformers import AutoModelForCausalLM,TrainingArguments,Trainer

model = AutoModelForCausalLM.from_pretrained(folder_path.model.gpt2)

from peft import LoraConfig,PeftModel,TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r = 4,
    lora_alpha = 32,
    lora_dropout = 0.1,
    target_modules = ["c_proj","c_attn","c_fc"]
    
)

model = PeftModel(model=model,peft_config=lora_config,adapter_name='test')

training_args = TrainingArguments(
    output_dir=folder_path.lora.gpt2,
    evaluation_strategy='epoch',
    learning_rate=1e-3,
    weight_decay=0.01,
    num_train_epochs=1,
    push_to_hub=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset['train'],
    eval_dataset=lm_dataset['test'],
    data_collator=data_collator
)
start_time = time.time()
trainer.train()
end_time = time.time()
print(f'Elapsed time : {end_time - start_time}s')

import math

eval_results = trainer.evaluate()
# print(f'Perplexity: {math.exp(eval_results['eval_loss']):.2f}')
print(eval_results)
prompt = "Somatic hypermutation allows the immune system to"

from transformers import pipeline

generator = pipeline('text-generation',model=folder_path.model.gpt2)
print(generator(prompt))

#print(lm_dataset['train'][0])       
