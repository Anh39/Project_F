import torch
from peft import PeftModel,LoraConfig,TaskType,PeftConfig,PeftType
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
import time
import folder_path as path

class loader:
    def __init__(self) -> None:
        pass


class gpt2:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(path.model.gpt2)
        self.model = AutoModelForCausalLM.from_pretrained(path.model.gpt2)
        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False
            )
        self.model = PeftModel(model=self.model,peft_config=self.peft_config)
        self.model.unload()
        self.use_lora = False
    def inference(self,input,max_token : int = 32):
        inputs = self.tokenizer(input,return_tensors="pt")
        if (self.use_lora == False):
            start_time = time.time()
            self.model.load_adapter(path.lora.test,'test')
            self.model.set_adapter('test')
            end_time = time.time()
            print(f'Elapsed time : {end_time-start_time}s')
            self.use_lora = True
        model_outputs = self.model.generate(**inputs,max_new_tokens = max_token)
        generated_tokens_ids = model_outputs.numpy()[0]
        result = self.tokenizer.decode(generated_tokens_ids)
        return result
    def inference_base(self,input,max_token : int = 32):
        inputs = self.tokenizer(input,return_tensors="pt")
        if (self.use_lora == True):
            self.model.delete_adapter('test')
            self.use_lora = False
        model_outputs = self.model.generate(**inputs,max_new_tokens = max_token)
        generated_tokens_ids = model_outputs.numpy()[0]
        result = self.tokenizer.decode(generated_tokens_ids)
        return result

#model.load_adapter('output_dir',adapters_name)
#model.set_adapter(adapters_name)
#model.unload()
#model.merge_adapter()
