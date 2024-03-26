import torch
from peft import PeftModel,LoraConfig,TaskType,PeftConfig,PeftType
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
import time,os
import source.folder_path as path
import uuid

class causal_model:
    strageris = {
        'greedy' : {},
        'beam_search' : {'do_sample':True,'num_beams':4},
        'contrastive' : {'penalty_alpha':0.6,'top_k':4},
        'multinomial_sampling' : {'do_sample':True,'num_beams':1},
        'beam_search_decoding' : {'num_beams':4},
        'diverse_beam_seach_decoding' : {'num_beams':4,'num_beam_groups':4,'diversity_penalty':1.0},
        'n_grams' : {'num_beams':4,'no_repeat_ngram_size':2},
        'top_k' : {'do_sample':True,'top_k':16},
        'top_p' : {'do_sample':True,'top_p':0.75,'top_k':0},
        'top_pk' : {'do_sample':True,'top_k':16,'top_p':0.75},
        'contrastive_using' : {'do_sample':True,'penalty_alpha':0.5,'top_k':16}
    }
    def __init__(self,model_path,lora_path,name : str = 'causal') -> None:
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        model = AutoModelForCausalLM.from_pretrained(model_path,quantization_config = nf4_config,device_map='auto')
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=1,
            target_modules=None
        )
        lora_name_unique = 'Empty' + str(uuid.uuid4())
        self.model : PeftModel = PeftModel(model=model,peft_config=peft_config,adapter_name=lora_name_unique)
        self.model.unload()
        self.model.print_trainable_parameters()
        self.use_lora = False
        self.lora_path = lora_path
        self.loaded_loras = {'Empty':lora_name_unique}
    def _inference(self,input,max_token : int = None,gen_stragery : str = 'greedy'):
        start_time = time.time()
        inputs = self.tokenizer(input,return_tensors="pt").to("cuda") # to gpu
        model_outputs = self.model.generate(**inputs,**self.strageris[gen_stragery],max_new_tokens = max_token)
        generated_tokens_ids = model_outputs.cpu().numpy()[0] # to cpu
        
        result = self.tokenizer.decode(generated_tokens_ids,skip_special_tokens=True)
        end_time = time.time()
        self.model.print_trainable_parameters()
        print(f'Inference time : {end_time-start_time}s')
        return result
    def add_lora(self,lora_path,lora_name):
        if (lora_name not in self.loaded_loras):
            lora_name_unique = lora_name + str(uuid.uuid4())
            self.loaded_loras[lora_name] = lora_name_unique
            start_time = time.time()
            self.model.load_adapter(lora_path,lora_name_unique,is_trainable=False)
            end_time = time.time()
            print(f'Adapter load time : {end_time-start_time}s')
    def unload_soft(self):
        lora_name_unique = self.loaded_loras['Empty']
        self.loaded_loras = {'Empty':lora_name_unique}
        self.model.set_adapter(self.loaded_loras['Empty'])
        self.use_lora = False
        torch.cuda.empty_cache()
    def unload_hard(self):
        self.unload_soft()
        self.model.unload()
    def inference_lora(self,input,max_token : int = None):
        if (self.use_lora == False):
            self.add_lora(self.lora_path,'Current')
            self.model.set_adapter(self.loaded_loras['Current'])
            self.use_lora = True
        return (self._inference(input,max_token))
    def prepare_for_continue_train(self):
        self.add_lora(self.lora_path,'Current')
        self.model.set_adapter(self.loaded_loras['Current'])
        return self.model.active_adapter
    def inference_base(self,input,max_token : int = None):
        if (self.use_lora == True):
            self.model.set_adapter(self.loaded_loras['Empty'])
        return (self._inference(input,max_token))
    
class gpt2(causal_model):
    def __init__(self, lora_path) -> None:
        super().__init__(path.model.gpt2, lora_path,name='gpt2')
        
class gemma2b(causal_model):
    def __init__(self, lora_path) -> None:
        super().__init__(path.model.gemma, lora_path,name='gemma2b')
        
class llama7b(causal_model):
    def __init__(self,lora_path) -> None:
        super().__init__(path.model.llama7b, lora_path,name='llama7b')
