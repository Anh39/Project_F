import torch
from peft import PeftModel,LoraConfig,TaskType,PeftConfig,prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM,Trainer,TrainingArguments,DataCollatorForLanguageModeling, AutoTokenizer,BitsAndBytesConfig, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
import time,os
import source.folder_path as path
import uuid
import tensorflow as tf
from source.data_processor import causal_data
import re
import random

class causal_lora_model:
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
    def __init__(self,model_path,lora_path,name : str = 'causal',lam : float = None,max_data_remember : int = 10,data_threshold : int = 4,seed : int = 39) -> None:
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=4,
            target_modules=None
        )
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )   
        model = AutoModelForCausalLM.from_pretrained(model_path,quantization_config = nf4_config,device_map='auto')
        self.loaded_loras = {'Empty':{'Train data':causal_data(self.tokenizer,block_size=1024)}}
        self.model : PeftModel = PeftModel(model=model,peft_config=self.peft_config,adapter_name='Empty')
        self.model.print_trainable_parameters()
        self.lora_path = lora_path
        self.max_remember = 1000000
        self._add_new_lora('Selector')
        self.lam = lam
        self.max_remember = max_data_remember
        self.data_threshold = data_threshold
        self.seed = seed
    def _selector_inference(self,input,max_token : int = None,gen_stragery : str = 'greedy'):
        inputs = self.tokenizer(input,return_tensors="pt").to("cuda") # to gpu
        model_outputs = self.model.generate(**inputs,**self.strageris[gen_stragery],max_new_tokens = max_token,return_dict_in_generate=True,output_logits=True)
        generated_tokens_ids=model_outputs['sequences'][0]
        i=len(model_outputs['logits'])-1
        j=len(generated_tokens_ids)-1
        result = []
        while(i>=0 and j>=0):
            logits = model_outputs['logits'][i][0]
            percent = torch.softmax(logits,dim=0)[generated_tokens_ids[j]]
            text = self.tokenizer.decode(generated_tokens_ids[j],skip_special_tokens=True)
            i-=1
            j-=1
            result.append((percent.item(),text))
        result.reverse()
        return result
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
    def _extract_category(self,input):
        if (self._switch_lora('Selector') == True):
            prompt = f'What subject category does **Content** belong to ?\n**Content** : "{input}".\nThe only category of **Content** is'
            result = self._selector_inference(prompt,max_token=16)
            i = 0
            first_layer = []
            try:
                first_layer_label = ''
                while ('\n' in result[i][1] or '*' in result[i][1] or ':' in result[i][1]):
                    i+=1
                while ('\n' not in result[i][1] and '*' not in result[i][1] and ':' not in result[i][1]):
                    if (self.lam == None):
                        ele = (result[i][0],self._normalize_string(result[i][1]))
                    else:
                        
                        dup = int(self.lam/result[i][0])
                        if (dup > 0):
                            dup = 1
                        print('Dup {}'.format(dup))
                        ele = (result[i][0],self._normalize_string(result[i][1]),dup)
                    first_layer_label += result[i][1]
                    first_layer.append(ele)
                    i+=1
                first_layer_label = self._normalize_string(first_layer_label)
            except Exception as e:
                print('adpter name ::::::::::::::::::::::::::::::::::::::::::::::::: \n',first_layer)
                print('PROMPTTTTTTTTTTTTTTTT :\n',prompt)
                print('RESSSSSSSSSSSSSSSSSSSSSSSSSSS :\n',result)
                print(e)
            return (first_layer_label,first_layer)
        else:
            print('Selector load error')
        return (None,None)
    def auto_inference(self,content : str = None,max_token : int = None,category : str = None,category_content : str =None):
        if (category != None):
            first_layer_label = category
        else:
            first_layer_label = self._extract_category(category_content)[0]
        if (first_layer_label != None):
            map_result = self._relative_mapping(first_layer_label)
            if map_result != None:
                print(f'Category : {map_result} found.')
                self._switch_lora(map_result)
            elif(self._load_lora(first_layer_label) == True):
                print(f'Category : {first_layer_label} loaded.')
                self._switch_lora(first_layer_label)
            else:
                print(f'Category : {first_layer_label} not found, use Base inference')
                if (self._switch_lora('Empty') == False):
                    print('Failed to switch to Base Inference')
        return self._inference(content,max_token=max_token)
    def __unload_soft(self):
        self.loaded_loras = {'Empty':'Empty'}
        self._add_new_lora('Selector')
        self.model.set_adapter(self.loaded_loras['Empty'])
        torch.cuda.empty_cache()
    def __unload_hard(self):
        # self.__unload_soft()
        self.model.adpter.unload()
    def _load_lora(self,lora_name):
        if (lora_name not in self.loaded_loras and os.path.exists(os.path.join(self.lora_path,lora_name))):
            start_time = time.time()
            self.model.load_adapter(os.path.join(self.lora_path,lora_name),lora_name,is_trainable=False)
            self.loaded_loras[lora_name] = {'Train data':causal_data(self.tokenizer,block_size=1024,max_data_remember=self.max_remember)}
            end_time = time.time()
            print(f'Adapter load time : {end_time-start_time}s')
            return True
        else:
            return False
    def _add_new_lora(self,lora_name):
        try:
            if (lora_name not in self.loaded_loras):
                self.loaded_loras[lora_name] = {'Train data':causal_data(self.tokenizer,block_size=1024,max_data_remember=self.max_remember)}
                self.model.add_adapter(lora_name,peft_config=self.peft_config)
                self.model.set_adapter(lora_name)
                return True
            else:
                return False
        except Exception as e:
            print(e)
            return False
    def _get_lora_names(self):
        result = []
        for ele in self.loaded_loras:
            result.append(ele)
        return result
    def _switch_lora(self,lora_name):
        if (lora_name in self.loaded_loras):
            # self.__unload_hard()
            # print(self.model.active_adapters)
            self.model.set_adapter(lora_name)

            return True
        else:
            return False
    def _normalize_string(self,input_str : str):
        return re.sub(r'[^A-Za-z_]','',input_str.strip().replace(' ','_').replace(':','').replace('*','').replace('.','').replace('<strong>','').replace('"','').replace('</strong>','').lower())
    def _relative_mapping(self,lora_name):
        normalized_lora_name = self._normalize_string(lora_name).split('_')
        for ele in self.loaded_loras:
            normalized_ele = self._normalize_string(ele)
            for n_lora_name_unit in normalized_lora_name:
                if (normalized_ele in n_lora_name_unit or n_lora_name_unit in normalized_ele):
                    return ele
    def selector(self,content):
        first_layer_label,first_layer = self._extract_category(content)
        if (first_layer_label != None):
            # prompt = f'What subject category does **Content** belong to ?\n**Content** : "{content}".\nThe only category of **Content** is'
            train_data = 'The category of "{}" is {}'.format(content,first_layer) #+ f'\n[Category] : {first_layer_label}'
            if (self.lam != None):
                for i in range(first_layer[0][2]):
                    self.add_train_data(train_data,'Selector',False)
            else:
                self.add_train_data(train_data,'Selector',False)
            map_result = self._relative_mapping(first_layer_label)
            if map_result != None:
                print(f'Category : {map_result} found.')
                self._switch_lora(map_result)
                first_layer_label = map_result
            else:
                # lora_names = ['"'+first_layer_label+'"']
                # for ele in self.loaded_loras:
                #     if (ele == 'Empty' or ele == 'Selector'):
                #         continue
                #     lora_names.append('"'+ele+'"')
                # if (len(lora_names) > 1):
                #     prompt = f'Group {", ".join(lora_names)} to {len(lora_names)-1} categories.'
                #     result = self.inference(prompt,max_token=64)
                #     print('Prompt:\n',prompt)
                #     print('Result:\n',result)
                
                if (self._load_lora(first_layer_label) == False):
                    print(f'Category : {first_layer_label} added.')
                    self._add_new_lora(first_layer_label)
                else:
                    print(f'Category : {first_layer_label} loaded.')
            return first_layer_label
        print('Select failed')
    def train_all(self,lr,epoch : int = 1,weight_decay : float = 0.01):
        for ele in self.loaded_loras:
            if ele not in ('Selector','Empty'):
                dataset = self.loaded_loras[ele]['Train data'].get_data(empty=False)
                self._split_train(dataset,ele,lr,epoch,weight_decay)
                # torch.cuda.empty_cache()
    def _split_train(self,dataset,adapter_name,lr,epoch : int = 1,weight_decay : float = 0.01):
        datasets = [[]]
        i=0
        # random.seed(self.seed)
        # random.shuffle(dataset)
        # random.seed(None)
        while (i < len(dataset)):
            last_dataset = datasets[len(datasets)-1]
            if (len(last_dataset) < self.data_threshold):
                last_dataset.append(dataset[i])
                i+=1
            else:
                datasets.append([])
        while(len(datasets)>0):
            dataset = datasets.pop()
            if (self.__continue_train(dataset,adapter_name,lr,epoch,weight_decay) == False):
                print(f'FATAL Train {adapter_name} error')
    def train_selector(self,lr,epoch : int = 1,weight_decay : float = 0.01):
        dataset = self.loaded_loras['Selector']['Train data'].get_data()
        self._split_train(dataset,'Selector',lr/5,epoch,weight_decay)
    def __continue_train(self,train_dataset,adapter_name,
                            lr,epoch : int = 1,weight_decay : float = 0.01):
        if (self._switch_lora(adapter_name) != True):
            return False
        collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer,mlm=False)
        train_args = TrainingArguments(
            output_dir=path.output.gemma.path,
            learning_rate=lr,
            num_train_epochs=epoch,
            weight_decay=weight_decay,
            evaluation_strategy="epoch",
            save_strategy="no",
        )
        # self.model.base_model = prepare_model_for_kbit_training(self.model.base_model)
        trainer = Trainer(
            model = self.model,
            tokenizer = self.tokenizer,
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
            data_collator=collator
        )
        print('LR : {}'.format(lr))
        start_time = time.time()
        try:
            trainer.train()
            end_time = time.time()
            self.model.save_pretrained(path.output.gemma.path,selected_adapters=self.model.active_adapters)
            print(f'Training time : {end_time-start_time}s')
        except Exception as e:
            print(e)
            return False
        return True
    def add_train_data(self,data,category : str = None,use_selector : bool = True):
        if (use_selector == True):
            category = self.selector(category)
        else:
            if (category not in ('Selector','Empty')):
                category = self._normalize_string(category)
            if (self._load_lora(category) == False):
                self._add_new_lora(category)
                
        self.loaded_loras[category]['Train data'].add_data(data)
