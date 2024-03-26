import torch
import evaluate
from source import folder_path,data_processor,model_loader
from source.model_loader import causal_model
from source.data_processor import mmlu_category,causal_data
from source.train import handler
import random
import json
import time

class causal_mmlu_eval:
    def __init__(self,model : causal_model) -> None:
        self.data_handler = data_processor.causal_mmlu()
        self.data_handler.custom_init()
        self.model = model
        self.f1_eval = evaluate.load('f1')
        self.accuracy_eval = evaluate.load('accuracy')
        self.repeat = 4
        self.trainer  = handler()
        self.dp = causal_data(tokenizer=self.model.tokenizer,block_size=1024)
    def _evaluate_base_single(self,test_data,k):
        prompt = '\n'.join(test_data[0]['Question'])
        device = torch.device("cuda")
        
        self.model.unload_hard()
        start_time = time.time()

        torch.cuda.reset_max_memory_allocated(device)
        start_mem_use = torch.cuda.max_memory_allocated(device)/(1024*1024)
        # print('Base inference')
        # print('Bases: ',self.model.inference_base("Context: ",max_token=256))
        result = self.model.inference_base(prompt,max_token=4)
        end_mem_use = torch.cuda.max_memory_allocated(device)/(1024*1024)

        end_time = time.time()
        # print(result)
        pred = result.split('Answer: ')[-1][:1]
        ref = test_data[0]['Answer'][:1]
        # print(pred,ref)
        return [pred,ref,end_time-start_time,start_mem_use,end_mem_use,end_mem_use-start_mem_use]
    def _evaluate_lora_single2(self,test_data,k):
        
        train_data = '\n'.join(test_data[0]['Question'][0:k-1])
        prompt = '\n'.join(test_data[0]['Question'])
        device = torch.device("cuda")
        
        for i in range(self.repeat):
            for j in range(k-1):
                self.dp.add_data(test_data[0]['Question'][j])
        train_data = self.dp.get_data()
        try:
            self.trainer.continue_train_gemma(self.model,train_dataset=train_data,epoch=2,lr=1e-3)
        except:
            self.trainer.train_gemma(self.model,train_dataset=train_data,epoch=2,lr=1e-3)
        start_time = time.time()
        
        torch.cuda.reset_max_memory_allocated(device)
        start_mem_use = torch.cuda.max_memory_allocated(device)/(1024*1024)
        #print('Lora inference')
        # print('Loras: ',self.model.inference_lora("Context: ",max_token=256))
        result = self.model.inference_lora(prompt,max_token=4)
        end_mem_use = torch.cuda.max_memory_allocated(device)/(1024*1024)
        end_time = time.time()  
        #print(result)
        for i in range(self.repeat):
            self.dp.add_data(test_data[0]['Question'][k-1] + '\nAnswer: ' + test_data[0]['Answer'] )
        pred = result.split('Answer: ')[-1][:1]
        ref = test_data[0]['Answer'][:1]
        print(pred,ref)
        return [pred,ref,end_time-start_time,start_mem_use,end_mem_use,end_mem_use-start_mem_use]
    def _dual_evaluate_single(self,category,seed,k):
        test_data = self.data_handler.get_data(category,seed=seed,k=k)
        base_res = self._evaluate_base_single(test_data,k)
        lora_res = self._evaluate_lora_single2(test_data,k)
        return (base_res,lora_res)
    def dual_evalute(self,seeds : list = [39],categories : list = [mmlu_category.anatomy],k:int =5):
        base_preds = []
        base_refs = []
        base_inference_times = []
        base_memory_allocated = []
        lora_preds = []
        lora_refs = []
        lora_inference_times = []
        lora_memory_allocated = []
        for i in range(len(categories)):
            try:
                base_res,lora_res = self._dual_evaluate_single(categories[i],seeds[i],k)
                base_preds.append(self.data_handler.reversed_mapping[base_res[0]])
                base_refs.append(self.data_handler.reversed_mapping[base_res[1]])
                base_inference_times.append(base_res[2])
                base_memory_allocated.append(base_res[3:6])
                lora_preds.append(self.data_handler.reversed_mapping[lora_res[0]])
                lora_refs.append(self.data_handler.reversed_mapping[lora_res[1]])
                lora_inference_times.append(lora_res[2])
                lora_memory_allocated.append(lora_res[3:6])
            except Exception as e:
                print(e)
        base_f1_result = self.f1_eval.compute(
            predictions=base_preds,
            references=base_refs,
            average='micro'
        )
        base_accuracy_result = self.accuracy_eval.compute(
            predictions=base_preds,
            references=base_refs
        )
        lora_f1_result = self.f1_eval.compute(
            predictions=lora_preds,
            references=lora_refs,
            average='micro'
        )
        lora_accuracy_result = self.accuracy_eval.compute(
            predictions=lora_preds,
            references=lora_refs
        )
        base_f1_result.update(base_accuracy_result)
        lora_f1_result.update(lora_accuracy_result)
        return ((base_f1_result,base_preds,base_refs,base_inference_times,base_memory_allocated),
                (lora_f1_result,lora_preds,lora_refs,lora_inference_times,lora_memory_allocated))
    def dual_random_evaluate(self,save : bool = False,num : int = 10,sample_amount : int = 5,ran_range : list = [1,100]):
        seed_list = []
        category_list = []
        for i in range(num):
            local_seed = random.randint(ran_range[0],ran_range[1])
            category = mmlu_category.get_random(local_seed)
            seed_list.append(local_seed)
            category_list.append(category)
        (base_result,lora_result) = self.dual_evalute(seed_list,category_list,k=sample_amount)
        if (save):
            content = []
            content.append({
                'model_name' : self.model.name
            })
            for i in range(num):
                try:
                    line = {
                        'seed' : seed_list[i],
                        'category' : category_list[i],
                        'sample_amount' : sample_amount,
                        'base_predict' : base_result[1][i],
                        'base_answer' : base_result[2][i],
                        'base_time' : base_result[3][i],
                        'base_initial_memory' : base_result[4][i][0],
                        'base_end_memory' : base_result[4][i][1],
                        'base_inference_memory' : base_result[4][i][2],
                        'lora_predict' : lora_result[1][i],
                        'lora_answer' : lora_result[2][i],
                        'lora_time' : lora_result[3][i],
                        'lora_initial_memory' : lora_result[4][i][0],
                        'lora_end_memory' : lora_result[4][i][1],
                        'lora_inference_memory' : lora_result[4][i][2],
                    }
                    content.append(line)
                except Exception as e:
                    print(e)
            with open(folder_path.output.eval_log,'w') as file:
                file.write(json.dumps(content))
        return (base_result[0],lora_result[0])
    def dual_random_evaluate_single_category(self,category,save : bool = False,num : int = 10,sample_amount : int = 5,ran_range : list = [1,100]):
        seed_list = []
        category_list = []
        for i in range(num):
            local_seed = random.randint(ran_range[0],ran_range[1])
            seed_list.append(local_seed)
            category_list.append(category)
        (base_result,lora_result) = self.dual_evalute(seed_list,category_list,k=sample_amount)
        if (save):
            content = []
            content.append({
                'model_name' : self.model.name
            })
            for i in range(num):
                try:
                    line = {
                        'seed' : seed_list[i],
                        'category' : category_list[i],
                        'sample_amount' : sample_amount,
                        'base_predict' : base_result[1][i],
                        'base_answer' : base_result[2][i],
                        'base_time' : base_result[3][i],
                        'base_initial_memory' : base_result[4][i][0],
                        'base_end_memory' : base_result[4][i][1],
                        'base_inference_memory' : base_result[4][i][2],
                        'lora_predict' : lora_result[1][i],
                        'lora_answer' : lora_result[2][i],
                        'lora_time' : lora_result[3][i],
                        'lora_initial_memory' : lora_result[4][i][0],
                        'lora_end_memory' : lora_result[4][i][1],
                        'lora_inference_memory' : lora_result[4][i][2],
                    }
                    content.append(line)
                except Exception as e:
                    print(e)
            with open(folder_path.output.eval_log,'w') as file:
                file.write(json.dumps(content))
        return (base_result[0],lora_result[0])