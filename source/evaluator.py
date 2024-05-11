import torch
from source import folder_path,data_processor,model_loader
from source.model_loader import causal_lora_model
from source.data_processor import mmlu_category,causal_data
import random
import json
import copy
import time

class causal_mmlu_eval_unit:
    def __init__(self,model : causal_lora_model,use_selector : bool = True) -> None:
        self.model = model
        self.use_selector = use_selector
        self.lr = 0.01
    def train(self):
        device = torch.device("cuda")
        try:
            torch.cuda.reset_max_memory_allocated(device)
            train_mem_init = torch.cuda.max_memory_allocated(device)/(1024*1024)
            train_time = time.time()
            self.model.train_all(epoch=1,lr=self.lr,weight_decay=0.01)
            train_time = time.time() - train_time
            train_mem_end = torch.cuda.max_memory_allocated(device)/(1024*1024)
            train_mem_use = train_mem_end - train_mem_init
        except Exception as e:
            print(e)
            train_mem_init = 'ERROR'
            train_mem_end = 'ERROR'
            train_mem_use = 'ERROR'
            train_time = 'ERROR'
        result = {
            'train_time' : train_time,
            'train_mem_init' : train_mem_init,
            'train_mem_use' : train_mem_use,
            'train_mem_end' : train_mem_end
        }
        return result
    def add_data(self,data,category_or_category_content):
        self.model.add_train_data(data,category_or_category_content,self.use_selector)
    def train_selector(self,epoch=1,weight_decay=0.01):
        self.model.train_selector(self.lr,epoch,weight_decay)
    def evaluate(self,prompt,answer,category,category_content):
        device = torch.device("cuda")
        try:
            torch.cuda.reset_max_memory_allocated(device)
            infer_mem_init = torch.cuda.max_memory_allocated(device)/(1024*1024)
            infer_time = time.time()
            result = self.model.auto_inference(prompt,max_token=1,category=category,category_content=category_content)
            infer_time = time.time() - infer_time 
            infer_mem_end = torch.cuda.max_memory_allocated(device)/(1024*1024)
            infer_mem_use = infer_mem_end - infer_mem_init
            pred = result.split('Answer: ')[-1][:1]
            ref = answer[:1]
        except Exception as e:
            print(e)
            infer_mem_init = 'ERROR'
            infer_mem_end = 'ERROR'
            infer_mem_use = 'ERROR'
            infer_time = 'ERROR'
        result = {
            'predict' : pred,
            'reference' : ref,
            'infer_time' : infer_time,
            'infer_mem_init' : infer_mem_init,
            'infer_mem_use' : infer_mem_use,
            'infer_mem_end' : infer_mem_end,
            'adapter' : self.model.model.active_adapter
        }
        return result

class causal_mmlu_eval:
    def __init__(self,model : causal_lora_model,train_dataset,eval_dataset,step : int = 10,use_selector : bool = True,fixed_adapter : str = None) -> None:
        self.unit = causal_mmlu_eval_unit(model,use_selector)
        self.step = step
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.use_selector = use_selector
        self.fixed_adapter = fixed_adapter
        self.lr = 0.01
        if (self.fixed_adapter != None):
            self.unit.model._add_new_lora(self.fixed_adapter)
    def _evaluate_all(self,use_category = False):
        result = []
        for data in self.eval_dataset:
            current_data = data
            # prompt = "Question : This is an example question ?\nChoices: A: answer 1. B: answer 2. C: answer 3. D: answer 4.\nAnswer : A: answer 1\n"
            prompt = "Select correct answer, A or B or C or D.\n"
            prompt += current_data['Question']
            if (self.fixed_adapter != None):
                temp_result = self.unit.evaluate(prompt,current_data['Answer'],category=self.fixed_adapter,category_content=current_data['Category Content'])
            elif (use_category):
                temp_result = self.unit.evaluate(prompt,current_data['Answer'],category=current_data['Category'],category_content=current_data['Category Content'])
            else:
                temp_result = self.unit.evaluate(prompt,current_data['Answer'],category=None,category_content=current_data['Category Content'])
            result.append(temp_result)
        return result
    def _train_stage(self,its : list):
        result = []
        for it in its:
            current_data = self.train_dataset[it]
            if (self.fixed_adapter != None):
                self.unit.add_data(current_data['Train'],self.fixed_adapter)
            elif (self.use_selector):
                self.unit.add_data(current_data['Train'],current_data['Category Content'])
            else:
                self.unit.add_data(current_data['Train'],current_data['Category'])
        result = self.unit.train()
        if (self.use_selector):
            self.unit.train_selector()
        return result
    def save_result(self,result):
        with open("result.json",'w') as file:
            file.write(json.dumps(result))
    def evaluate_and_train(self,start_from : int = 0,end_to : int = 0):
        self.unit.lr = self.lr
        result = []
        its = []
        e_res = []
        for i in range(start_from,min(end_to,len(self.train_dataset))):
            its.append(i)
            print(i,len(its),self.step)
            if (len(its) == self.step):
                try:
                    step_result = {
                        'Train' : self._train_stage(its),
                        'Evaluate' : self._evaluate_all()
                    }
                except Exception as e:
                    print(e)
                    e_res.append(e)
                    try:
                        with open('eres.txt','w') as file:
                            file.write(json.dumps(e_res))
                    except Exception as ee:
                        print(ee)
                result.append(step_result)
                its = []
            self.save_result(result)
        return result
    def _dual_evaluate_all(self,category,seed,k,amount):
        if (category == None):
            datas = []
            categoris = {}
            count = amount
            while(count > 0):
                cate = mmlu_category.get_random()
                if (cate in categoris):
                    categoris[cate] += 1
                else:
                    categoris[cate] = 1
                count -= 1
            for ele in categoris:
                local_amount = categoris[ele]
                local_data = self.data_handler.get_data(ele,amount=local_amount,k=k,seed=39)
                datas.extend(local_data)
            random.seed(seed)
            random.shuffle(datas)
            random.seed()
        else:
            datas = self.data_handler.get_data(category,amount=amount,k=k,seed=seed)
        base_results = []
        lora_results = []
        for data in datas:
            base_res = self._evaluate_base_single(data,k)
            lora_res = self._evaluate_lora_single(data,k)
            base_res.append(data['Context'])
            lora_res.append(data['Context'])
            base_results.append(base_res)
            lora_results.append(lora_res)
        if (category == None):
            final_res = [base_results,lora_results,categoris]
        else:
            final_res = [base_results,lora_results]
        return final_res

class causal_mmlu_eval_old:

    def _dual_evaluate_single(self,category,seed,k):
        test_data = self.data_handler.get_single_data(category,seed=seed,k=k)
        base_res = self._evaluate_base_single(test_data,k)
        lora_res = self._evaluate_lora_single(test_data,k)
        base_res.append(category)
        lora_res.append(category)
        return (base_res,lora_res)
    def dual_random_evaluate_category_separete(self,category = None,save : bool = False,num : int = 10,sample_amount : int = 5,seed : int = 39,ran_range : list = [1,1000000]):
        eval_res = self._dual_evaluate_all(category=category,seed=seed,k=sample_amount,amount=num)
        base_result = eval_res[0]
        lora_result = eval_res[1]
        if (len(eval_res) > 2):
            categoris = eval_res[2]
        base_preds = []
        base_refs = []
        lora_preds = []
        lora_refs = []
        for ele in base_result:
            base_preds.append(self.data_handler.reversed_mapping[ele[0]])
            base_refs.append(self.data_handler.reversed_mapping[ele[1]])
        for ele in lora_result:
            lora_preds.append(self.data_handler.reversed_mapping[ele[0]])
            lora_refs.append(self.data_handler.reversed_mapping[ele[1]])
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
        base_eval_result = base_f1_result
        lora_eval_result = lora_f1_result
        
        if (save):
            content = []
            header = {
                'model_name' : self.model.name,
                'seed' : seed,
                'sample_amount' : sample_amount,
                'eval_amount' : len(base_result),
                'base_f1' : base_eval_result['f1'],
                'lora_f1' : lora_eval_result['f1']
            }
            if (category == None):
                header['categoris'] = categoris
            content.append(header)
            for i in range(num):
                try:
                    line = {
                        'category' : base_result[i][len(base_result[i])-1],
                        'base_predict' : base_result[i][0],
                        'base_answer' : base_result[i][1],
                        'base_time' : base_result[i][2],
                        'base_initial_memory' : base_result[i][3],
                        'base_end_memory' : base_result[i][4],
                        'base_inference_memory' : base_result[i][5],
                        'lora_predict' : lora_result[i][0],
                        'lora_answer' : lora_result[i][1],
                        'lora_time' : lora_result[i][2],
                        'lora_initial_memory' : lora_result[i][3],
                        'lora_end_memory' : lora_result[i][4],
                        'lora_inference_memory' : lora_result[i][5],
                        'lora_train_time' : lora_result[i][6],
                        'lora_initial_train_memory' : lora_result[i][7],
                        'lora_end_train_memory' : lora_result[i][8],
                        'lora_train_memory' : lora_result[i][9]
                    }
                    content.append(line)
                except Exception as e:
                    print(e)
            with open(folder_path.output.eval_log_sep,'w') as file:
                file.write(json.dumps(content))
        return (base_eval_result,lora_eval_result)