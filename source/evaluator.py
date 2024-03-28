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
        prompt = '\n'.join(test_data['Question'])
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
        ref = test_data['Answer'][:1]
        # print(pred,ref)
        return [pred,ref,end_time-start_time,start_mem_use,end_mem_use,end_mem_use-start_mem_use]
    def _evaluate_lora_single(self,test_data,k):
        train_data = '\n'.join(test_data['Question'][0:k-1])
        prompt = '\n'.join(test_data['Question'])
        device = torch.device("cuda")
        
        for i in range(self.repeat):
            for j in range(k-1):
                self.dp.add_data(test_data['Question'][j])
        train_data = self.dp.get_data()
        try:
            torch.cuda.reset_max_memory_allocated(device)
            start_train_mem_use = torch.cuda.max_memory_allocated(device)/(1024*1024)
            start_train_time = time.time()
            self.trainer.continue_train_gemma(self.model,train_dataset=train_data,epoch=1,lr=1e-3)
            end_train_time = time.time()
            end_train_mem_use = torch.cuda.max_memory_allocated(device)/(1024*1024)
        except:
            torch.cuda.reset_max_memory_allocated(device)
            start_train_mem_use = torch.cuda.max_memory_allocated(device)/(1024*1024)
            start_train_time = time.time()
            self.trainer.train_gemma(self.model,train_dataset=train_data,epoch=1,lr=1e-3)
            end_train_time = time.time()
            end_train_mem_use = torch.cuda.max_memory_allocated(device)/(1024*1024)
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
            self.dp.add_data(test_data['Question'][k-1] + '\nAnswer: ' + test_data['Answer'] )
        pred = result.split('Answer: ')[-1][:1]
        ref = test_data['Answer'][:1]
        print(pred,ref)
        return [pred,ref,end_time-start_time,start_mem_use,end_mem_use,end_mem_use-start_mem_use,end_train_time-start_train_time,
                start_train_mem_use,end_train_mem_use,end_train_mem_use-start_train_mem_use]
    def _dual_evaluate_single(self,category,seed,k):
        test_data = self.data_handler.get_single_data(category,seed=seed,k=k)
        base_res = self._evaluate_base_single(test_data,k)
        lora_res = self._evaluate_lora_single(test_data,k)
        base_res.append(category)
        lora_res.append(category)
        return (base_res,lora_res)
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
    def dual_random_evaluate_category(self,category = None,save : bool = False,num : int = 10,sample_amount : int = 5,ran_range : list = [1,1000000]):
        seed_list = []
        category_list = []
        for i in range(num):
            local_seed = random.randint(ran_range[0],ran_range[1])
            seed_list.append(local_seed)
            if (category == None):
                local_category = mmlu_category.get_random(local_seed)
                category_list.append(local_category)
            else:
                category_list.append(category)
        
        base_result = []
        lora_result = []
        
        for i in range(len(category_list)):
            try:
                base_res,lora_res = self._dual_evaluate_single(category_list[i],seed_list[i],sample_amount)
                base_result.append(base_res)
                lora_result.append(lora_res)
            except Exception as e:
                print(e)    
                
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
            content.append({
                'model_name' : self.model.name,
                'sample_amount' : sample_amount,
                'eval_amount' : len(base_result),
                'base_f1' : base_eval_result['f1'],
                'lora_f1' : lora_eval_result['f1']
            })
            for i in range(num):
                try:
                    line = {
                        'seed' : seed_list[i],
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
                    }
                    content.append(line)
                except Exception as e:
                    print(e)
            with open(folder_path.output.eval_log,'w') as file:
                file.write(json.dumps(content))
        return (base_eval_result,lora_eval_result)
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