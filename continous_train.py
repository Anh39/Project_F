from datasets import load_dataset
import folder_path as path
from train import handler

class processor:
    def __init__(self,model,update_model : list = []) -> None:
        self.container = []
        self.model = model
        self.handler = handler()
        self.trainer = self.handler.get_trainer_for_gpt2(self.model)
        #self.lora = None
        self.update_model = update_model
        self.epoch = 8
        self.lr = 1e-3
        print('Continous Init',next(self.model.get_base_model().parameters()).device)
    def add_chat(self,chat_data : list):
        data = []
        data.append(f'User: {chat_data[0]}\n')
        for i in range(1,range(data)):
            data.append(f'Assistant: {chat_data[i]}\n')
        self.container.append(data)
    def get_chat(self,id):
        if (id >= 0 and id < len(self.container)):
            return self.container[id]
        return None
    def _train(self,raw_data):
        model = self.handler.wrap(self.model.get_base_model(),self.handler.get_lora_config_continous(4))
        self.trainer.model = model
        self.handler.train_model(self.trainer,raw_data,self.epoch,self.lr)
        model.save_pretrained(path.continous.q_lora)
        self.model = model
        for ele in self.update_model:
            ele.model = model
        print(f'Saved at {path.continous.q_lora}')




