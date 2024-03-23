from transformers import AutoTokenizer, AutoModelForCausalLM
from source import folder_path

tokenizer = AutoTokenizer.from_pretrained(folder_path.model.gemma)
model = AutoModelForCausalLM.from_pretrained(folder_path.model.gemma, device_map="auto")

input_text = "Hello, who are you ?"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids,max_new_tokens = 64)
print(tokenizer.decode(outputs[0]))