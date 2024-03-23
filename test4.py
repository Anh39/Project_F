from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from source import folder_path

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

tokenizer = AutoTokenizer.from_pretrained(folder_path.model.gemma)
model = AutoModelForCausalLM.from_pretrained(folder_path.model.gemma, quantization_config=quantization_config)

input_text = "Hello, who are you ?"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids,max_new_tokens = 64)
print(tokenizer.decode(outputs[0]))