from model_loader import gpt2

#t = handler()

#t.train_test()

gpt = gpt2()

print(gpt.inference_base('Hello'))