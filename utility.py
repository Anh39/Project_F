import warnings

with open('warning.txt','w') as file:
    file.write(f'')
def catch_warning(message,category,filename,lineno,file=None,line=None):
    # print(message)
    with open('warning.txt','a') as file:
        file.write(f'Warning: {message} at {filename}:{lineno}\n')
warnings.showwarning = catch_warning
# warnings.filterwarnings("ignore")

