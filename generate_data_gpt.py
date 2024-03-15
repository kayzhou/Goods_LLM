with open('/home/llm/liguanqun/毕设new/代码/gpt/初始对话.txt','r',encoding='utf-8') as f:
    # lines = f.readlines()[:100000]
    lines = f.readlines()

train_datas = []
temp_data = ''
for line in lines:

    if line!='\n':
        line = line.strip()
        temp_data+=(line+'\t')
    else:
        train_datas.append(temp_data)
        temp_data=''



with open('/home/llm/liguanqun/毕设new/代码/gpt/初始对话dataset.txt','w',encoding='utf-8') as f:
    for train_data in train_datas:
        f.write(train_data+'\n')

