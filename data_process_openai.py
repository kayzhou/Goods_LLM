from sklearn.model_selection import train_test_split

# 读取待微调的数据文件
with open("/Users/lgq/Documents/大模型/毕设/代码/代码/openaigpt/待微调数据.jsonl", "r") as f:
    data = f.readlines()[:1000]

# 数据划分
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# 将划分后的数据写入新的文件
with open("/Users/lgq/Documents/大模型/毕设/代码/代码/openaigpt/train_data.jsonl", "w") as f:
    f.writelines(train_data)

with open("/Users/lgq/Documents/大模型/毕设/代码/代码/openaigpt/test_data.jsonl", "w") as f:
    f.writelines(test_data)

