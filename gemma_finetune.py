import json
from tqdm import tqdm
from datasets import load_dataset

# Convert dataset to OAI messages
system_message = """你是一个知识丰富的人工智能助手，用户将用中文向你提问，你将根据你的知识用中文来如实回答问题
"""

def create_conversation(sample):
    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["human_answers"][0]} # for whatever reason the dataset uses a list of answers
        ]
    }

# Load dataset from the hub
# dataset_dict = load_dataset("/home/llm/liguanqun/毕设new/代码/gemma/Hello-SimpleAI/HC3-Chinese", name="baike")
dataset_dict = load_dataset("json", data_files="/home/llm/liguanqun/毕设new/代码/gemma/商品formatted_products.jsonl")
dataset = dataset_dict['train']
print(dataset)


print(create_conversation(dataset[0]))

# # Convert dataset to OAI messages
dataset = dataset.map(create_conversation, batched=False)
dataset = dataset.train_test_split(test_size=0.2, random_state=42)

# save datasets to disk
dataset["train"].to_json("/home/llm/liguanqun/毕设new/代码/gemma/train_dataset.json", orient="records")
dataset["test"].to_json("/home/llm/liguanqun/毕设new/代码/gemma/test_dataset.json", orient="records")






# # Load dataset from local JSONL file
# dataset = []
# with open("/home/llm/liguanqun/毕设new/代码/gemma/商品formatted_products.jsonl", "r", encoding="utf-8") as file:
#     for line in tqdm(file, desc="Processing lines"):
#         try:
#             data = json.loads(line)
#             dataset.append(data)
#         except json.JSONDecodeError:
#             print("Error decoding JSON line, skipping...")

# print(create_conversation(dataset[0]))
# # Convert dataset to OAI messages
# dataset = list(map(create_conversation, dataset))


# train_size = int(len(dataset) * 0.8)
# train_dataset = dataset[:train_size]
# test_dataset = dataset[train_size:]

# # save datasets to disk
# with open("/home/llm/liguanqun/毕设new/代码/gemma/train_dataset.json", "w", encoding="utf-8") as file:
#     json.dump(train_dataset, file, ensure_ascii=False, indent=4)

# with open("/home/llm/liguanqun/毕设new/代码/gemma/test_dataset.json", "w", encoding="utf-8") as file:
#     json.dump(test_dataset, file, ensure_ascii=False, indent=4)
