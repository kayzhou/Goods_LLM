from openai import OpenAI
client = OpenAI()

# # 创建文件并上传微调数据
# file_response = client.files.create(
#     file = open("/Users/lgq/Documents/大模型/毕设/代码/代码/openaigpt/train_data.jsonl", "rb"),
#     purpose = "fine-tune",
# )

# # 获取文件的ID
# file_id = file_response.id
# print(file_id)

# # 启动微调任务
# client.fine_tuning.jobs.create(
#     training_file=file_id,  # 传入文件ID而不是文件对象
#     model="gpt-3.5-turbo",
#     suffix = "800data"
# )



print(client.fine_tuning.jobs.list(limit=1))


# file_id = "file-lo8ZIyReS02dxLY13LciAsc4"
# completion = client.chat.completions.create(
#   model="ft:gpt-3.5-turbo-0125:personal:800data:91cDxJCY",
#   messages=[
#     {"role": "system", "content": "你是一个知识丰富的人工智能助手，用户将用中文向你提问，你将根据你的知识用中文来如实回答问题"},
#     {"role": "user", "content": "请对下面的商品进行分析判断：商品编号: 6901306669189, 商品名称: 阿香婆香辣牛肉酱孜然200g"}
#   ]
# )
# print(completion.choices[0].message)