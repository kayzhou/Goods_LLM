import re
import csv
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 读取已处理的最新商品索引
def load_last_processed_index(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            last_index = int(file.read().strip())
        return last_index
    return -1

# 保存最新的商品索引
def save_last_processed_index(filename, index):
    with open(filename, 'w') as file:
        file.write(str(index))

# 读取 CSV 文件，提取商品名称和商品编号
def read_csv(filename):
    products = []
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            product = {
                "商品名称": row["goods_name"],
                "商品编号": row["barcode"]
            }
            products.append(product)
    return products

# 生成 chat 中第二个 user 的 content 格式
def generate_user_content(product):
    return f"商品编号: {product['商品编号']}, 商品名称: {product['商品名称']}"

# 初始化模型和tokenizer
def initialize_model_and_tokenizer():
    model_id = "/home/llm/liguanqun/model/gemma-2b-it"
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda:0",
        torch_dtype=dtype,
    )
    return tokenizer, model

# 进行对话
def chat_with_model(tokenizer, model, chat, index):
    responses = []
    
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=1024, temperature=0.2,top_p=0.85,top_k=200,do_sample=True)
    generated_response = tokenizer.decode(outputs[0])
    print(f"\n dialog {index}:")
    print(generated_response)
    # responses.append(generated_response)
    
    # 从模型生成的响应中提取商品分析结果
    # result = generated_response.split("<end_of_turn>")[-2].split(":")[-1].strip()
    result = generated_response.split("<end_of_turn>")[-1].split("商品分析结果为: ")[-1].split("<eos>")[0].strip()
    print(result)
    
    return responses, result

# 主函数
def main():
    csv_filename = "/home/llm/liguanqun/毕设new/data/蚂蚁商联数据.csv"
    output_filename = "/home/llm/liguanqun/毕设new/代码/gemma/商品分析结果.txt"
    processed_index_file = "/home/llm/liguanqun/毕设new/代码/gemma/last_processed_index.txt"

    tokenizer, model = initialize_model_and_tokenizer()
    products = read_csv(csv_filename)
    last_processed_index = load_last_processed_index(processed_index_file)

    index = last_processed_index + 1
    for i in range(last_processed_index + 1, len(products)):
        product = products[i]
        chat = [
            { "role": "user", "content": "请对下面的商品进行分析判断：商品编号: 6900000058312, 商品名称: BBQ白木柄烧烤麻花针12支装" },
            { "role": "assistant", "content": "商品分析结果为: {\"商品名称\": \"BBQ白木柄烧烤麻花针12支装\", \"商品编号\": \"6900000058312\", \"多维标注\": {\"原料\": \"麻\", \"连装\": \"12支装\", \"包装方式\": \"支装\"}}"}
        ]
        user_content = generate_user_content(product)
        chat.append({"role": "user", "content": user_content})

        responses, result = chat_with_model(tokenizer, model, chat, index)
        
        try:
            if re.match(r'{"商品名称": "(.*?)", "商品编号": "(.*?)", "多维标注": {(.*?)}}$', result):
                with open(output_filename, 'a', encoding='utf-8') as output_file:
                    # 将商品分析结果写入txt文件
                    output_file.write(result + '\n')
                save_last_processed_index(processed_index_file, i)  # 保存最新的商品索引
            else:
                raise ValueError(f"Result doesn't match the expected format: {result}")
        except ValueError as e:
            save_last_processed_index(processed_index_file, i)  # 保存最新的商品索引
            print(f"Error in dialog {index}: {e}")
        
        index += 1

if __name__ == "__main__":
    main()
