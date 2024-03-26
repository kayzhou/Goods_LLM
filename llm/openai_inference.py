import csv
import time
import re
from openai_inference import OpenAI

client = OpenAI()

# 函数：保存最新的商品索引
def save_last_processed_index(file_path, index):
    with open(file_path, 'w') as f:
        f.write(str(index))

# 文件路径和参数
csv_file_path = '/Users/lgq/Documents/大模型/毕设/代码/毕设/data/蚂蚁商联数据.csv'
output_filename = '/Users/lgq/Documents/大模型/毕设/代码/代码/openaigpt/analysis_results.txt'
processed_index_file = '/Users/lgq/Documents/大模型/毕设/代码/代码/openaigpt/processed_index.txt'

# 读取上次处理的商品索引
try:
    with open(processed_index_file, 'r') as f:
        processed_index = int(f.read())
except FileNotFoundError:
    processed_index = 0

# 循环调用接口并保存结果到txt文件
with open(output_filename, 'a', encoding='utf-8') as outfile:
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            if i < processed_index:
                continue  # 跳过已处理的行

            product = {
                '商品编号': row['barcode'],
                '商品名称': row['goods_name']
            }

            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是个超级人工智能，可以解决一切问题"},
                    {"role": "user", "content": "请对下面的商品进行分析判断: 商品编号: 6900000058312, 商品名称: BBQ白木柄烧烤麻花针12支装"},
                    {"role": "assistant", "content": "{\"商品名称\": \"BBQ白木柄烧烤麻花针12支装\", \"商品编号\": \"6900000058312\", \"多维标注\": {\"原料\": \"麻\", \"连装\": \"12支装\", \"包装方式\": \"支装\"}}"},
                    {"role": "user", "content": f"请对下面的商品进行分析判断: 商品编号: {product['商品编号']}, 商品名称: {product['商品名称']}"}
                ],
                temperature = 0.2,
                top_p = 0.85,
                max_tokens = 1024
            )
            
            # 提取并保存结果到txt文件
            result = completion.choices[0].message.content
            print(result)
            if re.match(r'{"商品名称": "(.*?)", "商品编号": "(.*?)", "多维标注": {(.*?)}}$', result):
                # 将商品分析结果写入txt文件
                outfile.write(result + '\n')
                outfile.flush()  # 立即刷新文件缓冲区，确保每次写入都能立即生效
                save_last_processed_index(processed_index_file, i)  # 保存最新的商品索引
            else:
                print(f"Error in dialog {i}: Result doesn't match the expected format: {result}")

            time.sleep(20)  # 控制调用速率，避免超过API调用限制
