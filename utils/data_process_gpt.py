import json
from tqdm import tqdm

# 读取txt文件，转换格式，并写入新的txt文件
def convert_txt(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # 使用tqdm添加进度条
        for line in tqdm(lines, desc="Processing", unit="lines"):
            try:
                # 提取商品名称和其他信息
                name, _, details = line.strip().split('\t')
                name = name.strip()
                details = details.strip()
                
                # 提取多维标注信息
                multi_dimension = {}
                for item in details.split(','):
                    key, value = item.split(':')
                    multi_dimension[key.strip()] = value.strip()
                
                # 构建最终格式
                formatted_data = {
                    "商品名称": name,
                    "多维标注": multi_dimension
                }
                
                # 将格式化数据写入文件
                json.dump(formatted_data, f, ensure_ascii=False)
                f.write('\n')
            except ValueError:
                print("Error processing line:", line)

# 调用函数并传入输入和输出文件路径
input_file = '/home/llm/liguanqun/毕设new/代码/gpt/商品名称output.txt'
output_file = '/home/llm/liguanqun/毕设new/代码/gpt/json格式模型结果.txt'
convert_txt(input_file, output_file)
