import json
import tqdm

# 读取原始JSONL文件
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

# 处理数据并生成新的JSON列表
def process_data(input_data):
    processed_data = []
    for item in input_data:
        instruction = "你是一个电商行业的商品零售专家，擅长对商品进行消费群体的分类。具体来说，你能够根据商品名称和商品的中位数价格来划分商品所属的消费群体类别。"
        product_info = json.dumps({"商品名称": item["商品名称"], "商品中位数价格": item["商品中位数价格"]}, ensure_ascii=False)
        processed_item = {"instruction": instruction, "input": product_info}
        processed_data.append(processed_item)
    return processed_data

# 保存为新的JSON文件
def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# 主函数
def main():
    input_file = "/home/llm/liguanqun/案例大赛/data/consumer_2000.jsonl"
    output_file = "/home/llm/liguanqun/案例大赛/data/consumer_2000_test.json"

    # 读取原始JSONL数据
    data = read_jsonl(input_file)

    # 处理数据
    processed_data = process_data(data)

    # 保存为新的JSON文件
    save_json(processed_data, output_file)


if __name__ == "__main__":
    main()









# import json

# # 读取原始JSONL文件
# def read_jsonl(file_path, limit=None):
#     data = []
#     with open(file_path, 'r', encoding='utf-8') as file:
#         for idx, line in enumerate(file):
#             if limit is not None and idx >= limit:
#                 break
#             data.append(json.loads(line))
#     return data

# # 处理数据并生成新的JSON列表
# def process_data(input_data):
#     processed_data = []
#     for item in input_data:
#         instruction = "你是个擅长对商品进行分类的人工智能，请对商品进行分类"
#         product_info = json.dumps({"商品编号": item["商品编号"], "商品名称": item["商品名称"]}, ensure_ascii=False)
#         output = json.dumps({"一级分类": item["一级分类"], "二级分类": item["二级分类"], "三级分类": item["三级分类"]}, ensure_ascii=False)
#         processed_item = {"instruction": instruction, "input": product_info, "output": output}
#         processed_data.append(processed_item)
#     return processed_data

# # 保存为新的JSON文件
# def save_json(data, file_path):
#     with open(file_path, 'w', encoding='utf-8') as file:
#         json.dump(data, file, ensure_ascii=False)

# # 主函数
# def main():
#     input_file = "/home/llm/liguanqun/案例大赛/data/蚂蚁商联多标签.jsonl"  # 输入的JSONL文件路径
#     output_file = "/home/llm/liguanqun/llama3/data/蚂蚁商联多级分类llama3.json"  # 输出的JSONL文件路径

#     # 读取前一万行原始JSONL数据
#     input_data = read_jsonl(input_file, limit=10000)

#     # 处理数据
#     processed_data = process_data(input_data)

#     # 保存为新的JSONL文件
#     save_json(processed_data, output_file)

# if __name__ == "__main__":
#     main()