import json
import random

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
        instruction = "你是个擅长对商品进行多标签分类的商品零售专家，请根据给出的各项商品文本信息对商品进行多标签分类"
        product_info = json.dumps({"商品编号": item["商品编号"], "商品名称": item["商品名称"]}, ensure_ascii=False)
        output = json.dumps(item["多维标注"], ensure_ascii=False)
        processed_item = {"instruction": instruction, "input": product_info, "output": output}
        processed_data.append(processed_item)
    return processed_data

# 保存为新的JSON文件
def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# 主函数
def main():
    # 设置随机种子
    random.seed(42)
    
    # input_file = "/home/llm/liguanqun/llama3/data/蚂蚁商联数据_json串.jsonl"  # 输入的JSONL文件路径
    input_file = "/home/llm/liguanqun/案例大赛/data/蚂蚁商联多标签_去除低频label.jsonl"
    train_output_file = "/home/llm/liguanqun/案例大赛/data/蚂蚁商联多维标注对话new_去除低频label_10000_train.json"
    # test_output_files = [f"/home/llm/liguanqun/案例大赛/data/蚂蚁商联多维标注对话_去除低频label_2000_test_{i}.json" for i in range(5)]

    # 读取原始JSONL数据
    data = read_jsonl(input_file)
    total_count = len(data)
    train_count = int(0.8 * total_count)
    test_count = total_count - train_count

    # 打乱数据
    random.shuffle(data)

    # 划分训练集和测试集
    train_data = data[:train_count]
    test_data = data[train_count:]

    # 从训练集中随机抽取10000条数据
    sampled_train_data = random.sample(train_data, 10000)
    processed_train_data = process_data(sampled_train_data)
    save_json(processed_train_data, train_output_file)

    # # 从测试集中随机抽取2000条数据，且抽取五次，每次互斥
    # for i in range(5):
    #     if len(test_data) < 2000:
    #         raise ValueError("测试集数据不足2000条，无法抽取5组互斥的测试集")
        
    #     sampled_test_data = random.sample(test_data, 2000)
    #     processed_test_data = process_data(sampled_test_data)
    #     save_json(processed_test_data, test_output_files[i])

    #     # 从测试集中移除已抽取的数据
    #     test_data = [item for item in test_data if item not in sampled_test_data]

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
#         instruction = "你是个擅长对商品进行多标签分类的商品零售专家，请根据给出的各项商品文本信息对商品进行多标签分类"
#         product_info = json.dumps({"商品编号": item["商品编号"], "商品名称": item["商品名称"]}, ensure_ascii=False)
#         output = json.dumps(item["多维标注"], ensure_ascii=False)
#         processed_item = {"instruction": instruction, "input": product_info, "output": output}
#         processed_data.append(processed_item)
#     return processed_data

# # 保存为新的JSON文件
# def save_json(data, file_path):
#     with open(file_path, 'w', encoding='utf-8') as file:
#         json.dump(data, file, ensure_ascii=False, indent=4)

# # 主函数
# def main():
#     # input_file = "/home/llm/liguanqun/llama3/data/蚂蚁商联数据_json串.jsonl"  # 输入的JSONL文件路径
#     input_file = "/home/llm/liguanqun/案例大赛/data/蚂蚁商联多标签_去除低频label.jsonl"
#     output_file = "/home/llm/liguanqun/案例大赛/data/蚂蚁商联多维标注对话_去除低频label.json"  # 输出的JSONL文件路径

#     # 读取前一万行原始JSONL数据
#     input_data = read_jsonl(input_file, limit=10000)

#     # 处理数据
#     processed_data = process_data(input_data)

#     # 保存为新的JSONL文件
#     save_json(processed_data, output_file)

# if __name__ == "__main__":
#     main()





