# 导入csv模块
import csv
import pandas as pd
import json


df = pd.read_csv("/home/llm/liguanqun/毕设new/data/蚂蚁商联数据.csv", encoding='utf-8')
data = df.to_dict(orient="records")
dialogs = []
count = 0

# 跳过第一行，即表头
# next(data)
# f.open()
for item in data:  # 提取品牌和商品名以及规格
    count += 1
    cate_1 = item["cate_1"]
    cate_2 = item["cate_2"]
    cate_3 = item["cate_3"]
    cate_4 = item["cate_4"]
    barcode = item["barcode"]
    name = item["goods_name"]
    level = item["level"]
    label = item["label"]
    # brand = item["brand"]
    # goods_name = item["goods_name"]
    # goods_spec = item["goods_spec"]
    # 生成输入提示
    # input_prompt = f"品牌: {brand}\n品名: {goods_name}\n规格: {goods_spec}"
    input_prompt = f"商品编号: {barcode}商品名称: {name}"

    # 生成输出提示
    # output_prompt = f"输出为:\n一级分类: {item['ep_category_name_lv1']}\n二级分类: {item['ep_category_name_lv2']}\n三级分类: {item['ep_category_name_lv3']}\n四级分类: {item['ep_category_name_lv4']}\n"

    # 创建一个对话，包含user和assistant的角色和内容
    dialog = [
        {
            "inst": "",
            "role": "user1",
            "content": "请对下面的商品进行分析判断：商品编号: 6900000058312, 商品名称: BBQ白木柄烧烤麻花针12支装"
        },
        {
            "inst": "",
            "role": "assistant",
            "content": """商品分析结果为: {\"商品名称\": \"BBQ白木柄烧烤麻花针12支装\", \"商品编号\": \"6900000058312\", \"多维标注\": {\"原料\": \"麻\", \"连装\": \"12支装\", \"包装方式\": \"支装\"}}"""
        },
        {
            "inst": "",
            "role": "user2",
            "content": f"请按下面例子对商品进行分析，并填空:样例--输入为:{input_prompt}"
        }
    ]
    dialogs.append(dialog)

with open('/home/llm/liguanqun/毕设new/data/dialog_data.json', 'w', encoding='utf-8') as json_file:
    json.dump(dialogs, json_file, ensure_ascii=False, indent=4)

print("Dialogs saved to dialogs.json.")






# import pandas as pd
# import json
# from tqdm import tqdm

# # 读取CSV文件，仅处理前100行
# file_path = "/home/llm/liguanqun/毕设new/data/蚂蚁商联数据_未去重.csv"
# df = pd.read_csv(file_path)
# # df = df[:100]

# # 转换为想要的格式并保存到txt文件
# output_file_path = "/home/llm/liguanqun/毕设new/data/蚂蚁商联数据_json串.txt"
# with open(output_file_path, 'w', encoding='utf-8') as output_file:
#     result = {}
#     total_rows = len(df)
#     with tqdm(total=total_rows, desc='Processing Rows') as pbar:
#         for _, row in df.iterrows():
#             product_name = row["goods_name"]
#             barcode = str(row["barcode"])
#             category_1 = row["cate_1"]
#             category_2 = row["cate_2"]
#             category_3 = row["cate_3"]
#             category_4 = row["cate_4"]

#             if product_name not in result:
#                 result[product_name] = {
#                     "商品编号": barcode,
#                     "商品名称": product_name,
#                     "一级分类": category_1,
#                     "二级分类": category_2,
#                     "三级分类": category_3,
#                     "四级分类": category_4,
#                     "多维标注": {}
#                 }

#             result[product_name]["多维标注"][row["level"]] = row["label"]

#             pbar.update(1)  # 更新进度条

#     for product_info in result.values():
#         json.dump(product_info, output_file, ensure_ascii=False)
#         output_file.write('\n')

# print(f"前100行数据已保存到 {output_file_path}")
