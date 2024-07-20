from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import time


start_time = time.time()

model_id = "/home/llm/liguanqun/llama3/model/NousResearch-Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto"
)

# 函数：保存最新的商品索引
def save_last_processed_index(file_path, index):
    with open(file_path, 'w') as f:
        f.write(str(index))

test_label = 1
model_name = "llama3"
task = "duobiaoqian"
json_file_path = f"/home/llm/liguanqun/案例大赛/data/蚂蚁商联多维标注对话new_去除低频label_2000_test_{test_label}.json"
output_filename = f'/home/llm/liguanqun/案例大赛/results/{model_name}_{task}_analysis_results_test_{test_label}.jsonl'
processed_index_file = f'/home/llm/liguanqun/案例大赛/results/{model_name}_{task}_processed_index_test_{test_label}.txt'
# metrics_output_filename = f'/home/llm/liguanqun/llama3/data/{model_name}/{model_name}_{task}_metrics_results_test_{text_label}.json'

# 读取上次处理的商品索引
try:
    with open(processed_index_file, 'r') as f:
        processed_index = int(f.read())
except FileNotFoundError:
    processed_index = 0

# 存储真实标签和预测标签
true_labels = []
pred_labels = []

# 循环调用接口并保存结果到txt文件
with open(output_filename, 'a', encoding='utf-8') as outfile:
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        for i, row in enumerate(data):
            if i < processed_index:
                continue
            input_data = json.loads(row['input'])
            true_label = json.loads(row['output'])

            messages = [
                # {"role": "system", "content": row['instruction']},
                {"role": "system", "content": "你是一个电商行业的商品零售专家，擅长对商品打标签。具体来说，你能够从给出的商品编号和商品名称分析出商品所具有的各项标签属性（即商品的包含的level以及level的值label），从而对商品的各项标签进行分类。"},
                {
                    "role": "user",
                    "content": "请问我们已知的商品的level有哪些呢？"
                },
                {
                    "role": "assistant",
                    "content": f""" "已知的商品的level有：包装方式、口味、种类、原料、花纹、度数、
                                    含肉类型、配料、口感、颜色、适用人群、形态、功能、类型，    
                                    共14个level。"
                                    """
                },
                {
                    "role": "user",
                    "content": "请问level“包装方式”下包含哪些label呢？"
                },
                {
                    "role": "assistant",
                    "content": f""" "已知level“包装方式”下包含：包装、瓶装、支装、桶装、盒装、片装、
                                    套装、袋装、碗装、条装、个装、只装、卷装、筒装，该level下共14个label。" 
                                """
                },
                {
                    "role": "user",
                    "content": "请问level“口味”下包含哪些label呢？"
                },
                {
                    "role": "assistant",
                    "content": "已知level“口味”下包含：牛奶、原味、葡提，该level下共3个label。" 
                },
                {
                    "role": "user",
                    "content": "请问level“种类”下包含哪些label呢？"
                },
                {
                    "role": "assistant",
                    "content": f""" "已知level“种类”下包含：盘碟类、保温杯、本子、方便面、拖鞋、胶水/胶带、
                                    水产、中性笔、毛巾、肉类、平角裤、素食、碗具类、勺子类、船袜，该level下共15个label。" 
                                """
                },
                {
                    "role": "user",
                    "content": "请问level“原料”下包含哪些label呢？"
                },
                {
                    "role": "assistant",
                    "content": "已知level“原料”下包含：不锈钢、棉、竹制、木制、塑料，该level下共5个label。" 
                },
                {
                    "role": "user",
                    "content": "请问level“花纹”下包含哪些label呢？"
                },
                {
                    "role": "assistant",
                    "content": "已知level“花纹”下包含：印花，该level下共1个label。" 
                },
                {
                    "role": "user",
                    "content": "请问level“度数”下包含哪些label呢？"
                },
                {
                    "role": "assistant",
                    "content": "已知level“度数”下包含：0-3度，该level下共1个label。" 
                },
                {
                    "role": "user",
                    "content": "请问level“含肉类型”下包含哪些label呢？"
                },
                {
                    "role": "assistant",
                    "content": "已知level“含肉类型”下包含：鸡肉、牛肉、鱼肉，该level下共3个label。" 
                },
                {
                    "role": "user",
                    "content": "请问level“口感”下包含哪些label呢？"
                },
                {
                    "role": "assistant",
                    "content": "已知level“口感”下包含：酥脆、酸、麻辣、鲜、香辣，该level下共5个label。" 
                },
                {
                    "role": "user",
                    "content": "请问level“颜色”下包含哪些label呢？"
                },
                {
                    "role": "assistant",
                    "content": "已知level“颜色”下包含：蓝、彩色、黑，该level下共3个label。" 
                },
                {
                    "role": "user",
                    "content": "请问level“适用人群”下包含哪些label呢？"
                },
                {
                    "role": "assistant",
                    "content": "已知level“适用人群”下包含：男士、儿童、婴儿、女士，该level下共4个label。" 
                },
                {
                    "role": "user",
                    "content": "请问level“形态”下包含哪些label呢？"
                },
                {
                    "role": "assistant",
                    "content": "已知level“形态”下包含：片、方形、圆形，该level下共3个label。" 
                },
                {
                    "role": "user",
                    "content": "请问level“功能”下包含哪些label呢？"
                },
                {
                    "role": "assistant",
                    "content": "已知level“功能”下包含：补水保湿、抑菌，该level下共2个label。" 
                },
                {
                    "role": "user",
                    "content": "请问level“类型”下包含哪些label呢？"
                },
                {
                    "role": "assistant",
                    "content": "已知level“类型”下包含：面部护肤、中性，该level下共2个label。" 
                },
                {
                    "role": "user",
                    "content": f""" 
                                    "请根据下列商品编号和商品名称对该商品进行多标签分类，商品的标签从上述已知level和label中选取。
                                    商品编号: 6900000058312, 商品名称: BBQ白木柄烧烤麻花针12支装" 
                                """
                },
                # {"role": "user", "content": "商品编号: 6900000058312, 商品名称: BBQ白木柄烧烤麻花针12支装"},
                {"role": "assistant", "content": "{\"包装方式\": \"支装\"}"},
                {"role": "user", "content": f"商品编号: {input_data['商品编号']}, 商品名称: {input_data['商品名称']}"}
            ]

            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)

            outputs = model.generate(
                input_ids,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.25,
                top_p=0.2,
                bos_token_id=tokenizer.bos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )
            response = outputs[0][input_ids.shape[-1]:]
            response_str = tokenizer.decode(response, skip_special_tokens=True)

            # 记录真实标签和预测标签
            true_labels.append(true_label)
            try:
                pred_label = json.loads(response_str)
            except json.JSONDecodeError:
                pred_label = {}
            pred_labels.append(pred_label)

            # 将原始标签和预测标签写入jsonl文件
            output_line = {
                "label": json.dumps(true_label, ensure_ascii=False),
                "predict": json.dumps(pred_label, ensure_ascii=False)
            }
            print(f"第{i + 1}个")
            print(output_line)
            outfile.write(json.dumps(output_line, ensure_ascii=False) + '\n')
            outfile.flush()

            save_last_processed_index(processed_index_file, processed_index + i + 1)  # 保存最新的商品索引


end_time = time.time()
execution_time = end_time - start_time

# 打印程序运行时间
print(f"程序运行时间：{execution_time} 秒")
