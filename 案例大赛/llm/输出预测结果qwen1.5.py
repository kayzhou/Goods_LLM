from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import time


start_time = time.time()

model_id = "/home/llm/liguanqun/llama3/code/LLaMA-Factory/models/duobiaoqian/qwen1.5-7b-chat-lora-sft"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto"
)

# 函数：保存最新的商品索引
def save_last_processed_index(file_path, index):
    with open(file_path, 'w') as f:
        f.write(str(index))

test_label = 5
model_name = "qwen1.5"
task = "duobiaoqian"
json_file_path = f"/home/llm/liguanqun/llama3/code/LLaMA-Factory/data/蚂蚁商联多维标注对话new_去除低频label_2000_test_{test_label}.json"
output_filename = f'/home/llm/liguanqun/llama3/data/{model_name}/test{test_label}/{model_name}_sft_lora_{task}_analysis_results_test_{test_label}.jsonl'
processed_index_file = f'/home/llm/liguanqun/llama3/data/{model_name}/test{test_label}/{model_name}_sft_lora_{task}_processed_index_test_{test_label}.txt'
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
                {"role": "system", "content": row['instruction']},
                {"role": "user", "content": "商品编号: 6900000058312, 商品名称: BBQ白木柄烧烤麻花针12支装"},
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
