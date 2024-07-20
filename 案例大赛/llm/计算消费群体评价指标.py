import json
import jieba
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 读取CSV文件
def read_csv(file_path):
    return pd.read_csv(file_path)

# 读取jsonl文件
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# 提取消费群体和所属类型
def extract_consumer_groups(data, goods_name_column='goods_name'):
    consumer_groups = {}
    for item in data:
        goods_name = item[goods_name_column].strip("\"")
        predict_data = json.loads(item['predict'])
        groups = [
            predict_data.get(f'消费群体1', ""),
            predict_data.get(f'消费群体2', ""),
            predict_data.get(f'消费群体3', "")
        ]
        consumer_groups[goods_name] = groups
    return consumer_groups

# 计算分类指标
def compute_classification_metrics(labels, predictions):
    accuracy = accuracy_score(labels, predictions)
    macro_precision = precision_score(labels, predictions, average='macro', zero_division=0)
    macro_recall = recall_score(labels, predictions, average='macro', zero_division=0)
    macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    micro_precision = precision_score(labels, predictions, average='micro', zero_division=0)
    micro_recall = recall_score(labels, predictions, average='micro', zero_division=0)
    micro_f1 = f1_score(labels, predictions, average='micro', zero_division=0)
    weighted_precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    weighted_recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    weighted_f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    return accuracy, macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1, weighted_precision, weighted_recall, weighted_f1

# 主函数
def main(csv_file_path, jsonl_file_path, metrics_output_filename):
    csv_data = read_csv(csv_file_path)
    jsonl_data = read_jsonl(jsonl_file_path)
    
    csv_consumer_groups = csv_data[['goods_name', '消费群体']]
    
    jsonl_consumer_groups = extract_consumer_groups(jsonl_data)
    
    labels, predictions = [], []
    
    for _, row in csv_consumer_groups.iterrows():
        goods_name = row['goods_name']
        if goods_name in jsonl_consumer_groups:
            label = row['消费群体']
            predicts = jsonl_consumer_groups[goods_name]
            
            labels.append(label)
            if label in predicts:
                predictions.append(label)
            else:
                predictions.append("")

    if not labels or not predictions:
        print("No valid data found.")
        return
    
    accuracy, macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1, weighted_precision, weighted_recall, weighted_f1 = compute_classification_metrics(labels, predictions)
    
    metrics = {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
    }
    
    print(metrics)
    
    # 将评价指标写入json文件
    with open(metrics_output_filename, 'w', encoding='utf-8') as metrics_file:
        json.dump(metrics, metrics_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    model_name = 'llama3'
    # 将 'your_file.csv' 和 'your_file.jsonl' 替换为你的实际文件路径
    csv_file_path = "/home/llm/liguanqun/案例大赛/data/consumer_6.csv"
    jsonl_file_path = f"/home/llm/liguanqun/案例大赛/results/{model_name}_consumer_analysis_results_test.jsonl"
    metrics_output_filename = f"/home/llm/liguanqun/llama3/data/results/消费群体/{model_name}_metrics_output.json"
    main(csv_file_path, jsonl_file_path, metrics_output_filename)
