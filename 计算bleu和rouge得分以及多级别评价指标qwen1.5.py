import json
import jieba
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_chinese import Rouge

# 确保已经下载了相关的nltk数据
import nltk
# nltk.download('punkt')

# 读取jsonl文件
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# 使用jieba进行中文分词
def chinese_tokenize(text):
    return list(jieba.cut(text))

# 计算BLEU-4分数
def calculate_bleu(label, predict):
    label_tokens = chinese_tokenize(label)
    predict_tokens = chinese_tokenize(predict)
    return sentence_bleu([label_tokens], predict_tokens, smoothing_function=SmoothingFunction().method3)

# 计算ROUGE分数
def calculate_rouge(label, predict):
    rouge = Rouge()
    scores = rouge.get_scores(" ".join(chinese_tokenize(predict)), " ".join(chinese_tokenize(label)))
    return scores[0]

# 提取四级分类
def extract_fourth_category(category_str):
    try:
        category_dict = json.loads(category_str)
        if '四级分类' in category_dict:
            return category_dict['四级分类'].split('>')[0]  # 修改为适当的索引值
        else:
            return None
    except (IndexError, KeyError, json.JSONDecodeError):
        return None

# 计算分类指标
def compute_classification_metrics(labels, predictions):
    accuracy = accuracy_score(labels, predictions)
    macro_precision = precision_score(labels, predictions, average='macro', zero_division=0)
    macro_recall = recall_score(labels, predictions, average='macro', zero_division=0)
    macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    micro_precision = precision_score(labels, predictions, average='micro', zero_division=0)
    micro_recall = recall_score(labels, predictions, average='micro', zero_division=0)
    micro_f1 = f1_score(labels, predictions, average='micro', zero_division=0)
    return accuracy, macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1

# 主函数
def main(file_path, metrics_output_filename):
    data = read_jsonl(file_path)
    labels, predictions = [], []
    score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
    
    for i, item in enumerate(data):
        try:
            label_fourth = extract_fourth_category(item['label'])
            predict_fourth = extract_fourth_category(item['predict'])
            
            if label_fourth is not None and predict_fourth is not None:
                labels.append(label_fourth)
                predictions.append(predict_fourth)
                
                # 计算ROUGE
                if len(chinese_tokenize(predict_fourth)) == 0 or len(chinese_tokenize(label_fourth)) == 0:
                    result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
                else:
                    result = calculate_rouge(label_fourth, predict_fourth)
                
                for k, v in result.items():
                    score_dict[k].append(round(v["f"] * 100, 4))
                
                # 计算BLEU-4
                bleu_score = calculate_bleu(label_fourth, predict_fourth)
                score_dict["bleu-4"].append(round(bleu_score * 100, 4))
        except Exception as e:
            print(f"Error processing item {i}: {e}")
            continue
    
    if not labels or not predictions:
        print("No valid data found.")
        return
    
    accuracy, macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1 = compute_classification_metrics(labels, predictions)
    avg_scores = {k: float(np.mean(v)) for k, v in score_dict.items()}
    
    metrics = {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        "avg_bleu-4": avg_scores['bleu-4'],
        "avg_rouge-1": avg_scores['rouge-1'],
        "avg_rouge-2": avg_scores['rouge-2'],
        "avg_rouge-l": avg_scores['rouge-l']
    }
    
    print(metrics)
    
    # 将评价指标写入json文件
    with open(metrics_output_filename, 'w', encoding='utf-8') as metrics_file:
        json.dump(metrics, metrics_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # 将 'your_file.jsonl' 替换为你的实际文件路径
    file_path = "/home/llm/liguanqun/案例大赛/results/qwen1.5_sft_lora_duojibie_analysis_results_test_1.jsonl"
    metrics_output_filename = "/home/llm/liguanqun/llama3/data/results/chinese-llama3/sft/test1/第1级类别-qwen1.5_sft_lora_duojibie_analysis_results_metrics.json"
    main(file_path, metrics_output_filename)
