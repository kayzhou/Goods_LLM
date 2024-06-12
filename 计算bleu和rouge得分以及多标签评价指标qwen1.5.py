import json
import jieba
import nltk
import numpy as np
from rouge_chinese import Rouge
from sklearn.metrics import hamming_loss, accuracy_score, precision_recall_fscore_support
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import time

# 确保已经下载了相关的nltk数据
nltk.download('punkt')

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

# 计算多标签分类的评价指标
def calculate_metrics(true_labels, pred_labels):
    all_labels = sorted(set(label for d in true_labels for label in d.keys()))
    
    true_binary = []
    pred_binary = []
    
    for true, pred in zip(true_labels, pred_labels):
        true_binary.append([1 if label in true else 0 for label in all_labels])
        pred_binary.append([1 if label in pred else 0 for label in all_labels])
    
    hl = hamming_loss(true_binary, pred_binary)
    accuracy = accuracy_score(true_binary, pred_binary)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(true_binary, pred_binary, average='macro')
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(true_binary, pred_binary, average='micro')
    
    return {
        'hamming_loss': hl,
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
    }

# 主函数
def main(file_path, metrics_output_filename):
    start_time = time.time()
    
    data = read_jsonl(file_path)
    
    # 存储结果
    score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
    true_labels = []
    pred_labels = []
    
    for item in data:
        label = json.loads(item['label'])
        predict = json.loads(item['predict'])
        
        # 将字典转为字符串
        label_str = json.dumps(label, ensure_ascii=False)
        predict_str = json.dumps(predict, ensure_ascii=False)
        
        # 计算ROUGE
        if len(" ".join(chinese_tokenize(predict_str)).split()) == 0 or len(" ".join(chinese_tokenize(label_str)).split()) == 0:
            result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
        else:
            result = calculate_rouge(label_str, predict_str)
        
        for k, v in result.items():
            score_dict[k].append(round(v["f"] * 100, 4))
        
        # 计算BLEU-4
        bleu_score = calculate_bleu(label_str, predict_str)
        score_dict["bleu-4"].append(round(bleu_score * 100, 4))
        
        true_labels.append(label)
        pred_labels.append(predict)
    
    # 计算平均分
    avg_scores = {k: float(np.mean(v)) for k, v in score_dict.items()}
    
    # 计算多标签分类的评价指标
    metrics = calculate_metrics(true_labels, pred_labels)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # 打印程序运行时间
    print(f"程序运行时间：{execution_time} 秒")
    
    # 将运行时间写入评价指标字典
    metrics['execution_time'] = execution_time
    
    # 将所有评价指标写入metrics字典
    metrics.update({
        "avg_bleu-4": avg_scores['bleu-4'],
        "avg_rouge-1": avg_scores['rouge-1'],
        "avg_rouge-2": avg_scores['rouge-2'],
        "avg_rouge-l": avg_scores['rouge-l']
    })
    print(metrics)
    
    # 将评价指标写入json文件
    with open(metrics_output_filename, 'w', encoding='utf-8') as metrics_file:
        json.dump(metrics, metrics_file, ensure_ascii=False, indent=4)

# 调用主函数
model_name = "qwen1.5"
task = "duobiaoqian"
test_label = 1
model_label = "base"
file_path = f'/home/llm/liguanqun/llama3/data/results/{model_name}/{model_label}/test{test_label}/{model_name}_{task}_analysis_results_test_{test_label}.jsonl'  # 请将此路径替换为实际的jsonl文件路径
metrics_output_filename = f'/home/llm/liguanqun/llama3/data/results/{model_name}/{model_label}/test{test_label}/{model_name}_{task}_evaluation_metrics_{test_label}.json'  # 请将此文件名替换为实际的输出文件名
main(file_path, metrics_output_filename)
