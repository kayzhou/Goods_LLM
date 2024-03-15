import json
import math
from tqdm import tqdm

# 定义一个函数，用来读取文件中的数据
def read_file(filename):
    data = []
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    data.append(json.loads(line))
                except json.decoder.JSONDecodeError as e:
                    print(f"JSON 解析错误: {e}")
    except FileNotFoundError:
        print(f"文件 '{filename}' 未找到")
    return data

# 定义一个函数，用来计算n元语法匹配的个数
def count_ngram_matches(reference, hypothesis, n):
    reference_words = reference.split()
    hypothesis_words = hypothesis.split()

    reference_ngrams = {}
    for i in range(len(reference_words) - n + 1):
        ngram = tuple(reference_words[i:i+n])
        if ngram in reference_ngrams:
            reference_ngrams[ngram] += 1
        else:
            reference_ngrams[ngram] = 1

    matches = 0
    for i in range(len(hypothesis_words) - n + 1):
        ngram = tuple(hypothesis_words[i:i+n])
        if ngram in reference_ngrams and reference_ngrams[ngram] > 0:
            matches += 1
            reference_ngrams[ngram] -= 1

    return matches

# 定义一个函数，用来计算BLEU分数
def calculate_bleu(references, hypothesis, max_n=4, weights=None):
    reference_lens = [len(reference.split()) for reference in references]
    hypothesis_len = len(hypothesis.split())

    if weights is None:
        weights = [1 / max_n] * max_n

    closest_reference_len = min(reference_lens, key=lambda x: abs(x - hypothesis_len))
    brevity_penalty = min(1, closest_reference_len / hypothesis_len) if hypothesis_len > 0 else 0

    precisions = []
    for n in range(1, max_n + 1):
        matches = sum(count_ngram_matches(reference, hypothesis, n) for reference in references)
        precision = matches / sum(count_ngram_matches(reference, hypothesis, n) for reference in references) if matches > 0 else 0
        precisions.append(precision)

    geometric_mean = (p for p in precisions if p > 0)
    geometric_mean = math.exp(sum(math.log(p) for p in geometric_mean) / len(precisions)) if geometric_mean else 0

    bleu = brevity_penalty * geometric_mean
    return bleu


# 读取原始数据文件和模型预测结果文件
original_data = read_file("/home/llm/liguanqun/毕设new/data/蚂蚁商联数据_json串.txt")
predicted_data = read_file("/home/llm/liguanqun/毕设new/代码/gemma/商品分析结果.txt")

# 初始化累加器和计数器
total_bleu_score = 0
num_matches = 0

# 构建预测数据字典，以商品名称为键，多维标注为值
predicted_dict = {product["商品名称"]: product["多维标注"] for product in predicted_data}

# 遍历原始数据中的每个商品，并显示进度条
for original_product in tqdm(original_data, desc="Processing"):
    original_name = original_product["商品名称"]
    
    # 检查原始商品名称是否在预测数据中
    if original_name in predicted_dict:
        found_match = True
        original_multidim = original_product["多维标注"]
        predicted_multidim = predicted_dict[original_name]
        
        references = [json.dumps(original_multidim)]
        hypothesis = json.dumps(predicted_multidim)
        
        bleu_score = calculate_bleu(references, hypothesis)
        print(f"商品名称: {original_name}, BLEU 分数: {bleu_score}")
        
        # 累加 BLEU 分数
        total_bleu_score += bleu_score
        num_matches += 1
    else:
        found_match = False
        # print(f"未找到匹配的商品名称: {original_name}")

# 计算平均 BLEU 分数
average_bleu_score = total_bleu_score / num_matches if num_matches > 0 else 0
print(f"平均 BLEU 分数: {average_bleu_score}")