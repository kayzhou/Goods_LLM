import os
import csv
import torch
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

import logging
# 配置日志记录以输出训练过程信息
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt='%H:%M:%S', level=logging.INFO)

class Config:
    data_path = "/home/llm/liguanqun/毕设new/data/蚂蚁商联数据.csv"
    model_path = "/home/llm/liguanqun/model/chinese-roberta-wwm-ext"
    checkpoint_path = "/home/llm/liguanqun/model/chinese-roberta-wwm-ext/checkpoints/"
    result_file = "/home/llm/liguanqun/毕设new/代码/roberta/results.csv"
    max_len = 64
    batch_size = 16
    epochs = 5

# 加载并预处理数据的函数
def load_data(config):
    data = pd.read_csv(config.data_path, encoding='utf-8', low_memory=False)
    # data=data[:100]
    label_encoders = [LabelEncoder() for _ in range(4)]
    
    # 将分类标签编码为数字
    for i in range(4):
        data[f'encoded_label_{i+1}'] = label_encoders[i].fit_transform(data[f'cate_{i+1}'])
        
    # 将数据划分为训练集和测试集
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    return data, train_data, test_data

# 加载tokenizer的函数
def load_tokenizer(config):
    return BertTokenizer.from_pretrained(config.model_path)

# 处理PyTorch中数据的自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index]['goods_name']
        barcode = self.data.iloc[index]['barcode']
        formatted_text = f'<id>{barcode}</id><name>{text}</name>'
        labels = [self.data.iloc[index][f'encoded_label_{i+1}'] for i in range(4)]
        # print(labels)
        
        # 对输入文本进行标记化和格式化
        encoding = self.tokenizer(
            formatted_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt',
            padding='max_length'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

# 检查GPU是否可用
if torch.cuda.is_available():
    device = torch.device('cuda:1')
    print('GPU可用。使用GPU:', torch.cuda.get_device_name(device))
else:
    device = torch.device('cpu')
    print('GPU不可用。使用CPU.')
# device = torch.device('cpu')

# 训练和测试模型的函数
def train_and_test_model(config, level, data, train_data, test_data, tokenizer):
    # 定义模型和训练参数
    model = BertForSequenceClassification.from_pretrained(config.model_path, num_labels=data[f'encoded_label_{level}'].nunique())
    model.to(device)
    batch_size = config.batch_size
    epochs = config.epochs

    # 定义优化器和损失函数
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()

    # 可选：加载之前保存的检查点
    load_checkpoint = False  # 如果为True，加载检查点；否则，从头开始训练

    if load_checkpoint:
        checkpoint_path = os.path.join(config.checkpoint_path, f'cate{level}', f'model_checkpoint_epoch_{epoch + 1}.pt')  
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

    # 准备数据加载器
    train_dataset = CustomDataset(train_data, tokenizer, max_len=64)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 初始化用于存储最后一个epoch结果的列表
    last_epoch_results = []

    start_time = time.time()  # 记录开始时间

    # 训练模型
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels[:, level-1])
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        average_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {average_loss}')
        
        # 仅保存最后一个epoch的结果
        if epoch == epochs - 1:
            last_epoch_results.append({
                'cate': f'cate{epoch+1}',
                'Train Loss': average_loss,
                'Test Loss': 0,  
                'Test Accuracy': 0,  
                'Precision': 0,
                'Recall': 0,
                'F1 Score': 0
            })

            # 保存检查点（包括权重和偏置）
            checkpoint_path = os.path.join(config.checkpoint_path, f'cate{level}', f'model_checkpoint_epoch_{epoch + 1}.pt')

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': average_loss,
            }, checkpoint_path)

            # 测试模型
            model.eval()
            test_dataset = CustomDataset(test_data, tokenizer, max_len=config.max_len)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch in tqdm(test_loader, desc='Testing'):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    outputs = model(input_ids, attention_mask=attention_mask)
                    preds = torch.argmax(outputs.logits, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels[:, level-1].cpu().numpy())

            # 计算测试损失
            test_loss = criterion(outputs.logits, labels[:, level-1])
            last_epoch_results[0]['Test Loss'] = test_loss.item() 

            # 计算并保存最后一个epoch的测试准确度、测试损失和F1分数
            accuracy = accuracy_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds, average='weighted')
            recall = recall_score(all_labels, all_preds, average='weighted')
            f1 = f1_score(all_labels, all_preds, average='weighted')
            last_epoch_results[0]['Test Accuracy'] = accuracy
            last_epoch_results[0]['Test Loss'] = test_loss.item() 
            last_epoch_results[0]['F1 Score'] = f1
            last_epoch_results[0]['Precision'] = precision 
            last_epoch_results[0]['Recall'] = recall
            print(f'Test Accuracy: {accuracy}')

            # 将准确率等结果保存到CSV文件
            result_file = config.result_file

            with open(result_file, 'a') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([f"cate{level}", average_loss, test_loss.item(), accuracy, precision, recall, f1])
            print(f'结果已保存至 {result_file}')

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算经过的时间
    print(f'Total training time for cate{level}: {elapsed_time} seconds')


# 主函数，执行训练和测试过程
def main():
    config = Config()
    data, train_data, test_data = load_data(config)
    tokenizer = load_tokenizer(config)
    levels = [1, 2, 3, 4]

    # 将准确率等结果保存到CSV文件
    result_file = config.result_file
    if os.path.exists(result_file):
        os.remove(result_file)
    with open(result_file, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["cate", "Train Loss", "Test Loss", "Test Accuracy", "Precision", "Recall", "F1 Score"])

    start_time_total = time.time()  # 记录整个程序运行的开始时间

    for level in levels:
        print(f"cate_{level}")
        train_and_test_model(config, level, data, train_data, test_data, tokenizer)

    end_time_total = time.time()  # 记录整个程序运行的结束时间
    elapsed_time_total = end_time_total - start_time_total  # 计算整个程序运行的总时间

    # 将整个程序运行的总时间记录到文件中
    with open("/home/llm/liguanqun/毕设new/代码/roberta/time.txt", "a") as f:
        f.write(f"Total training time for all categories: {elapsed_time_total} seconds\n")

# 脚本的入口点
if __name__ == '__main__':
    main()
