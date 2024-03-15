import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import jieba
from gensim.models import Word2Vec

# 检查GPU可用性并选择设备
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# 读取CSV数据
df = pd.read_csv("/home/llm/liguanqun/毕设new/data/蚂蚁商联数据.csv")
# df = df[:100000]
df['cate_4'].fillna('default_value', inplace=True)
print(df['cate_1'].value_counts())
print(df['cate_2'].value_counts())
print(df['cate_3'].value_counts())
print(df['cate_4'].value_counts())

# 中文分词
df['tokenized_text'] = df['goods_name'].apply(lambda x: ' '.join(jieba.cut(x)))

# Train Word2Vec model
word2vec_model = Word2Vec(sentences=df['tokenized_text'], vector_size=100, window=5, min_count=1, workers=4)

# Transform the sentences into word embeddings
X_w2v = df['tokenized_text'].apply(lambda x: [word2vec_model.wv[word] for word in x])
X_w2v = torch.tensor([torch.tensor(sent).mean(dim=0).numpy() for sent in X_w2v], dtype=torch.float32).to(device)
print(X_w2v)
print(X_w2v.shape)

# 调整y的取值范围
y_range = range(1, 5)  # Assuming cate_1 to cate_4
results = []

for i in range(1, 5):
    print(f'cate_{i}')
    # 创建新的标签列并将其移动到GPU
    y = torch.tensor(df[f'cate_{i}'].astype('category').cat.codes.values, dtype=torch.long).to(device)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_w2v, y, test_size=0.2, random_state=42)

    # 获取词汇表的大小
    vocab_size = X_w2v.shape[1]
    print(vocab_size)

    # 定义GRU模型并将其移动到GPU
    class GRUClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(GRUClassifier, self).__init__()
            self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            _, x = self.gru(x)
            x = x[-1, :, :]  # 取最后一个时间步的输出
            x = self.fc(x)
            return x

    model = GRUClassifier(input_size=vocab_size, hidden_size=128, output_size=len(df[f'cate_{i}'].unique())).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 创建DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 模型训练
    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as t:
            for batch_X, batch_y in t:
                optimizer.zero_grad()
                output = model(batch_X.unsqueeze(1).to(device))  # 添加一个维度并将其移动到GPU
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                t.set_postfix(loss=total_loss / len(t))

        # 模型评估
        model.eval()
        with torch.no_grad():
            test_output = model(X_test.unsqueeze(1).to(device))  # 添加一个维度并将其移动到GPU
            _, predicted = torch.max(test_output, 1)

            # 收集评估指标
            accuracy = accuracy_score(y_test.cpu().numpy(), predicted.cpu().numpy())
            precision = precision_score(y_test.cpu().numpy(), predicted.cpu().numpy(), average='weighted', zero_division=1)
            recall = recall_score(y_test.cpu().numpy(), predicted.cpu().numpy(), average='weighted', zero_division=1)
            f1 = f1_score(y_test.cpu().numpy(), predicted.cpu().numpy(), average='weighted')

            # 打印并保存结果
            results.append({
                'Category': f'cate_{i}',
                'Epoch': epoch + 1,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'Loss': total_loss / len(train_loader)
            })
            print(f'Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}')

# 将结果保存到CSV文件
results_df = pd.DataFrame(results)
results_df.to_csv('/home/llm/liguanqun/毕设new/代码/gru/wv-evaluation_results.csv', index=False)
