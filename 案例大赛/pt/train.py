import json
from tqdm import tqdm
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, hamming_loss
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


# 读取JSONL文件，限制只读取前num_lines行
def read_jsonl(file_path, num_lines=200000):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [f.readline().strip() for _ in range(num_lines)]
    return [json.loads(line) for line in lines if line]

# 假设文件名为 data.jsonl
file_path = "/home/llm/liguanqun/案例大赛/data/蚂蚁商联多标签_去除低频label.jsonl"
samples = read_jsonl(file_path)

# 提取所有可能的键值对
all_labels = set()
for sample in tqdm(samples, desc="Processing samples"):
    for key, value in sample["多维标注"].items():
        all_labels.add(f"{key}:{value}")

all_labels = sorted(all_labels)  # 保持顺序一致
label2id = {label: idx for idx, label in enumerate(all_labels)}

def encode_labels(multidimensional_labels, label2id):
    encoded = [0] * len(label2id)
    for key, value in multidimensional_labels.items():
        label = f"{key}:{value}"
        if label in label2id:
            encoded[label2id[label]] = 1
    return encoded

# 使用多线程并行化标签编码
with ThreadPoolExecutor() as executor:
    encoded_samples = list(tqdm(executor.map(lambda sample: encode_labels(sample["多维标注"], label2id), samples), total=len(samples), desc="Encoding labels"))

class CustomDataset(Dataset):
    def __init__(self, samples, labels, tokenizer, max_length):
        self.samples = samples
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text = self.samples[idx]["商品名称"]
        labels = self.labels[idx]
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        return inputs, torch.tensor(labels, dtype=torch.float)

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.manual_seed(42)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    
    device = torch.device(f'cuda:{rank}')
    torch.manual_seed(42)
    tokenizer = BertTokenizer.from_pretrained('/home/llm/liguanqun/model/bert-base-chinese')
    dataset = CustomDataset(samples, encoded_samples, tokenizer, max_length=64)

    # 划分训练集和测试集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_dataset, batch_size=16, sampler=train_sampler, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, sampler=test_sampler, num_workers=4)

    class MultiLabelBERT(nn.Module):
        def __init__(self, model_name, num_labels):
            super(MultiLabelBERT, self).__init__()
            self.bert = BertModel.from_pretrained(model_name)
            self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
            
        def forward(self, input_ids, attention_mask, token_type_ids=None):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            logits = self.classifier(outputs.pooler_output)
            return logits

    num_labels = len(all_labels)
    model = MultiLabelBERT('/home/llm/liguanqun/model/bert-base-chinese', num_labels).to(device)
    
    model = DDP(model, device_ids=[rank], output_device=rank)
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    results = {
        "train_loss": [],
        "eval_results": []
    }

    model.train()
    num_epochs = 3  
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Training epoch {epoch+1}/{num_epochs}"):
            inputs, labels = batch
            inputs = {key: val.to(device) for key, val in inputs.items()}
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(**inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        torch.distributed.barrier()  # 等待所有进程完成当前epoch的训练
        epoch_loss_tensor = torch.tensor(epoch_loss, dtype=torch.float32, device=device)
        dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
        epoch_loss_tensor /= world_size
        epoch_loss = epoch_loss_tensor.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        results["train_loss"].append(avg_epoch_loss)
        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")
    
    model_name = "bert-base-chinese"
    # 保存最后一个epoch的模型
    if rank == 0:
        torch.save(model.module.state_dict(), f"/home/llm/liguanqun/案例大赛/data/多标签分类结果/model/20万_{model_name}_final_model.pth")

    model.eval()
    preds = []
    true_labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = batch
            inputs = {key: val.to(device) for key, val in inputs.items()}
            labels = labels.to(device)
            
            logits = model(**inputs)
            preds.append(logits.sigmoid().round().cpu().numpy())
            true_labels.append(labels.cpu().numpy())

    preds = np.concatenate(preds)
    true_labels = np.concatenate(true_labels)

    gathered_preds = [torch.zeros_like(torch.tensor(preds, device=device)) for _ in range(world_size)]
    gathered_true_labels = [torch.zeros_like(torch.tensor(true_labels, device=device)) for _ in range(world_size)]
    dist.all_gather(gathered_preds, torch.tensor(preds, device=device))
    dist.all_gather(gathered_true_labels, torch.tensor(true_labels, device=device))

    if rank == 0:

        gathered_preds = torch.cat(gathered_preds).cpu().numpy()
        gathered_true_labels = torch.cat(gathered_true_labels).cpu().numpy()

        macro_f1 = f1_score(gathered_true_labels, gathered_preds, average='macro')
        micro_f1 = f1_score(gathered_true_labels, gathered_preds, average='micro')
        macro_precision = precision_score(gathered_true_labels, gathered_preds, average='macro')
        micro_precision = precision_score(gathered_true_labels, gathered_preds, average='micro')
        macro_recall = recall_score(gathered_true_labels, gathered_preds, average='macro')
        micro_recall = recall_score(gathered_true_labels, gathered_preds, average='micro')
        accuracy = accuracy_score(gathered_true_labels, gathered_preds)
        hamming = hamming_loss(gathered_true_labels, gathered_preds)

        eval_results = {
                "model_name": model_name,
                "macro_f1": macro_f1,
                "micro_f1": micro_f1,
                "macro_precision": macro_precision,
                "micro_precision": micro_precision,
                "macro_recall": macro_recall,
                "micro_recall": micro_recall,
                "accuracy": accuracy,
                "hamming_loss": hamming
            }
        
        results["eval_results"].append(eval_results)

        print(f"Macro F1 Score: {macro_f1}")
        print(f"Micro F1 Score: {micro_f1}")
        print(f"Macro Precision: {macro_precision}")
        print(f"Micro Precision: {micro_precision}")
        print(f"Macro Recall: {macro_recall}")
        print(f"Micro Recall: {micro_recall}")
        print(f"Accuracy: {accuracy}")
        print(f"Hamming Loss: {hamming}")

        with open(f'/home/llm/liguanqun/案例大赛/data/多标签分类结果/20万_{model_name}_training_results.json', 'w') as f:
            # for line in results:
            #     f.write(line + '\n')
            json.dump(results, f)

    cleanup()

if __name__ == "__main__":
    import os
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    train(rank, world_size)