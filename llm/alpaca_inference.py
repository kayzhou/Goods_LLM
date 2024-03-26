# 导入所需的库和模块
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import datetime
import os
import re
import traceback

# 定义一个对话类
class Dialog:
    def __init__(self, role, content, inst=None):
        self.role = role  # 角色，可以是"user"或"assistant"
        self.content = content  # 内容，是一个字符串
        self.inst = inst  # 指令，是一个字符串或None

    def __repr__(self):
        return f"{self.role}: {self.content}"

# 定义一个对话数据集类
class DialogDataset:
    def __init__(self, data_file):
        self.data_file = data_file  # 数据文件的路径，是一个json文件
        self.dialogs = []  # 对话列表，每个元素是一个Dialog对象的列表
        self.load_data()  # 加载数据

    def load_data(self):
        # 从json文件中读取数据，并转换为Dialog对象
        with open(self.data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            # print(data)
            print("数据读取完毕")
            for dialog in data:
                # print(dialog)
                # time.sleep(20)
                dialog_list = []
                for turn in dialog:
                    # print(turn)
                    # time.sleep(20)
                    role = turn["role"]
                    content = turn["content"]
                    inst = turn.get("inst", None)  # 如果没有指令，则默认为None
                    dialog_list.append(Dialog(role, content, inst))
                    # print(dialog_list)
                    # time.sleep(10)
                self.dialogs.append(dialog_list)
            print("dialog合并完毕")

    def __len__(self):
        # 返回对话数据集的长度，即对话的个数
        return len(self.dialogs)

    def __getitem__(self, index):
        # 返回对话数据集中指定索引的对话，即一个Dialog对象的列表
        return self.dialogs[index]

# 定义一个对话生成器类
class DialogGenerator:
    def __init__(self, dataset, model_name, device="cuda:1"):
        self.dataset = dataset
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        self.load_model()
        self.resume_index_file = "/home/llm/liguanqun/毕设new/代码/alpaca/resume_index.txt"  # 添加一个保存索引值的文件路径

    def save_resume_index(self, index):
        with open(self.resume_index_file, "w") as f:
            f.write(str(index))

    def load_resume_index(self):
        if os.path.exists(self.resume_index_file):
            with open(self.resume_index_file, "r") as f:
                return int(f.read())
        else:
            return 0
    
    def load_model(self):
        # 从huggingface hub上加载分词器和模型，并将其移动到指定设备上
        self.tokenizer = AutoTokenizer.from_pretrained(
            "".join(self.model_name))
        self.model = AutoModelForCausalLM.from_pretrained(
            "".join(self.model_name))
        self.model.to(self.device)

    def generate_dialog(self, dialog):
        try:
            # 根据给定的对话列表生成对话的输出，并返回一个字符串
            input_ids = []  # 输入id列表，每个元素是一个整数
            # output_ids = []  # 输出id列表，每个元素是一个整数
            output_str = ""  # 输出字符串
            for turn in dialog:
                print(turn)
                if turn.role == "user1":
                    # 如果是用户角色，则将内容和指令拼接起来，并添加特殊标记
                    input_str = f"<s> [INST] <<SYS>> {turn.inst} <</SYS>> {turn.content} [/INST] "
                    # print(turn.content)
                    input_ids.extend(self.tokenizer.encode(
                        input_str, add_special_tokens=False))
                elif turn.role == "assistant":
                    # 如果是助理角色，则将内容作为输出，并添加特殊标记
                    input_str += f"{turn.content}</s>"
                    input_ids.extend(self.tokenizer.encode(
                        turn.content, add_special_tokens=False))
                    input_ids.extend(self.tokenizer.encode(
                        "</s>", add_special_tokens=False))  # 添加结束标记
                elif turn.role == "user2":
                    # 如果是助理角色，则将内容作为输出，并添加特殊标记
                    input_str += f"<s>[INST]{turn.content}[/INST]"
                    input_ids.extend(self.tokenizer.encode(
                        turn.content, add_special_tokens=False))
                    input_ids.extend(self.tokenizer.encode(
                        "</s>", add_special_tokens=False))  # 添加结束标记
            # 将输入id和输出id转换为张量，并移动到指定设备上
            # print(input_str)
            input_ids = torch.tensor([input_ids]).to(self.device)
            # output_ids = torch.tensor([output_ids]).to(self.device)
            # 使用模型生成输出，设置最大长度和温度等参数
            generated_ids = self.model.generate(
                input_ids, max_length=1024, temperature=0.2, do_sample=True, top_k=200, top_p=0.85)
            # 将生成的id转换为字符串，并添加到输出字符串中
            generated_str = self.tokenizer.decode(
                generated_ids[0], skip_special_tokens=True)
            output_str += f"{generated_str}\n"
            # 使用split方法分割字符串
            parts = re.split("商品分析结果为: |输出为: ", output_str)
            # 只保留后面的部分
            output_str_0 = parts[0]
            output_str_1 = parts[1]
            output_str_2 = parts[2]
            # 检查output_str_2是否满足正则表达式的格式
            if not re.match(r'{"商品名称": "(.*?)", "商品编号": "(.*?)", "多维标注": {(.*?)}}$', output_str_2):
                raise ValueError(
                    'output_str_2 does not match the regular expression')
        except IndexError as e:
            print("list index out of range")
            with open('/home/llm/liguanqun/毕设new/代码/alpaca/exception_data_new.txt', 'a', encoding='utf-8') as f:
                f.write(str(e) + '\n' + str(output_str) +
                        '\n' + traceback.format_exc()+'\n'+'\n')
        except ValueError as e:
            print("输出不符合格式")
            with open('/home/llm/liguanqun/毕设new/代码/alpaca/exception_data_new.txt', 'a', encoding='utf-8') as f:
                f.write(str(e) + '\n' + str(output_str) +
                        '\n' + traceback.format_exc()+'\n'+'\n')
        else:
            with open("/home/llm/liguanqun/毕设new/代码/alpaca/normal_data_new.txt", 'a', encoding='utf-8') as f:
                f.write(output_str_2)
                print(output_str_2)
        # return output_str, output_str_2 
    def run(self):
        resume_index = self.load_resume_index()  # 加载索引值
        if resume_index > 0:
            print(f"Resuming from dialog {resume_index + 1}")
        else:
            print("Starting from the beginning")
        try:
            for i, dialog in enumerate(self.dataset[resume_index:]):
                index = resume_index + i
                print(f"Dialog {index + 1}:")
                self.generate_dialog(dialog)
                self.save_resume_index(index + 1)  # 保存当前对话索引
        except KeyboardInterrupt:
            print("Interrupted by user, saving progress...")
            self.save_resume_index(index + 1)  # 如果中断，保存当前进度


# 创建一个对话数据集的实例，传入数据文件的路径
dataset = DialogDataset("/home/llm/liguanqun/毕设new/data/dialog_data.json")
generator = DialogGenerator(
    dataset, "/home/llm/liguanqun/model/chinese-alpaca-2-7b-hf")
# 运行对话生成器
generator.run()
