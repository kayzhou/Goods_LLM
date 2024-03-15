import torch
from gpt_model import GPT

if __name__ == '__main__':
    # 设置GPU设备
    device = torch.device('cuda:1')

    # 加载模型
    model = GPT().to(device)
    model.load_state_dict(torch.load('/home/llm/liguanqun/毕设new/代码/gpt/GPT.pt'))
    model.eval()

    # 读取商品名称文件
    input_file_path = '/home/llm/liguanqun/毕设new/代码/gpt/商品名称.txt'
    output_file_path = '/home/llm/liguanqun/毕设new/代码/gpt/商品名称output.txt'

    with open(input_file_path, 'r', encoding='utf-8') as input_file, \
         open(output_file_path, 'w', encoding='utf-8') as output_file:

        for line in input_file:
            # 处理每行商品名称
            sentence = line.strip() + '\t'
            
            # 生成模型回答
            answer = model.answer(sentence)

            # 打印结果并写入输出文件
            print(f"商品名称: {sentence}")
            print(f"模型回答: {answer}\n")
            output_file.write(f"{sentence}\t{answer}\n")

    print("测试完成。结果已保存到", output_file_path)
