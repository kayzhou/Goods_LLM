# # Linux
# import os

# cpu = os.popen('/proc/cpuinfo').read()

# cpu = cpu.strip().replace('\n', '').replace('\r', '').split(" ")
# cpu = cpu[len(cpu)-1]

# print(cpu)

# import torch


# 检查GPU是否可用
# if torch.cuda.is_available():
#     device = torch.device('cuda:0')
#     print('GPU可用。使用GPU:', torch.cuda.get_device_name(device))
# else:
#     device = torch.device('cpu')
#     print('GPU不可用。使用CPU.')

import torch
print(torch.__version__)


# import transformers
# print(transformers.__version__)




# huggingface-cli download \
# --resume-download google/gemma-2b \
# --local-dir /home/llm/liguanqun/model/gemma-2b \
# --local-dir-use-symlinks False \
# --token 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

import torch

try:
    assert torch.cuda.get_device_capability()[0] >= 8
    print("当前硬件支持 Flash Attention。")
except AssertionError:
    print("当前硬件不支持 Flash Attention。")
