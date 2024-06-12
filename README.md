# 基于语言模型的商品多级与多标签分类研究

## 关键代码流程

### 新增案例大赛相关代码

### 传统深度模型 > dl
-  cnn_tf.py TF-IDF结合CNN模型
-  cnn_wv.py Word2Vec结合CNN模型
-  rnn_tf.py TF-IDF结合RNN模型
-  rnn_wv.py Word2Vec结合RNN模型
-  lstm_tf.py TF-IDF结合LSTM模型
-  lstm_wv.py Word2Vec结合LSTM模型
-  gru_tf.py TF-IDF结合GRU模型
-  gru_wv.py Word2Vec结合GRU模型

### 预训练模型 > pt
-  bert.py BERT模型
-  albert.py ALBERT模型
-  roberta.py RoBERTa模型
-  ernie.py ERNIE模型

### 大语言模型 > llm
-  alpaca_inference.py 输出Alpaca模型的结果
-  gemma_finetune.py 对gemma模型微调
-  gemma_finetune_inference.py 输出微调Gemma模型结果
-  gemma_inference.py 输出Gemma模型结果（需要每次访问加载模型，可优化）
-  gemma_merge_lora.py 将LoRA合并到原始模型
-  gpt_get_vocab.py 生成GPT-1.0字典信息
-  gpt_inference.py 输出GPT-1.0模型结果
-  gpt_model.py GPT-1.0模型结构（自己构造）
-  gpt_train.py 训练GPT-1.0
-  openai_finetune.py 微调GPT-3.5-turbo模型
-  openai_finetune_inference.py 输出微调后的GPT-3.5-turbo结果
-  openai_inference.py 输出GPT-3.5-turbo模型结果

### 辅助代码 > utils
-  BLEU.py 计算BLEU得分
-  data_generate_llm.py 生成llm所需的对话数据格式
-  data_process_alpaca.py Alpaca模型的数据预处理
-  data_process_finetune_openai.py 微调GPT-3.5-turbo模型的数据预处理
-  data_process_gemma.py Gemma模型的数据预处理
-  data_process_gpt.py GPT-1.0模型的数据预处理
-  data_process_openai.py GPT-3.5-turbo模型的数据预处理
-  hardware_information.py 查看硬件信息
