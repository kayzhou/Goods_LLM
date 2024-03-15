**<span style="font-size: 30px;">基于语言模型的零售商品品类与标签识别</span>**

基础流程

1. 先获取四个传统深度学习模型的结果
2. 再获取四个预训练模型的结果
3. 最后获取四个大语言模型的结果

关键代码流程

-  cnn_tf.py TF-IDF结合CNN模型
-  cnn_wv.py Word2Vec结合CNN模型
-  rnn_tf.py TF-IDF结合RNN模型
-  rnn_wv.py Word2Vec结合RNN模型
-  lstm_tf.py TF-IDF结合LSTM模型
-  lstm_wv.py Word2Vec结合LSTM模型
-  gru_tf.py TF-IDF结合GRU模型
-  gru_wv.py Word2Vec结合GRU模型
-  bert.py BERT模型
-  albert.py ALBERT模型
-  roberta.py RoBERTa模型
-  ernie.py ERNIE模型
-  train_gpt.py 训练gpt模型
-  output_gpt.py 输出gpt模型结果
-  alpaca.py 输出Alpaca模型的结果
-  gemma.py 输出Gemma模型的结果
-  gemma_inference.py 输出微调Gemma模型的结果
-  output_openai.py 输出openai的结果
-  output_finetune_openai.py OpenAI微调后的结果输出

辅助代码

-  BLEU.py 计算BLEU得分
-  data_process_gpt.py gpt模型的数据预处理
-  generate_data_gpt.py 处理原始数据集为指定格式
-  get_vocab_gpt.py 生成字典信息
-  gpt_model.py gpt模型的实现
-  data_process_alpaca.py Alpaca模型的数据预处理
-  gemma_finetune.py 微调gemma模型
-  gemma_merge_lora.py 将LoRA适配器合并到原始的gemma模型中
-  data_process_finetune_openai.py 微调的OpenAI模型的数据预处理
-  data_process_openai.py OpenAI模型的数据预处理
-  Hardware_Information.py 查看硬件信息
-  finetune_openai.py 调用OpenAI的接口进行微调训练