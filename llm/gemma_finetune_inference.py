from transformers import TrainingArguments
from tqdm import tqdm
args = TrainingArguments(
    output_dir="liguanqun/毕设new/代码/gemma/gemma-product", # directory to save and repository id
    num_train_epochs=3,                     # number of training epochs
    per_device_train_batch_size=1,          # batch size per device during training
    gradient_accumulation_steps=2,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=10,                       # log every 10 steps
    save_strategy="epoch",                  # save checkpoint every epoch
    learning_rate=2e-4,                     # learning rate, based on QLoRA paper
    # bf16=True,                              # use bfloat16 precision if you have supported GPU
    # tf32=True,                              # use tf32 precision if you have supported GPU
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",           # use constant learning rate scheduler
    push_to_hub=False,                       # push model to hub
    report_to="tensorboard",                # report metrics to tensorboard
)



import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, pipeline

peft_model_id = args.output_dir

# Load Model with PEFT adapter
model = AutoPeftModelForCausalLM.from_pretrained(
    peft_model_id,
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
# load into pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)



from datasets import load_dataset
from random import randint



eval_dataset = load_dataset("json", data_files="/home/llm/liguanqun/毕设new/代码/gemma/test_dataset.json", split="train")
output_file = "/home/llm/liguanqun/毕设new/代码/gemma/微调结果.txt"


# Open file in write mode to clear its contents
with open(output_file, "w") as f:
    f.write("")

for i in tqdm(range(len(eval_dataset))):
    prompt = pipe.tokenizer.apply_chat_template(eval_dataset[i]["messages"][:2], tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=1024, do_sample=True, temperature=0.1, top_k=200, top_p=0.65, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)
    
    generated_text = outputs[0]['generated_text'][len(prompt):].strip()
    # Remove unwanted prefix
    generated_text = generated_text.replace("Generated Answer:", "").replace("商品分析结果为:", "").strip()

    print(f"Query:\n{eval_dataset[i]['messages'][1]['content']}")
    print(f"Original Answer:\n{eval_dataset[i]['messages'][2]['content']}")
    print(f"Generated Answer:\n{outputs[0]['generated_text'][len(prompt):].strip()}")
    
    with open(output_file, "a") as f:
        # f.write(f"Query:\n{eval_dataset[i]['messages'][1]['content']}\n")
        # f.write(f"Original Answer:\n{eval_dataset[i]['messages'][2]['content']}\n")
        f.write(f"{generated_text}")
        f.write("\n")

