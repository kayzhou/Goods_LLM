import torch
from transformers import TrainingArguments

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


### COMMENT IN TO MERGE PEFT AND BASE MODEL ####
from peft import AutoPeftModelForCausalLM

# Load PEFT model on CPU
model = AutoPeftModelForCausalLM.from_pretrained(
    args.output_dir,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
# Merge LoRA and base model and save
merged_model = model.merge_and_unload()
merged_model.save_pretrained(args.output_dir,safe_serialization=True, max_shard_size="2GB")