from datasets import load_dataset
from colorama import Fore

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, prepare_model_for_kbit_training
import torch

dataset = load_dataset("data", split='train')
print(Fore.YELLOW + str(dataset[2]) + Fore.RESET) 

def format_chat_template(batch, tokenizer):
    system_prompt = """You are a helpful, honest and harmless assistant designed to help engineers. Think through each question logically and provide an answer. Don't make things up, if you're unable to answer a question advise the user that you're unable to answer as it is outside of your scope."""

    samples = []

    # Access the inputs from the batch
    questions = batch["question"]
    answers = batch["answer"]

    for i in range(len(questions)):
        row_json = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": questions[i]},
            {"role": "assistant", "content": answers[i]}
        ]

        # Apply chat template - Qwen uses a different format
        text = tokenizer.apply_chat_template(
            row_json, 
            tokenize=False, 
            add_generation_prompt=False
        )
        samples.append(text)

    return {
        "instruction": questions,
        "response": answers,
        "text": samples
    }

# Use Qwen 0.6B model (no auth required)
base_model = "Qwen/Qwen2.5-0.5B"  # Using the latest available small Qwen model

tokenizer = AutoTokenizer.from_pretrained(
    base_model, 
    trust_remote_code=True,
)

# Add pad token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

train_dataset = dataset.map(
    lambda x: format_chat_template(x, tokenizer), 
    num_proc=8, 
    batched=True, 
    batch_size=10
)
print(Fore.LIGHTMAGENTA_EX + str(train_dataset[0]) + Fore.RESET) 

# Quantization config for efficient training
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",  # Changed to auto for better device management
    quantization_config=quant_config,
    cache_dir="./workspace",
    trust_remote_code=True,
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# LoRA configuration
peft_config = LoraConfig(
    r=64,  # Reduced for smaller model
    lora_alpha=128,  # Reduced accordingly
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Qwen specific modules
    task_type="CAUSAL_LM",
)

# Training configuration
training_args = SFTConfig(
    output_dir="./qwen-0.5b-finetuned",
    num_train_epochs=3,  # Reduced for initial testing
    per_device_train_batch_size=2,  # Small batch size for memory efficiency
    gradient_accumulation_steps=8,
    warmup_steps=100,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    eval_steps=100,
    save_total_limit=2,
    remove_unused_columns=False,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args,
    peft_config=peft_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    max_seq_length=512,  # Reasonable sequence length
)

# Start training
trainer.train()

# Save the model
trainer.save_model('./final_qwen_model')
trainer.model.save_pretrained("./final_qwen_checkpoint")
tokenizer.save_pretrained("./final_qwen_checkpoint")

print(Fore.GREEN + "Training completed! Model saved to ./final_qwen_checkpoint" + Fore.RESET)