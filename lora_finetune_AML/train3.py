from datasets import load_dataset
from colorama import Fore

from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
import torch

def format_chat_template(batch, tokenizer):
    system_prompt = """You are a helpful, honest and harmless assitant designed to help engineers. Think through each question logically and provide an answer. Don't make things up, if you're unable to answer a question advise the user that you're unable to answer as it is outside of your scope."""

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

        # Apply chat template and append the result to the list
        tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
        text = tokenizer.apply_chat_template(row_json, tokenize=False)
        samples.append(text)

    # Return a dictionary with lists as expected for batched processing
    return {
        "instruction": questions,
        "response": answers,
        "text": samples  # The processed chat template text for each row
    }

if __name__ == '__main__':
    dataset = load_dataset("data", split='train')
    print(Fore.YELLOW + str(dataset[2]) + Fore.RESET) 

    base_model = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(
            base_model, 
            trust_remote_code=True
    )

    # Single-threaded processing (avoids multiprocessing issues)
    train_dataset = dataset.map(
        lambda x: format_chat_template(x, tokenizer), 
        batched=True, 
        batch_size=10
    )
    print(Fore.LIGHTMAGENTA_EX + str(train_dataset[0]) + Fore.RESET) 

    # CPU-only model loading (no quantization)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",  # This will use CPU automatically
        torch_dtype=torch.float32,  # Use float32 for CPU
        cache_dir="./workspace",
    )

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    # LoRA configuration for CPU training
    peft_config = LoraConfig(
        r=16,  # Reduced from 256 for CPU training
        lora_alpha=32,  # Reduced from 512
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Specific modules instead of "all-linear"
        task_type="CAUSAL_LM",
    )

    # Training configuration optimized for CPU
    training_config = SFTConfig(
        output_dir="meta-llama/Llama-3.2-1B-SFT",
        num_train_epochs=3,  # Reduced from 50 for faster training
        per_device_train_batch_size=1,  # Small batch size for CPU
        gradient_accumulation_steps=4,  # Simulate larger batch size
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=50,
        eval_steps=50,
        warmup_steps=10,
        dataloader_num_workers=0,  # No multiprocessing for data loading
        fp16=False,  # Disable fp16 for CPU
        bf16=False,  # Disable bf16 for CPU
    )

    trainer = SFTTrainer(
        model,
        train_dataset=train_dataset,
        args=training_config,
        peft_config=peft_config,
    )

    print("Starting training on CPU (this will be slow but will work)...")
    trainer.train()

    trainer.save_model('complete_checkpoint')
    trainer.model.save_pretrained("final_model")
    print("Training completed!")