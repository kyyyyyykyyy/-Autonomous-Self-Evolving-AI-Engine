import os
import glob
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTConfig, SFTTrainer

def run_auto_train(
    golden_data_path="storage/golden_dataset/train.jsonl",
    base_model_id="Qwen/Qwen2.5-0.5B-Instruct",
    output_dir="storage/adapters/latest"
):
    if not os.path.exists(golden_data_path) or os.path.getsize(golden_data_path) == 0:
        print("[-] Data golden kosong. Training dilewati.")
        return False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )

    # Lanjutkan dari adapter yang sudah ada jika tersedia
    if os.path.exists(output_dir):
        model = PeftModel.from_pretrained(base_model, output_dir, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(base_model, lora_config)

    dataset = load_dataset("json", data_files=golden_data_path, split="train")

    def format_chat(batch):
        texts = [tokenizer.apply_chat_template(m, tokenize=False) for m in batch["messages"]]
        return {"text": texts}

    dataset = dataset.map(format_chat, batched=True)

    # Parameter training & tokenization dipindah ke SFTConfig
    training_args = SFTConfig(
        output_dir="storage/adapters/temp",
        dataset_text_field="text",
        max_seq_length=512,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        logging_steps=10,
        fp16=(device == "cuda"),
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args
    )

    print("[*] Menjalankan Autonomous Fine-Tuning...")
    trainer.train()

    # Simpan adapter baru
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[✓] Model berhasil diperbarui di {output_dir}")
    return True