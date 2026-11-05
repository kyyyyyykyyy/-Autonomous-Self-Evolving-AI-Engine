import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class SelfEvolvingEngine:
    def __init__(self, base_model_id="Qwen/Qwen2.5-0.5B-Instruct", adapter_dir="storage/adapters/latest"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_model_id = base_model_id
        self.adapter_dir = adapter_dir

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        self.load_model()

    def load_model(self):
        print(f"[*] Memuat base model ke {self.device}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )

        if os.path.exists(self.adapter_dir):
            print(f"[*] Menggabungkan adapter LoRA terbaru dari {self.adapter_dir}...")
            self.model = PeftModel.from_pretrained(self.model, self.adapter_dir)
        self.model.eval()

    def reload_adapter(self):
        """Memuat ulang bobot LoRA baru tanpa mematikan proses"""
        if os.path.exists(self.adapter_dir):
            print("[*] Hot-reloading adapter baru...")
            self.model = PeftModel.from_pretrained(self.model.base_model, self.adapter_dir)
            self.model.eval()

    def generate(self, messages, max_tokens=256, temperature=0.7):
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()