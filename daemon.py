import os
import json
import time
import random
from core.engine import SelfEvolvingEngine
from core.critic import QualityCritic
from trainer.continuous_train import run_auto_train

# Seed topik agar AI bisa mengeksplorasi sendiri tanpa cari data manual
EXPLORATION_SEEDS = [
    "Sains fisika kuantum", "Algoritma sorting dan struktur data",
    "Arsitektur deep learning modern", "Sejarah peradaban kuno",
    "Optimasi query database SQL", "Konsep IoT dan microcontroller",
    "Logika matematika diskrit", "Prinsip keamanan sistem operasi"
]

def append_jsonl(file_path, data):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

def count_lines(file_path):
    if not os.path.exists(file_path):
        return 0
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

def main():
    print("[*] Memulai Autonomous Engine Daemon...")
    engine = SelfEvolvingEngine()
    critic = QualityCritic()
    
    golden_path = "storage/golden_dataset/train.jsonl"
    TARGET_BUFFER_THRESHOLD = 50  # Latih otomatis setiap ada 50 data berkualitas baru

    while True:
        try:
            # 1. Tahap Sintetis: Model membuat pertanyaan dari seed konsep
            topic = random.choice(EXPLORATION_SEEDS)
            q_gen_prompt = [
                {"role": "user", "content": f"Buat satu pertanyaan konseptual yang menantang dan mendalam tentang topik: {topic}. Hanya berikan pertanyaannya saja."}
            ]
            synth_question = engine.generate(q_gen_prompt, max_tokens=64, temperature=0.9)

            # 2. Model menjawab pertanyaannya sendiri
            a_gen_prompt = [
                {"role": "user", "content": synth_question}
            ]
            synth_answer = engine.generate(a_gen_prompt, max_tokens=300, temperature=0.5)

            # 3. Critic memvalidasi kualitas data
            if critic.evaluate(synth_question, synth_answer):
                sample = {
                    "messages": [
                        {"role": "user", "content": synth_question},
                        {"role": "assistant", "content": synth_answer}
                    ]
                }
                append_jsonl(golden_path, sample)
                total_samples = count_lines(golden_path)
                print(f"[+] Sampel valid tersimpan ({total_samples}/{TARGET_BUFFER_THRESHOLD}): {synth_question[:40]}...")

                # 4. Cek apakah ambang batas tercapai untuk auto-retrain
                if total_samples >= TARGET_BUFFER_THRESHOLD:
                    print("\n[!] Ambang batas data tercapai. Memicu Auto-Training...")
                    success = run_auto_train(golden_data_path=golden_path)
                    if success:
                        engine.reload_adapter()
                        # Reset file golden untuk siklus iterasi berikutnya
                        open(golden_path, "w").close()
                        print("[✓] Buffer di-reset. Melanjutkan siklus evolusi berikutnya.\n")

            time.sleep(2)  # Delay antar siklus eksplorasi agar CPU/GPU tetap stabil

        except KeyboardInterrupt:
            print("\n[-] Daemon dimatikan.")
            break
        except Exception as e:
            print(f"[!] Error pada siklus: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()