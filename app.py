import json
import os
from core.engine import SelfEvolvingEngine
from core.critic import QualityCritic

def append_jsonl(file_path, data):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

def main():
    engine = SelfEvolvingEngine()
    critic = QualityCritic()
    golden_path = "storage/golden_dataset/train.jsonl"

    print("\n" + "="*50)
    print(" AUTONOMOUS AI TERMINAL (Self-Improving Engine)")
    print(" Setiap interaksi valid otomatis dipelajari oleh model.")
    print("="*50 + "\n")

    history = []

    while True:
        try:
            user_input = input("\nUser : ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "keluar"]:
                break

            history.append({"role": "user", "content": user_input})
            response = engine.generate(history, max_tokens=300)
            print(f"\nAI   : {response}")

            history.append({"role": "assistant", "content": response})

            # Otomatis simpan interaksi harian ke dataset pelatihan
            if critic.evaluate(user_input, response):
                sample = {
                    "messages": [
                        {"role": "user", "content": user_input},
                        {"role": "assistant", "content": response}
                    ]
                }
                append_jsonl(golden_path, sample)

            if len(history) > 6:
                history = history[-6:]

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()