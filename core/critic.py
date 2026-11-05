import re

class QualityCritic:
    """
    Menyaring pasangan Q&A sintetis/interaksi agar hanya data 
    berkualitas tinggi yang masuk ke pipeline pelatihan.
    """
    def __init__(self, min_chars: int = 20, max_chars: int = 1500):
        self.min_chars = min_chars
        self.max_chars = max_chars

    def evaluate(self, prompt: str, response: str) -> bool:
        prompt = prompt.strip()
        response = response.strip()

        # 1. Validasi panjang karakter
        if len(response) < self.min_chars or len(response) > self.max_chars:
            return False

        # 2. Deteksi repetisi ekstrem / hallucination loop
        words = response.lower().split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.4:  # Terlalu banyak kata berulang
                return False

        # 3. Filter kegagalan respon umum
        refusal_patterns = [
            r"i don't know",
            r"saya tidak tahu",
            r"as an ai",
            r"sebagai model bahasa"
        ]
        for pattern in refusal_patterns:
            if re.search(pattern, response, re.IGNORECASE) and len(response) < 60:
                return False

        return True