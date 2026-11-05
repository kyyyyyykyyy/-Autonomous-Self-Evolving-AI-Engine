# 🧠 Autonomous Self-Evolving AI Engine

### Continuous Self-Play, Synthetic Data Generation, Logic Critic Verification & LoRA Hot-Reloading

<p align="center">
  <strong>A Local Autonomous AI Engine That Continuously Generates, Evaluates, Learns, and Evolves Through Incremental Fine-Tuning.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/PyTorch-2.6%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-12.4-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA">
  <img src="https://img.shields.io/badge/HuggingFace-Transformers%20%7C%20TRL-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face">
  <img src="https://img.shields.io/badge/PEFT-LoRA-8A2BE2?style=for-the-badge" alt="PEFT LoRA">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="MIT License">
</p>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Project Background](#-project-background)
- [Core Concept](#-core-concept)
- [System Architecture](#-system-architecture)
- [How It Works](#-how-it-works)
- [Core Components](#-core-components)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [System Requirements](#-system-requirements)
- [Installation Guide](#-installation-guide)
- [Configuration](#-configuration)
- [Running the System](#-running-the-system)
- [Autonomous Learning Workflow](#-autonomous-learning-workflow)
- [Golden Dataset](#-golden-dataset)
- [LoRA Fine-Tuning](#-lora-fine-tuning)
- [Live Adapter Hot-Reloading](#-live-adapter-hot-reloading)
- [Monitoring and Logs](#-monitoring-and-logs)
- [Storage Management](#-storage-management)
- [Performance Considerations](#-performance-considerations)
- [Advantages](#-advantages)
- [Limitations and Risks](#-limitations-and-risks)
- [Troubleshooting](#-troubleshooting)
- [Development Roadmap](#-development-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)

---

# 📌 Overview

**Autonomous Self-Evolving AI Engine** adalah proyek eksperimental untuk membangun sistem Large Language Model (LLM) lokal yang memiliki kemampuan menghasilkan data sintetis, mengevaluasi kualitas jawabannya sendiri, mengumpulkan data berkualitas, dan melakukan incremental fine-tuning secara otomatis.

Berbeda dengan pendekatan fine-tuning konvensional yang membutuhkan proses pengumpulan dataset dan training secara manual, proyek ini mengimplementasikan sebuah mekanisme yang disebut:

**Continuous Self-Improvement Flywheel.**

Melalui mekanisme tersebut, sistem dirancang untuk menjalankan siklus pembelajaran berulang:

1. Menghasilkan pertanyaan dan jawaban secara mandiri.
2. Mengevaluasi kualitas jawaban menggunakan Quality Critic Engine.
3. Memfilter respons yang tidak memenuhi standar kualitas.
4. Menyimpan pasangan pertanyaan dan jawaban yang lolos ke Golden Dataset.
5. Memulai proses fine-tuning ketika jumlah data mencapai ambang batas tertentu.
6. Menghasilkan adapter LoRA baru.
7. Memuat adapter terbaru ke inference engine tanpa harus menghentikan seluruh aplikasi.

Dengan demikian, sistem memiliki fondasi untuk menjalankan eksperimen pembelajaran berkelanjutan menggunakan sumber daya komputasi lokal.

**Foundation Model:** Qwen2.5-0.5B-Instruct  
**Fine-Tuning Method:** Supervised Fine-Tuning (SFT) menggunakan LoRA  
**Training Framework:** Hugging Face Transformers, TRL, dan PEFT  
**Execution Environment:** Python, PyTorch, dan NVIDIA CUDA

> Important: Sistem ini merupakan eksperimen autonomous self-improvement berbasis synthetic data generation dan incremental fine-tuning. Proses tersebut tidak menjamin peningkatan kecerdasan secara otomatis. Kualitas perkembangan model tetap bergantung pada kualitas data, kemampuan evaluator, konfigurasi training, dan evaluasi hasil model.

---

# 🎯 Project Background

Sebagian besar workflow pengembangan LLM memiliki pola yang relatif statis:

- Developer mengumpulkan dataset.
- Dataset dibersihkan dan diproses secara manual.
- Model menjalani proses fine-tuning.
- Model hasil training dimuat kembali.
- Jika ingin meningkatkan kemampuan model, proses tersebut diulangi.

Pendekatan tersebut cukup efektif, tetapi memiliki keterbatasan ketika ingin membangun sistem yang dapat terus melakukan eksperimen dan pembelajaran secara mandiri.

Proyek ini mencoba mengeksplorasi pendekatan alternatif melalui kombinasi beberapa teknologi:

- Autonomous Self-Play.
- Synthetic Data Generation.
- Heuristic Quality Evaluation.
- Golden Dataset Management.
- Continuous Supervised Fine-Tuning.
- Parameter-Efficient Fine-Tuning.
- Live Model Adapter Reloading.

Tujuan utamanya bukan sekadar menjalankan chatbot lokal, melainkan membangun sebuah engine eksperimental yang dapat mengelola siklus pengumpulan data dan pembaruan model secara otomatis.

---

# 🧩 Core Concept

Sistem ini dibangun berdasarkan empat mekanisme utama.

## 1. Autonomous Self-Play & Exploration

Sistem dapat menghasilkan pertanyaan dan jawaban secara mandiri berdasarkan knowledge seed atau topik eksplorasi yang tersedia.

Contohnya:

- Ilmu komputer.
- Matematika.
- Sains.
- Algoritma.
- Logika.
- Pengetahuan umum.
- Pemecahan masalah.

Model menghasilkan pertanyaan, kemudian mencoba menjawab pertanyaan tersebut.

Hasil interaksi ini menjadi kandidat data sintetis yang nantinya dievaluasi oleh Quality Critic Engine.

Tujuan mekanisme ini adalah menghasilkan kandidat dataset baru tanpa harus selalu menyediakan dataset eksternal secara manual.

## 2. Quality Critic Engine

Tidak semua jawaban yang dihasilkan model layak digunakan sebagai data training.

Karena itu, proyek ini menggunakan sistem evaluasi otomatis bernama Quality Critic Engine.

Komponen ini bertugas menyaring respons berdasarkan sejumlah indikator kualitas, seperti:

- Panjang respons.
- Kepadatan repetisi.
- Pola pengulangan kata atau kalimat.
- Respons kosong atau terlalu dangkal.
- Pola jawaban yang terindikasi mengalami looping.
- Indikator kualitas logika berdasarkan heuristik yang diterapkan.

Hanya data yang memenuhi kriteria seleksi yang diteruskan ke Golden Dataset.

**Catatan:** Heuristic critic bukan pembuktian kebenaran faktual. Filter berbasis pola dapat mengurangi sebagian respons buruk, tetapi tidak dapat menjamin seluruh halusinasi terdeteksi.

## 3. Continuous LoRA Fine-Tuning

Ketika jumlah data berkualitas mencapai threshold yang ditentukan, sistem dapat memicu proses incremental Supervised Fine-Tuning secara otomatis.

Training menggunakan:

- Hugging Face Transformers.
- Hugging Face TRL.
- Hugging Face PEFT.
- LoRA (Low-Rank Adaptation).

Pendekatan LoRA memungkinkan proses fine-tuning dilakukan dengan melatih sejumlah kecil parameter tambahan tanpa harus memperbarui seluruh parameter foundation model.

## 4. Live Adapter Hot-Reloading

Setelah training selesai, adapter LoRA terbaru dapat dimuat kembali ke inference engine.

Tujuannya adalah mengurangi kebutuhan untuk menghentikan dan menjalankan ulang keseluruhan aplikasi ketika adapter model diperbarui.

Implementasi hot-reloading harus memperhatikan sinkronisasi antara proses inference dan penggantian adapter agar tidak terjadi konflik akses GPU atau penggunaan checkpoint yang belum lengkap.

---

# 🏗️ System Architecture

Berikut gambaran arsitektur utama sistem:

```text
                 AUTONOMOUS SELF-IMPROVEMENT FLYWHEEL
                                  |
                 +----------------+----------------+
                 |                                 |
                 v                                 v
        SYNTHETIC EXPLORATION                REAL USER CHATS
        (Self-Generated Q&A)                (Interactive CLI)
                 |                                 |
                 +----------------+----------------+
                                  |
                                  v
                     +------------------------+
                     |   QUALITY CRITIC ENGINE|
                     +------------------------+
                     | - Length Validation    |
                     | - Repetition Detection |
                     | - Loop Detection       |
                     | - Heuristic Evaluation |
                     +-----------+------------+
                                 |
                                 v
                        QUALITY VALIDATION
                                 |
                    +------------+------------+
                    |                         |
                    v                         v
               REJECTED                   ACCEPTED
                    |                         |
                    v                         v
              Discard Data             GOLDEN DATASET
                                        train.jsonl
                                             |
                                             v
                                    THRESHOLD CHECK
                                             |
                                             v
                                  CONTINUOUS SFT TRAINER
                                             |
                                  +----------+----------+
                                  |                     |
                                  v                     v
                               PEFT /                CHECKPOINT
                                LoRA                 VALIDATION
                                  |                     |
                                  +----------+----------+
                                             |
                                             v
                                       NEW ADAPTER
                                             |
                                             v
                                    ADAPTER HOT-RELOAD
                                             |
                                             v
                                      UPDATED MODEL
                                             |
                                             v
                                  CONTINUOUS INFERENCE
```

### Architecture Explanation

| Component | Responsibility |
|---|---|
| Synthetic Exploration | Menghasilkan pertanyaan dan jawaban sintetis. |
| Real User Chats | Menyediakan kandidat data dari percakapan pengguna. |
| Quality Critic Engine | Mengevaluasi dan menyaring kualitas data. |
| Golden Dataset | Menyimpan pasangan Q&A yang lolos seleksi. |
| Threshold Controller | Memeriksa apakah jumlah data memenuhi syarat training. |
| Continuous SFT Trainer | Melakukan incremental supervised fine-tuning. |
| PEFT / LoRA | Mengelola parameter tambahan untuk fine-tuning. |
| Checkpoint Validation | Memastikan hasil training tersedia dan dapat dimuat. |
| Adapter Hot-Reload | Mengganti adapter yang digunakan inference engine. |
| Inference Engine | Menghasilkan respons menggunakan model aktif. |

---

# ⚙️ How It Works

Secara keseluruhan, sistem dirancang bekerja melalui beberapa tahap.

## Stage 1 — Model Initialization

Ketika aplikasi dijalankan, inference engine memuat foundation model Qwen2.5-0.5B-Instruct ke perangkat komputasi yang tersedia.

Model tersebut menjadi fondasi utama untuk seluruh proses generasi dan percakapan.

## Stage 2 — Autonomous Exploration

Background daemon menjalankan aktivitas eksplorasi secara berkala.

Sistem menghasilkan pertanyaan berdasarkan knowledge seed yang tersedia, kemudian meminta model menghasilkan jawaban.

Hasilnya dikumpulkan sebagai kandidat synthetic training data.

## Stage 3 — Quality Evaluation

Setiap kandidat data diteruskan ke Quality Critic Engine.

Respons yang tidak memenuhi kriteria kualitas ditolak.

Respons yang lolos diteruskan ke proses penyimpanan.

## Stage 4 — Golden Dataset Accumulation

Data valid disimpan ke:

```text
storage/golden_dataset/train.jsonl
```

Dataset bertambah secara bertahap selama sistem menemukan data baru yang memenuhi kriteria.

## Stage 5 — Automatic Training Trigger

Ketika jumlah sampel baru mencapai threshold tertentu, background daemon dapat memicu proses training.

Contoh konfigurasi:

```python
TRAINING_THRESHOLD = 50
```

Artinya, proses training dipicu ketika jumlah data baru yang memenuhi kriteria mencapai 50 sampel, sesuai mekanisme counter yang digunakan implementasi.

## Stage 6 — Incremental Fine-Tuning

Trainer membaca Golden Dataset dan menjalankan Supervised Fine-Tuning menggunakan LoRA.

Adapter hasil training disimpan pada direktori checkpoint sementara.

## Stage 7 — Adapter Deployment

Setelah training berhasil dan checkpoint lolos pemeriksaan, adapter baru dapat dipindahkan atau dipublikasikan ke direktori adapter aktif.

## Stage 8 — Hot-Reloading

Inference engine memuat adapter terbaru tanpa harus menghentikan keseluruhan aplikasi.

Siklus eksplorasi, evaluasi, pengumpulan data, dan fine-tuning kemudian dapat dilanjutkan.

---

# 🧠 Core Components

## 1. Inference Engine

File:

```text
core/engine.py
```

Bertanggung jawab terhadap:

- Memuat foundation model.
- Menjalankan text generation.
- Mengelola proses inference.
- Mengintegrasikan adapter LoRA.
- Menyediakan mekanisme adapter hot-reloading.

## 2. Quality Critic Engine

File:

```text
core/critic.py
```

Bertanggung jawab terhadap:

- Memeriksa panjang respons.
- Mengidentifikasi repetisi ekstrem.
- Mendeteksi pola looping.
- Menjalankan validasi berbasis heuristik.
- Menentukan apakah kandidat data layak disimpan.

## 3. Continuous Trainer

File:

```text
trainer/continuous_train.py
```

Bertanggung jawab terhadap:

- Membaca Golden Dataset.
- Menyiapkan data training.
- Menginisialisasi LoRA configuration.
- Menjalankan SFTTrainer.
- Menyimpan checkpoint adapter baru.

## 4. Autonomous Daemon

File:

```text
daemon.py
```

Bertanggung jawab terhadap:

- Menjalankan background exploration.
- Menghasilkan kandidat data sintetis.
- Mengintegrasikan hasil percakapan pengguna jika diaktifkan.
- Memanggil Quality Critic Engine.
- Mengelola akumulasi dataset.
- Memantau threshold training.
- Memicu proses fine-tuning.

## 5. Interactive CLI Application

File:

```text
app.py
```

Menyediakan antarmuka terminal interaktif untuk:

- Berkomunikasi dengan model.
- Mengajukan pertanyaan.
- Melakukan eksperimen percakapan.
- Menghasilkan kandidat data interaksi pengguna.

Percakapan tidak otomatis menjadi data training hanya karena terjadi. Data harus melalui mekanisme seleksi yang diterapkan sistem.

---

# 🛠️ Technology Stack

| Component | Technology | Description |
|---|---|---|
| Programming Language | Python 3.11.x | Bahasa utama pengembangan sistem. |
| Deep Learning Framework | PyTorch 2.6+ | Komputasi tensor dan akselerasi GPU. |
| GPU Acceleration | NVIDIA CUDA 12.4 | Akselerasi training dan inference. |
| Foundation Model | Qwen2.5-0.5B-Instruct | Model bahasa dasar. |
| Model Framework | Hugging Face Transformers | Loading dan inference model. |
| Fine-Tuning Method | LoRA | Parameter-Efficient Fine-Tuning. |
| Adapter Management | Hugging Face PEFT | Pengelolaan adapter. |
| Training Framework | Hugging Face TRL | Implementasi supervised fine-tuning. |
| Dataset Management | Hugging Face Datasets | Pemrosesan dataset training. |
| Background Automation | Python Threading / Schedule | Pengelolaan aktivitas background. |
| Progress Monitoring | tqdm | Monitoring proses melalui terminal. |

---

# 📂 Project Structure

Struktur direktori yang digunakan dalam rancangan proyek:

```text
ncm-from-scratch/
│
├── core/
│   ├── __init__.py
│   ├── engine.py
│   └── critic.py
│
├── trainer/
│   ├── __init__.py
│   └── continuous_train.py
│
├── storage/
│   │
│   ├── golden_dataset/
│   │   └── train.jsonl
│   │
│   └── adapters/
│       │
│       ├── latest/
│       │
│       └── temp/
│
├── daemon.py
├── app.py
├── requirements.txt
├── README.md
└── LICENSE
```

### Directory Explanation

| Directory / File | Function |
|---|---|
| `core/` | Menyimpan komponen utama inference dan evaluasi kualitas. |
| `core/engine.py` | Model inference, generation, dan adapter management. |
| `core/critic.py` | Quality filtering dan heuristic validation. |
| `trainer/` | Menyimpan implementasi training. |
| `continuous_train.py` | Background incremental LoRA training. |
| `storage/` | Tempat penyimpanan dataset dan adapter. |
| `golden_dataset/` | Menyimpan dataset hasil seleksi. |
| `adapters/latest/` | Lokasi adapter terbaru yang digunakan sistem. |
| `adapters/temp/` | Lokasi checkpoint training sementara. |
| `daemon.py` | Background autonomous service. |
| `app.py` | Interactive CLI application. |
| `requirements.txt` | Daftar dependensi Python. |
| `README.md` | Dokumentasi proyek. |
| `LICENSE` | Informasi lisensi proyek. |

---

# 💻 System Requirements

## 1. Hardware Requirements

| Hardware | Minimum Target | Recommended |
|---|---|---|
| GPU | NVIDIA GPU dengan VRAM sekitar 4 GB | RTX 3060 / RTX 4050 atau lebih tinggi |
| RAM | 8 GB | 16 GB atau lebih |
| Storage | 15–20 GB ruang kosong | 30 GB atau lebih |
| CPU | 64-bit multi-core processor | Modern multi-core processor |

**Important:** Kebutuhan memori aktual bergantung pada ukuran batch, sequence length, precision, optimizer, konfigurasi LoRA, dan implementasi training. GPU dengan VRAM 4 GB tidak menjamin seluruh konfigurasi training dapat berjalan tanpa Out Of Memory (OOM).

## 2. Software Requirements

| Software | Version |
|---|---|
| Operating System | Windows 10/11 64-bit atau Linux Ubuntu 22.04+ |
| Python | 3.11.x |
| PyTorch | 2.6+ |
| CUDA | Build PyTorch yang kompatibel dengan CUDA 12.4 |
| NVIDIA Driver | Versi yang mendukung CUDA runtime terkait |
| Git | Latest stable version |

**Note:** CUDA Toolkit yang terpasang secara sistem tidak selalu harus sama dengan CUDA runtime yang digunakan wheel PyTorch. Pastikan driver NVIDIA kompatibel dengan build PyTorch yang dipilih.

---

# 🚀 Installation Guide

Panduan berikut menggunakan Windows PowerShell sebagai contoh utama.

## Step 1 — Clone Repository

```powershell
git clone https://github.com/kyyyyyykyyy/ncm-from-scratch.git

cd ncm-from-scratch
```

## Step 2 — Create Python Virtual Environment

Pastikan Python 3.11 sudah terinstal.

Periksa versi Python:

```powershell
py -3.11 --version
```

Buat virtual environment:

```powershell
py -3.11 -m venv venv
```

Aktifkan environment:

```powershell
.\venv\Scripts\Activate.ps1
```

Jika menggunakan Command Prompt:

```cmd
venv\Scripts\activate.bat
```

Jika PowerShell menolak menjalankan activation script karena Execution Policy:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Kemudian aktifkan kembali environment.

## Step 3 — Upgrade pip

```powershell
python -m pip install --upgrade pip
```

## Step 4 — Install PyTorch with CUDA Support

Untuk lingkungan yang menggunakan CUDA 12.4:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Pastikan versi PyTorch dan wheel CUDA sesuai dengan kombinasi sistem yang digunakan.

## Step 5 — Install AI Dependencies

```powershell
pip install transformers datasets peft accelerate trl schedule tqdm
```

## Step 6 — Verify PyTorch and GPU

Jalankan:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU Not Available')"
```

Jika GPU berhasil terdeteksi, output akan menampilkan informasi perangkat NVIDIA.

Contoh:

```text
PyTorch: 2.6.0+cu124
CUDA Available: True
GPU: NVIDIA GeForce RTX 3050
```

Output di atas merupakan contoh. Versi aktual mengikuti instalasi sistem.

## Step 7 — Verify Project Dependencies

Pastikan seluruh library dapat diimpor:

```powershell
python -c "import torch, transformers, datasets, peft, trl, accelerate; print('All dependencies imported successfully.')"
```

Jika tidak terdapat error, environment siap digunakan untuk pengujian berikutnya.

---

# ⚙️ Configuration

Beberapa parameter penting yang dapat digunakan dalam pengembangan sistem:

| Parameter | Example Value | Description |
|---|---|---|
| Base Model | `Qwen/Qwen2.5-0.5B-Instruct` | Foundation model. |
| Training Threshold | `50` | Target jumlah sampel sebelum training. |
| LoRA Rank | `16` | Rank adapter LoRA. |
| Training Precision | `FP16` | Mixed/half precision sesuai dukungan GPU. |
| Dataset Path | `storage/golden_dataset/train.jsonl` | Lokasi dataset. |
| Adapter Path | `storage/adapters/latest` | Lokasi adapter aktif. |
| Temporary Path | `storage/adapters/temp` | Lokasi checkpoint sementara. |

Contoh konfigurasi LoRA:

```python
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ]
)
```

Konfigurasi di atas merupakan contoh konfigurasi LoRA. Implementasi aktual harus disesuaikan dengan arsitektur foundation model dan konfigurasi trainer.

---

# 🎮 Running the System

Sistem dirancang untuk dijalankan menggunakan dua terminal terpisah.

Pastikan kedua terminal menggunakan virtual environment yang sama.

## Terminal 1 — Autonomous Background Daemon

Jalankan:

```powershell
python daemon.py
```

Daemon bertugas menjalankan aktivitas autonomous exploration, quality filtering, dataset accumulation, dan training trigger.

Contoh gambaran log:

```text
[*] Starting Autonomous Engine Daemon...
[*] Loading foundation model...
[*] Initializing autonomous exploration...

[+] Generating synthetic question...
[+] Evaluating generated response...
[+] Sample accepted by Quality Critic.
[+] Valid samples stored (12/50)

[+] Generating synthetic question...
[+] Evaluating generated response...
[+] Sample accepted by Quality Critic.
[+] Valid samples stored (13/50)

[!] Training threshold reached.
[*] Starting Continuous Fine-Tuning...

[*] Initializing LoRA configuration...
[*] Loading Golden Dataset...
[*] Running SFTTrainer...

[+] Training completed.
[+] New adapter checkpoint saved.
[*] Updating active adapter...

[✓] Adapter successfully reloaded.
```

Log tersebut merupakan ilustrasi alur yang diharapkan, bukan jaminan output aktual.

## Terminal 2 — Interactive Chat Application

Buka terminal baru.

Aktifkan kembali environment:

```powershell
cd ncm-from-scratch

.\venv\Scripts\Activate.ps1
```

Jalankan aplikasi:

```powershell
python app.py
```

Terminal ini digunakan untuk melakukan percakapan langsung dengan model.

Contoh:

```text
========================================
       AUTONOMOUS AI ENGINE
========================================

User: Explain how a binary search algorithm works.

AI: Binary search is an efficient searching
algorithm that operates on sorted data...

User: Why is its complexity O(log n)?

AI: Because the search interval is reduced
by approximately half in each iteration...

========================================
```

Percakapan yang memenuhi kriteria seleksi dapat digunakan sebagai kandidat data tambahan apabila mekanisme integrasi percakapan pengguna telah diaktifkan.

---

# 🔄 Autonomous Learning Workflow

Salah satu bagian utama proyek ini adalah mekanisme pembelajaran berulang.

Berikut alur konseptualnya:

```text
        +------------------------+
        |   Foundation Model     |
        +-----------+------------+
                    |
                    v
        +------------------------+
        | Autonomous Exploration |
        +-----------+------------+
                    |
                    v
        +------------------------+
        | Generate Synthetic Q&A |
        +-----------+------------+
                    |
                    v
        +------------------------+
        |   Quality Critic       |
        +-----------+------------+
                    |
             Quality Check
                    |
          +---------+---------+
          |                   |
          v                   v
        REJECT              ACCEPT
          |                   |
          v                   v
       Discard          Golden Dataset
                              |
                              v
                       Threshold Check
                              |
                              v
                       LoRA Fine-Tuning
                              |
                              v
                       New Adapter
                              |
                              v
                       Hot-Reloading
                              |
                              v
                       Updated Model
                              |
                              +------> Repeat
```

### Why This Matters

Pendekatan ini memungkinkan sistem menjalankan eksperimen pengumpulan data dan fine-tuning secara berulang.

Namun, keberhasilan setiap siklus harus dinilai melalui evaluasi model yang terpisah.

Model yang telah dilatih belum tentu lebih baik daripada versi sebelumnya.

Oleh karena itu, pengujian kualitas sebelum dan sesudah training tetap diperlukan.

---

# 📚 Golden Dataset

Golden Dataset merupakan kumpulan pasangan pertanyaan dan jawaban yang telah melewati proses quality filtering.

Lokasi:

```text
storage/golden_dataset/train.jsonl
```

Format JSONL memungkinkan setiap baris menyimpan satu objek JSON secara independen.

Contoh:

```json
{"question": "What is binary search?", "answer": "Binary search is an efficient algorithm for searching sorted data by repeatedly dividing the search interval in half."}
{"question": "What is the difference between RAM and VRAM?", "answer": "RAM is the primary system memory used by the CPU, while VRAM is dedicated memory used by the GPU."}
```

### Dataset Quality Principles

Dataset yang baik seharusnya memiliki karakteristik berikut:

- Jawaban relevan terhadap pertanyaan.
- Tidak mengandung repetisi berlebihan.
- Tidak mengalami looping.
- Memiliki struktur yang dapat dipelajari.
- Tidak mengandung informasi yang sengaja dibuat-buat.
- Memiliki format konsisten.
- Tidak mengandung data sensitif yang tidak seharusnya digunakan untuk training.

**Important:** Penyaringan otomatis tidak menggantikan validasi manusia. Untuk dataset yang akan digunakan dalam pengembangan model serius, audit sampel dan pemeriksaan faktual tetap disarankan.

---

# 🧬 LoRA Fine-Tuning

Proyek ini menggunakan metode Low-Rank Adaptation (LoRA).

LoRA merupakan pendekatan Parameter-Efficient Fine-Tuning (PEFT) yang memungkinkan model mempelajari perubahan perilaku melalui parameter tambahan yang relatif kecil.

Alih-alih memperbarui seluruh bobot foundation model, LoRA menambahkan low-rank matrices pada modul tertentu.

### Benefits of LoRA

- Mengurangi jumlah parameter yang perlu dilatih.
- Mengurangi kebutuhan penyimpanan checkpoint.
- Memungkinkan adapter dikelola secara terpisah.
- Mendukung penggunaan kembali foundation model yang sama.
- Memudahkan eksperimen dengan beberapa versi adapter.

### Training Configuration

Contoh parameter yang digunakan:

```python
r = 16
lora_alpha = 32
lora_dropout = 0.05
```

Target module:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

Penggunaan target module tersebut harus disesuaikan dengan struktur model yang dipilih.

---

# 🔁 Live Adapter Hot-Reloading

Salah satu fitur eksperimental utama adalah mekanisme adapter hot-reloading.

Dalam workflow fine-tuning konvensional, model sering kali perlu dimuat ulang setelah adapter diperbarui.

Proyek ini mencoba mengurangi kebutuhan tersebut dengan memisahkan foundation model dan adapter LoRA.

### Concept

```text
              ACTIVE INFERENCE ENGINE
                       |
                       v
                 Foundation Model
                       |
                       v
                  Adapter v1
                       |
                User Interaction
                       |
                       v
                 Background SFT
                       |
                       v
                  Adapter v2
                       |
                       v
                 Adapter Validation
                       |
                       v
                 Adapter Swap
                       |
                       v
                  Adapter v2
                       |
                       v
                Updated Inference
```

### Expected Benefits

- Tidak perlu selalu memuat ulang foundation model.
- Mengurangi downtime aplikasi.
- Memungkinkan pembaruan adapter secara bertahap.
- Memisahkan proses training dan inference.

### Important Implementation Considerations

Agar mekanisme ini benar-benar aman, implementasi perlu memperhatikan:

1. Training dilakukan pada checkpoint sementara.
2. Adapter hanya dipublikasikan setelah proses training selesai.
3. Adapter baru harus dapat dimuat sebelum menggantikan adapter aktif.
4. Pergantian adapter harus disinkronkan dengan inference.
5. Jika adapter baru gagal dimuat, adapter sebelumnya harus tetap tersedia.
6. Penggunaan GPU bersama antara training dan inference harus dikelola dengan benar.

Dengan demikian, hot-reloading merupakan target desain sistem, sedangkan tingkat zero-downtime aktual bergantung pada implementasi dan pengujian mekanismenya.

---

# 📊 Monitoring and Logs

Monitoring sangat penting untuk memahami perilaku sistem selama proses autonomous learning.

Beberapa informasi yang sebaiknya dipantau:

| Metric | Purpose |
|---|---|
| Generated Samples | Jumlah data sintetis yang dihasilkan. |
| Accepted Samples | Jumlah data yang lolos quality filtering. |
| Rejected Samples | Jumlah data yang ditolak. |
| Dataset Size | Jumlah data dalam Golden Dataset. |
| Training Trigger | Status threshold training. |
| Training Loss | Perubahan loss selama training. |
| GPU Utilization | Penggunaan GPU. |
| VRAM Usage | Konsumsi memori GPU. |
| Training Duration | Waktu yang diperlukan untuk training. |
| Active Adapter | Versi adapter yang sedang digunakan. |

Monitoring tersebut dapat membantu mendeteksi masalah seperti data berkualitas rendah, training yang tidak stabil, dan konsumsi resource berlebihan.

---

# 🧹 Storage Management

Proses pengembangan AI lokal dapat menghasilkan cache dan checkpoint berukuran cukup besar.

Berikut beberapa cara mengelola penyimpanan.

## 1. Clear Pip Cache

```powershell
pip cache purge
```

## 2. Set Hugging Face Cache Directory

Jika ingin mengarahkan cache Hugging Face ke drive D:

```powershell
[System.Environment]::SetEnvironmentVariable(
    'HF_HOME',
    'D:\ncm-from-scratch\.hf_cache',
    'User'
)
```

Setelah mengubah environment variable, buka terminal baru agar perubahan dapat dikenali.

## 3. Manage Adapter Checkpoints

Pastikan direktori berikut dikelola dengan benar:

```text
storage/
│
└── adapters/
    │
    ├── latest/
    │
    └── temp/
```

Checkpoint lama yang tidak lagi diperlukan dapat diarsipkan atau dihapus setelah memastikan adapter aktif dan backup tersedia.

Jangan menghapus checkpoint yang masih digunakan oleh proses training atau inference.

---

# ⚡ Performance Considerations

Karena proyek ini dirancang untuk berjalan secara lokal, pengelolaan resource merupakan aspek penting.

### GPU Memory

Penggunaan VRAM dipengaruhi oleh:

- Ukuran foundation model.
- Precision yang digunakan.
- Batch size.
- Sequence length.
- Optimizer.
- LoRA configuration.
- Jumlah proses yang berjalan bersamaan.

### Background Training

Training dan inference yang berjalan secara bersamaan dapat menyebabkan peningkatan konsumsi VRAM.

Sistem perlu memiliki mekanisme pengaturan resource agar proses background tidak mengganggu penggunaan aplikasi utama.

### Laptop Thermal Management

Jika sistem digunakan untuk training dalam waktu lama:

- Pastikan ventilasi laptop tidak terhalang.
- Gunakan pendinginan yang memadai.
- Pantau temperatur GPU dan CPU.
- Hindari beban training berlebihan.
- Perhatikan thermal throttling.

### Dataset Quality

Jumlah data bukan satu-satunya faktor yang menentukan kualitas model.

Dataset yang kecil tetapi berkualitas dapat lebih berguna daripada dataset besar yang mengandung banyak kesalahan.

---

# ✅ Advantages

## 1. Autonomous Data Generation

Sistem dirancang menghasilkan kandidat training data secara mandiri melalui self-play dan exploration.

## 2. Automated Quality Filtering

Quality Critic Engine membantu mengurangi masuknya data yang repetitif, dangkal, atau mengalami looping.

## 3. Parameter-Efficient Fine-Tuning

LoRA memungkinkan proses fine-tuning dilakukan tanpa memperbarui seluruh parameter foundation model.

## 4. Incremental Learning Workflow

Data dapat dikumpulkan secara bertahap dan digunakan untuk memicu proses training berikutnya.

## 5. Adapter-Based Model Updates

Pemisahan foundation model dan adapter memberikan fleksibilitas dalam pengelolaan model.

## 6. Local Execution

Sistem dirancang untuk menjalankan inference dan training menggunakan perangkat lokal.

## 7. Experimental Extensibility

Arsitekturnya dapat dikembangkan dengan evaluator yang lebih canggih, dataset eksternal, dan metode training lainnya.

---

# ⚠️ Limitations and Risks

Proyek ini memiliki beberapa keterbatasan yang perlu dipahami.

## 1. Self-Training Does Not Guarantee Intelligence Improvement

Model yang menghasilkan data untuk dirinya sendiri tidak otomatis menghasilkan pengetahuan baru yang benar.

Jika data sintetis berkualitas buruk, proses training justru dapat memperkuat kesalahan yang sudah ada.

## 2. Risk of Model Collapse

Penggunaan synthetic data secara berulang tanpa kontrol kualitas yang memadai dapat menyebabkan penurunan keragaman respons dan memperkuat pola kesalahan.

Karena itu, evaluasi dataset dan model sangat penting.

## 3. Heuristic Critic Limitations

Filter berbasis heuristik tidak dapat memahami seluruh konteks, membuktikan kebenaran fakta, atau mendeteksi semua bentuk halusinasi.

## 4. Limited Foundation Model Capacity

Qwen2.5-0.5B-Instruct merupakan model berukuran relatif kecil.

Kemampuannya dalam memahami konteks kompleks dan mempertahankan pengetahuan faktual memiliki keterbatasan dibandingkan model dengan parameter lebih besar.

## 5. Training Resource Consumption

Meskipun LoRA lebih hemat resource dibandingkan full fine-tuning, proses training tetap membutuhkan VRAM, RAM, CPU, dan daya listrik.

## 6. Catastrophic Forgetting

Fine-tuning berulang pada dataset yang sempit dapat menyebabkan model kehilangan sebagian kemampuan yang sebelumnya dimiliki.

Evaluasi regression dan penggunaan dataset referensi dapat membantu mengurangi risiko tersebut.

## 7. Data Privacy

Jika percakapan pengguna digunakan sebagai kandidat data training, data tersebut perlu melalui kebijakan privasi, persetujuan pengguna, dan sanitasi informasi sensitif.

Local execution saja tidak otomatis menjamin bahwa seluruh proses bebas dari komunikasi jaringan. Hal tersebut perlu diperiksa melalui implementasi dan konfigurasi aktual.

---

# 🛠️ Troubleshooting

## Problem 1 — CUDA Not Available

Error:

```text
CUDA Available: False
```

Possible solutions:

1. Pastikan NVIDIA GPU terdeteksi.
2. Periksa versi driver NVIDIA.
3. Pastikan wheel PyTorch sesuai dengan CUDA runtime yang digunakan.
4. Instal ulang PyTorch dengan build CUDA yang sesuai.

Verifikasi:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

## Problem 2 — CUDA Out of Memory

Error:

```text
torch.cuda.OutOfMemoryError
```

Possible solutions:

- Kurangi batch size.
- Kurangi sequence length.
- Kurangi beban proses inference.
- Tutup aplikasi GPU lain.
- Evaluasi penggunaan precision.
- Sesuaikan konfigurasi training.

## Problem 3 — Module Not Found

Error:

```text
ModuleNotFoundError
```

Solution:

```powershell
pip install transformers datasets peft accelerate trl schedule tqdm
```

Pastikan virtual environment aktif sebelum menginstal library.

## Problem 4 — Training Fails

Beberapa penyebab yang mungkin:

- Dataset tidak valid.
- Format JSONL salah.
- Target module LoRA tidak cocok.
- Versi library tidak kompatibel.
- VRAM tidak mencukupi.
- Checkpoint tidak dapat ditulis.

Periksa log training untuk mengetahui sumber kesalahan sebenarnya.

## Problem 5 — Adapter Loading Failed

Pastikan:

- Checkpoint training telah selesai ditulis.
- File konfigurasi adapter tersedia.
- Bobot adapter tersedia.
- Adapter sesuai dengan foundation model.
- Tidak ada proses lain yang sedang mengganti direktori adapter.

Simpan adapter lama sebagai fallback sebelum mengganti adapter aktif.

---

# 🗺️ Development Roadmap

Roadmap berikut menggambarkan perkembangan dan eksperimen yang direncanakan dalam proyek.

### Completed / Implemented Milestones

- [x] Autonomous Self-Play Data Synthesizer.
- [x] Heuristic Critic & Repetition Filter.
- [x] Continuous SFTTrainer with PEFT/LoRA.
- [x] Live Adapter Hot-Reloading Mechanism.

### Future Research & Development

- [ ] Integrasi Stage 1 Next-Concept Modeling.
- [ ] Eksperimen Residual Vector Quantization (RVQ).
- [ ] Eksperimen Vector Quantized Variational Autoencoder (VQ-VAE).
- [ ] Non-autoregressive Hierarchical Coarse-to-Fine Parallel Decoding.
- [ ] Advanced Logic Verification.
- [ ] External Knowledge Validation.
- [ ] Dataset Quality Scoring.
- [ ] Model Regression Testing.
- [ ] Adapter Versioning and Rollback.
- [ ] Advanced Resource Scheduling.
- [ ] Automated Evaluation Benchmark.
- [ ] Long-Term Learning Stability Experiments.

Roadmap dapat berubah mengikuti hasil eksperimen, kebutuhan pengembangan, dan ketersediaan resource.

---

# 🤝 Contributing

Kontribusi dari developer, peneliti, dan penggemar AI sangat terbuka.

Jika ingin berkontribusi:

1. Fork repository.
2. Buat branch baru.
3. Implementasikan perubahan.
4. Lakukan pengujian.
5. Pastikan perubahan tidak merusak fitur yang sudah ada.
6. Buat Pull Request.

Contoh:

```bash
git fork
git checkout -b feature/new-improvement

# Implement your changes

git add .
git commit -m "Add new improvement"

git push origin feature/new-improvement
```

### Contribution Ideas

Beberapa area yang menarik untuk dikembangkan:

- Improved Quality Critic.
- Better Synthetic Data Generation.
- Advanced Training Scheduler.
- Memory-Efficient Training.
- Adapter Management.
- Evaluation Benchmarks.
- Model Stability Research.
- Better CLI Experience.

Diskusi, laporan bug, ide, dan saran pengembangan sangat dihargai.

---

# 🔐 License

Proyek ini menggunakan lisensi **MIT License**.

Silakan lihat file [LICENSE](LICENSE) untuk informasi lengkap mengenai ketentuan penggunaan, modifikasi, distribusi, dan kontribusi.

---

# 👨‍💻 Author

**Muhamad Adzky Maulana**

Independent AI Developer & Experimental AI Systems Builder.

Project: **Autonomous Self-Evolving AI Engine**

GitHub Repository:

https://github.com/kyyyyyykyyy/ncm-from-scratch

---

<p align="center">
  <strong>Built with curiosity, experimentation, and a passion for autonomous AI systems.</strong>
</p>

<p align="center">
  <i>"Don't just use LLMs. Build systems that can learn, evaluate, adapt, and evolve."</i>
</p>

<p align="center">
  ⭐ If you find this project interesting, consider giving the repository a star!
</p>

---

**Disclaimer:** This is an experimental AI research and development project. Autonomous data generation, self-evaluation, and iterative fine-tuning do not guarantee improved reasoning, factual accuracy, or general intelligence. Always evaluate model performance and validate generated training data before deploying updated adapters.