# SVG Scaling Laws 

Exploring scaling laws for decoder-only Transformer language models trained on SVG (Scalable Vector Graphics) code. Models are trained at 5 scales (1M–88M parameters) under both Standard Parameterization (SP) and Maximal Update Parameterization (µP), with power-law scaling curves fitted to validation loss.

---

## Project Structure

```
.
├── ML_Final project.ipynb   
├── nanoGPT/                         
│   ├── train.py            
│   ├── model.py              
│   ├── data/svg/                   
│   └── config/
│       ├── train_svg_sp_tiny.py
│       ├── train_svg_sp_small.py
│       ├── train_svg_sp_medium.py
│       ├── train_svg_sp_large.py
│       ├── train_svg_sp_xl.py
│       ├── train_svg_sp_tiny_lr_1e-4.py
│       ├── train_svg_sp_tiny_lr_3e-4.py
│       ├── train_svg_sp_tiny_lr_1e-3.py
│       ├── train_svg_sp_tiny_lr_3e-3.py
│       └── train_svg_sp_tiny_lr_1e-2.py
├── nanoGPT_mup/                     
│   ├── train.py                     
│   ├── model.py                     
│   ├── data/svg/                  
│   └── config/
│       ├── train_svg_mup_tiny_final.py
│       ├── train_svg_mup_small_final.py
│       ├── train_svg_mup_medium_final.py
│       ├── train_svg_mup_large_final.py
│       ├── train_svg_mup_xl_final.py
│       ├── train_svg_mup_tiny_lr_00001.py
│       ├── train_svg_mup_tiny_lr_00003.py
│       ├── train_svg_mup_tiny_lr_0001.py
│       ├── train_svg_mup_tiny_lr_0003.py
│       └── train_svg_mup_tiny_lr_001.py
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone this repo

```bash
git clone https://github.com/YashasviSah/svg-scaling-laws
cd svg-scaling-laws
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Clone nanoGPT (required for training)

```bash
git clone https://github.com/karpathy/nanoGPT.git
git clone https://github.com/karpathy/nanoGPT.git nanoGPT_mup
```

The notebook patches both copies automatically (adds metrics logging to `nanoGPT`, adds µP support to `nanoGPT_mup`).

---

## Reproducing the Results

All code is in the main notebook. Run cells top to bottom. Each part is labelled.

### Part 1 — Data Preprocessing

Cells 1–12. Downloads datasets from HuggingFace, cleans SVGs, trains a BPE tokenizer (vocab size 4096), and produces binary data files.

**Datasets used:**
- `starvector/svg-icons-simple` — 80,434 icons
- `starvector/svg-emoji-simple` — 4,114 emoji
- `starvector/svg-fonts-simple` — 65,000 font glyphs (subsampled)

**Outputs:** `train.bin` (109M tokens), `val.bin`, `test.bin`, `svg_tokenizer.json`, `meta.pkl`

### Part 2 — Standard Parameterization Scaling Study

Cells 13–39 (inside `nanoGPT/` directory).

**Learning rate sweep** on tiny model (5 LRs: 1e-4 to 1e-2):
```bash
cd nanoGPT
python train.py config/train_svg_sp_tiny_lr_1e-4.py
python train.py config/train_svg_sp_tiny_lr_3e-4.py
python train.py config/train_svg_sp_tiny_lr_1e-3.py
python train.py config/train_svg_sp_tiny_lr_3e-3.py
python train.py config/train_svg_sp_tiny_lr_1e-2.py
```

**Scale training** (best LR = 3e-3, 1 epoch each):
```bash
python train.py config/train_svg_sp_tiny.py
python train.py config/train_svg_sp_small.py
python train.py config/train_svg_sp_medium.py
python train.py config/train_svg_sp_large.py
python train.py config/train_svg_sp_xl.py
```

### Part 3 — µP Scaling Study

Cells 44–89 (inside `nanoGPT_mup/` directory).

**µP LR sweep** on tiny model:
```bash
cd nanoGPT_mup
python train.py config/train_svg_mup_tiny_lr_00001.py
python train.py config/train_svg_mup_tiny_lr_00003.py
python train.py config/train_svg_mup_tiny_lr_0001.py
python train.py config/train_svg_mup_tiny_lr_0003.py
python train.py config/train_svg_mup_tiny_lr_001.py
```

**Scale training** (best µP LR = 1e-2, 1 epoch each):
```bash
python train.py config/train_svg_mup_tiny_final.py
python train.py config/train_svg_mup_small_final.py
python train.py config/train_svg_mup_medium_final.py
python train.py config/train_svg_mup_large_final.py
python train.py config/train_svg_mup_xl_final.py
```

### Part 4 — Best Model Training & Generation

Cells 91–114 (inside `nanoGPT_mup/`).

Best model: **µP Medium** (val loss 1.22, test perplexity 8.31), trained for ~4 epochs.

```bash
python train.py config/train_svg_part4_resume_best.py
```

Generation and evaluation (XML validity, SVG render rate, prefix completion) are run from the notebook cells.

---

## Key Results

| Model | Params | SP Val Loss | µP Val Loss |
|-------|--------|-------------|-------------|
| Tiny   | 1.9M  | 1.639 | 1.454 |
| Small  | 4.3M  | 1.418 | 1.240 |
| Medium | 7.0M  | 1.314 | 1.669 |
| Large  | 17.5M | 1.150 | 4.296 |
| XL     | 91.6M | 4.512 | 4.768 |

**Best model (µP Small, trained for 4 epochs) evaluation:**
- Best val loss: 1.115 | Test loss: 1.140 | Test perplexity: 3.13
- XML validity rate: 82.4%
- SVG render rate: 82.4% (unconditional: 100%, prefix: 40%)
- Samples generated: 12 unconditional + 5 prefix-conditioned

## Hardware

All experiments run on Google Colab with an NVIDIA A100 GPU (High-RAM enabled).

Approximate training times per epoch:
- Tiny: ~2 min | Small: ~4 min | Medium: ~4 min | Large: ~9 min | XL: ~32 min

---

## References

- [nanoGPT](https://github.com/karpathy/nanoGPT) — Karpathy (base training code)
- [mup](https://github.com/microsoft/mup) — Microsoft (µP implementation)
- Kaplan et al. (2020) — Scaling Laws for Neural Language Models
- Hoffmann et al. (2022) — Chinchilla
- Yang et al. (2022) — Tensor Programs V (µP)
- Rodriguez et al. (2023) — StarVector
