# From Static to Contextual Word Embeddings
### Evaluating word2vec and BERT on Word Similarity and Small-Scale Text Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-orange)
![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white)

A controlled comparison of static (**word2vec**) and contextual (**BERT**) word embeddings across two main NLP evaluation tasks:
1. **Intrinsic evaluation:** Word similarity across five standard benchmarks (`WordSim-353`, `SimLex-999`, `SimVerb-3500`, `MEN`, `SCWS`).
2. **Extrinsic evaluation:** Small-scale sentiment classification on `SST-2`, repeated across 5 random seeds.

The project also includes a **quantified polysemy experiment**, **layer-wise BERT probing**, and **rigorous statistical significance testing** (bootstrap CIs, paired t-tests, McNemar's test) to determine whether observed performance differences represent true architectural advantages or sampling noise.

---

## 📁 Repository Structure

```text
.
├── W2VvsBERT.ipynb          # Main notebook containing the full experimental pipeline
├── set1.csv / set2.csv      # WordSim-353 raw annotator splits (source material)
├── README.md                # Project documentation
└── Datasets/                # Auto-downloaded at runtime (not committed as blobs)
    ├── combined.csv         # WordSim-353 (353 pairs)
    ├── SimLex-999.txt       # SimLex-999 dataset
    ├── SimVerb-3500.txt     # SimVerb-3500 dataset
    ├── MEN_dataset_natural_form_full.txt
    └── ratings.txt          # SCWS dataset
```

---

## ⚙️ How the Pipeline Works

The main notebook (`W2VvsBERT.ipynb`) is designed to run top-to-bottom in Google Colab with zero manual file uploads.

### 1. Install Dependencies
```bash
pip install gensim transformers scikit-learn datasets statsmodels tqdm torch pandas numpy scipy
```

### 2. Automated Dataset Retrieval
Datasets are fetched directly at runtime into a local `Datasets/W2VvsBERT/` folder:
* **Primary Source (GitHub Raw):** Benchmark text/CSV files are pulled via `wget`.
* **Fallback (Google Drive):** Mirrored drive storage accessed via `gdown`.
* **Hugging Face Hub:** `SST-2` is loaded directly via `datasets.load_dataset("stanfordnlp/sst2")`.

```python
import os

os.makedirs("Datasets/W2VvsBERT", exist_ok=True)
raw_base_url = "https://raw.githubusercontent.com/<your-username>/<your-repo>/main/Datasets/"
files = ["combined.csv", "MEN_dataset_natural_form_full.txt", "ratings.txt", "SimLex-999.txt", "SimVerb-3500.txt"]

for file in files:
    os.system(f"wget -q -O Datasets/W2VvsBERT/{file} {raw_base_url}{file}")
```

### 3. Load Models
```python
# word2vec-google-news-300 (~1.6 GB via Gensim downloader)
w2v_model = load_word2vec_model(pretrained=True)   

# bert-base-uncased (13 hidden layers via Hugging Face)
tokenizer, bert_model = load_bert_model()            
```

### 4. Experimental Stages
* **Part 1 — Multi-Benchmark Word Similarity:** Evaluates cosine similarity against human judgments using Spearman's $\rho$ and 95% bootstrap confidence intervals for the difference ($\text{BERT} - \text{word2vec}$). Bare word vectors for BERT are extracted using a neutral carrier sentence (`"This is a {word}."`), while real sentence contexts are used for SCWS.
* **Polysemy Experiment:** Quantifies within-sense vs. across-sense similarity across 13 polysemous words (e.g., *bank*, *bat*, *plant*, *pitch*).
* **Layer-wise Probing:** Sweeps all 13 BERT hidden-state layers to identify optimal representation layers for word similarity vs. classification.
* **Part 2 — Multi-Seed SST-2 Classification:** Runs logistic regression on averaged word2vec vectors vs. mean-pooled BERT vectors across 5 random seeds (`10`, `42`, `123`, `999`, `2024`) on a stratified subsample (500 train / 200 test).

### 5. Automated Results Export
Generated outputs are automatically saved locally or synced to Google Drive:
* `word_similarity_multi_benchmark_summary.csv`
* `{dataset}_pairs.csv` (per-pair scores)
* `polysemy_results.csv`
* `layer_wise_word_similarity.csv`
* `layer_wise_classification.csv`
* `classification_multi_seed_results.csv`
* `summary_results.csv` (aggregated final summary)

---

## 📊 Key Results

### Intrinsic Benchmark Performance (Seed 42 / Aggregated)

| Benchmark | word2vec $\rho$ | BERT $\rho$ | 95% CI ($\text{BERT} - \text{w2v}$) | Verdict |
| :--- | :---: | :---: | :---: | :--- |
| **WordSim-353** | 0.694 | 0.670 | $[-0.097, 0.048]$ | Not significant |
| **SimLex-999** | 0.442 | 0.450 | $[-0.040, 0.055]$ | Not significant |
| **SimVerb-3500** | **0.364** | 0.258 | $[-0.137, -0.072]$ | **word2vec significantly better** |
| **MEN** | **0.782** | 0.721 | $[-0.080, -0.042]$ | **word2vec significantly better** |
| **SCWS** | **0.666** | 0.634 | $[-0.060, -0.003]$ | **word2vec significantly better** |

### Additional Experimental Findings
* **Polysemy Gap:** BERT achieved a **0.221** mean similarity gap between within-sense and across-sense usages. (Static word2vec has a gap of `0.0` by construction).
* **Layer Probing:**
  * **Word Similarity:** Layer 1 yielded the highest correlation ($\rho = 0.678$).
  * **Classification:** Layer 7 yielded the highest accuracy ($F_1 = 0.820$).
* **Extrinsic SST-2 Classification:**
  * **word2vec Accuracy:** $0.796 \pm 0.027$
  * **BERT Accuracy:** $0.817 \pm 0.033$
  * **Statistical Significance:** Paired t-test on $F_1$ scores resulted in $t = 1.57, p = 0.191$ (not statistically significant across random seeds).
* **Computational Efficiency:** Extracting embeddings via BERT took approximately **$557\times$ longer** than static word2vec lookups.

> **Takeaway:** Contextual BERT embeddings did not outperform static word2vec vectors on isolated word-pair similarity in this setup, and its performance edge on small-scale SST-2 classification was not statistically robust across random seeds—despite requiring over $500\times$ the compute cost.

---

## 🛠 Runtime Notes

* **Environment:** Designed primarily for **Google Colab**. CPU runtime is sufficient; a GPU is optional but recommended for faster BERT forward passes.
* **Execution Time:** ~15–25 minutes (dominated by downloading the ~1.6 GB word2vec vectors and running BERT forward passes).
* **Hugging Face API:** Setting an `HF_TOKEN` secret in Colab helps prevent potential download rate limits.
