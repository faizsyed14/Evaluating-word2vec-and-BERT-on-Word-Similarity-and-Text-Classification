From Static to Contextual Word Embeddings
Evaluating word2vec and BERT on Word Similarity and Small-Scale Text Classification
A controlled comparison of static (word2vec) and contextual (BERT) word embeddings across two tasks:
Intrinsic evaluation — word similarity on five benchmarks (WordSim-353, SimLex-999, SimVerb-3500, MEN, SCWS)
Extrinsic evaluation — small-scale sentiment classification on SST-2, repeated across 5 random seeds
The project also includes a quantified polysemy experiment, layer-wise BERT probing, and statistical significance testing (bootstrap CIs, paired t-tests, McNemar's test) to determine whether observed differences are real or sampling noise.
Project Structure
├── W2VvsBERT.ipynb          # Main notebook — full experimental pipeline ├── Datasets/                # Auto-downloaded on first run (see below) │   ├── combined.csv             # WordSim-353 (353 pairs) │   ├── SimLex-999.txt             │   ├── SimVerb-3500.txt           │   ├── MEN_dataset_natural_form_full.txt │   └── ratings.txt               # SCWS ├── set1.csv / set2.csv      # WordSim-353 raw annotator splits (source material for combined.csv) └── README.md 
How the Pipeline Works
The notebook is designed to run top to bottom in Google Colab with zero manual file handling. Here is the exact execution flow:
Install dependencies
pip install gensim pip install transformers scikit-learn datasets statsmodels tqdm pip install torch 2. Pull datasets automatically
Datasets are not stored in this repo as large binary blobs. Instead, they are fetched at runtime from two sources:
Primary source — GitHub (raw files): small text/CSV benchmark files (combined.csv, SimLex-999.txt, SimVerb-3500.txt, MEN_dataset_natural_form_full.txt, ratings.txt) are pulled via wget from this repository's raw GitHub URLs into a local Datasets/W2VvsBERT/ folder.
Fallback — Google Drive: the same files are also mirrored in a shared Google Drive folder and can be pulled with gdown --folder <drive-folder-url> if GitHub is unreachable or you want to swap in your own dataset versions.
import os os.makedirs("Datasets/W2VvsBERT", exist_ok=True)  raw_base_url = "https://raw.githubusercontent.com/<your-username>/<your-repo>/main/Datasets/" files = ["combined.csv", "MEN_dataset_natural_form_full.txt", "ratings.txt",          "SimLex-999.txt", "SimVerb-3500.txt"]  for file in files:     os.system(f"wget -q -O Datasets/W2VvsBERT/{file} {raw_base_url}{file}") 
Note: SST-2 is not manually downloaded — it is pulled directly from the Hugging Face Hub via datasets.load_dataset("stanfordnlp/sst2") the first time Part 2 runs, and cached locally afterward.
3. Load shared models (once)
w2v_model  = load_word2vec_model(pretrained=True)   # word2vec-google-news-300 (~1.6 GB, gensim downloader) tokenizer, bert_model = load_bert_model()            # bert-base-uncased, 13 hidden layers (Hugging Face) 
4. Run Part 1 — Multi-benchmark word similarity
For each benchmark, both models score every word pair by cosine similarity; scores are correlated against human judgments via Spearman's ρ, with bootstrap 95% CIs on the BERT−word2vec difference.
comparison_df, pair_dfs = run_multi_benchmark_word_similarity(w2v_model, tokenizer, bert_model)

BERT word vectors are extracted using a neutral carrier sentence ("This is a {word}.") rather than a bare token, keeping BERT in-distribution. For SCWS, real sentence contexts are used directly instead.
5. Run the polysemy experiment
Quantifies whether BERT's within-sense similarity exceeds its across-sense similarity for 13 polysemous words (e.g., bank, bat, plant, pitch).
polysemy_df = run_polysemy_experiment(tokenizer, bert_model, w2v_model)
6. Run layer-wise probing
Sweeps all 13 BERT hidden-state layers to find which layer correlates best with human similarity judgments and which performs best for classification.
layer_sim_df = layer_wise_word_similarity_probe(df_wordsim, w2v_model, tokenizer, bert_model)
7. Run Part 2 — Multi-seed SST-2 classification
Loads a stratified, class-balanced subsample of SST-2 (500 train / 200 test, 250/100 per class), repeated across 5 seeds (10, 42, 123, 999, 2024). The same logistic regression classifier is trained on averaged word2vec vectors vs. mean-pooled BERT vectors.
per_seed_df, summary, t_test, mcnemar_result = run_multi_seed_classification(w2v_model, tokenizer, bert_model, seeds=(10, 42, 123, 999, 2024))
8. All results are saved to CSV automatically
word_similarity_multi_benchmark_summary.csv {dataset}_pairs.csv                (per-pair scores for each benchmark) polysemy_results.csv layer_wise_word_similarity.csv layer_wise_classification.csv classification_multi_seed_results.csv classification_disagreements.csv summary_results.csv                (final one-table summary of everything)
9. (Optional) Persist results to Google Drive
Colab's local disk is wiped on runtime restart. To keep results permanently:
from google.colab import drive drive.mount('/content/drive')  summary.to_csv('/content/drive/My Drive/Colab Notebooks/summary_results.csv', index=False)
Actual Results (seed 42 / 5-seed aggregate)


Polysemy: mean BERT within-sense vs. across-sense similarity gap = 0.221 (word2vec gap = 0 by construction, since it has no context sensitivity).
Best BERT layer: layer 1 for word similarity (ρ = 0.678); layer 7 for classification (F1 = 0.820).
SST-2 classification (mean ± std over 5 seeds): word2vec accuracy = 0.796 ± 0.027, BERT accuracy = 0.817 ± 0.033. Paired t-test on F1: t = 1.57, p = 0.191 — not statistically significant across seeds.
Embedding extraction cost: BERT took ≈557× longer than word2vec to embed the same sentences.
Takeaway: contextual BERT embeddings did not outperform static word2vec vectors on isolated word-pair similarity in this setup, and its edge in SST-2 classification was not statistically robust across seeds — while costing over 500× more compute. This nuances the common assumption that "contextual is always better," and is discussed in depth in the accompanying poster.
Requirements
gensim torch transformers scikit-learn datasets statsmodels scipy pandas numpy tqdm
Runtime Notes
Designed for Google Colab (CPU is sufficient; no GPU required).
Full run time: ~15–25 minutes, dominated by the word2vec model download (~1.6 GB) and BERT forward passes.
Set a Hugging Face token (HF_TOKEN) as a Colab secret to avoid rate-limit warnings on model downloads.
