# Quora Question-Pair Similarity using NLP/ML

A lab-ready implementation of **Set No. 49 — CO-3 Experiment 3**. It extracts TF-IDF and pairwise features from two questions, combines them, and trains a Logistic Regression classifier to predict whether the questions are duplicates.

## Features used

1. **TF-IDF** with unigrams and bigrams.
2. **Absolute TF-IDF difference**: `|v1 - v2|`.
3. **Elementwise TF-IDF product**: `v1 * v2`.
4. **Pairwise numeric features**:
   - number of words in each question
   - absolute word-count difference
   - number of common words
   - Jaccard similarity
   - common content words and content-word Jaccard
   - word overlap ratio
   - exact-text match indicator
   - character-length difference
5. Logistic Regression classifier, train/test split, accuracy, classification report, and confusion matrix.

## Project structure

```text
quora-question-pair-similarity/
├── main.py
├── requirements.txt
├── README.md
└── data/
    └── quora_sample.csv
```

## Run in the lab

### 1. Check Python
```bash
python --version
```
If `python` does not work, try:
```bash
python3 --version
```

### 2. Install packages
From the project folder:
```bash
python -m pip install -r requirements.txt
```
If pip is unavailable:
```bash
python -m ensurepip --upgrade
python -m pip install -r requirements.txt
```

If the computer blocks installation or has no internet, try a preconfigured Python environment/Anaconda if available, or ask the lab instructor which environment is supported.

### 3. Execute
```bash
python main.py
```

To use another CSV:
```bash
python main.py --data path/to/your_file.csv
```

## Dataset format

CSV needs these three column names:

- `question1`
- `question2`
- `is_duplicate` — `1` for similar/duplicate questions, `0` for different questions.

The included CSV is a **small demonstration dataset** so the code can run without downloading the full Quora dataset. Results on this tiny dataset are only for demonstrating the pipeline and are not representative of real-world performance. For a meaningful evaluation, use a larger labelled Quora Question Pairs dataset.

## Short algorithm for viva

1. Read labelled question pairs from CSV.
2. Clean text by lowercasing and normalizing whitespace.
3. Split data into training and testing sets.
4. Fit TF-IDF on training questions and transform each question into a sparse vector.
5. For each pair, calculate absolute TF-IDF difference and elementwise product.
6. Extract pairwise lexical features such as Jaccard similarity, common words, and length differences.
7. Horizontally combine the sparse TF-IDF features and numeric pairwise features.
8. Train Logistic Regression on the combined features.
9. Predict test labels and evaluate using accuracy, precision, recall, F1-score, and confusion matrix.

## Important note

The first `pip install` needs internet access unless packages are already installed or cached. The script itself uses a local sample CSV and does not need to download a dataset.
