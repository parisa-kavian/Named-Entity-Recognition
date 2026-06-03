[README.md](https://github.com/user-attachments/files/28556394/README.md)
<div align="center">

# Named Entity Recognition with BERT

### Token-level NER on CoNLL-2003, fine-tuned from `bert-base-uncased`

<a href="https://colab.research.google.com/github/parisa-kavian/Named-Entity-Recognition/blob/main/NER.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat-square&logo=huggingface&logoColor=black"/>
<img src="https://img.shields.io/badge/Model-bert--base--uncased-blue?style=flat-square"/>
<img src="https://img.shields.io/badge/Dataset-CoNLL--2003-orange?style=flat-square"/>
<img src="https://img.shields.io/badge/Metric-seqeval-success?style=flat-square"/>

</div>

> A **Named Entity Recognition** system that fine-tunes **`bert-base-uncased`** for token
> classification on the **CoNLL-2003** benchmark, identifying people, organizations,
> locations, and miscellaneous entities in raw text. The project ships both as a clean,
> end-to-end **notebook** and as a small, **layered codebase** (domain / application /
> infrastructure) that separates data, model, and metric concerns.

---

## What it does

NER extracts and classifies the key spans in a sentence. Trained on CoNLL-2003, the model
tags four entity types:

| Tag | Entity |
| --- | --- |
| **PER** | People |
| **ORG** | Organizations / institutions |
| **LOC** | Locations |
| **MISC** | Other named entities |

The core engineering detail is **sub-word label alignment**: BERT splits words into
word-pieces, so each original NER tag is re-mapped onto its sub-words, with special tokens
masked out using the ignore index `-100`.

---

## Pipeline

```
CoNLL-2003
   │  BertTokenizerFast (is_split_into_words=True)
   ▼
[ Tokenize + align labels to sub-words ]
   │
   ▼
[ Fine-tune bert-base-uncased for token classification ]
   │
   ▼
[ Evaluate with seqeval (P / R / F1 / accuracy) ] --> [ Inference pipeline ]
```

---

## Results

Evaluation uses **seqeval** and reports entity-level precision, recall, F1, and token
accuracy on the held-out **test** split (3 epochs, `bert-base-uncased`).

| Metric | Test |
| --- | --- |
| Precision | — |
| Recall | — |
| F1 | — |
| Accuracy | — |

<!-- After running the notebook, paste your run's numbers from the "Evaluate" cell here. -->

---

## Getting started

```bash
git clone https://github.com/parisa-kavian/Named-Entity-Recognition.git
cd Named-Entity-Recognition
pip install -r requirements.txt
```

**Option A — notebook (recommended):** open `NER.ipynb` in Jupyter or
[Colab](https://colab.research.google.com/github/parisa-kavian/Named-Entity-Recognition/blob/main/NER.ipynb)
and run the cells end to end. A GPU (e.g. Colab **T4**) is recommended.

**Option B — scripts:**

```bash
python main.py     # train and save the model + tokenizer to ./ner_model
python test.py     # load the saved model and run inference
```

---

## Repository structure

```
.
├── NER.ipynb                  # end-to-end notebook: tokenize -> train -> evaluate -> infer
├── main.py                    # entry point: trains and saves the model
├── test.py                    # loads the saved model and runs inference
├── domain/
│   └── models.py              # core data models (NER tokens / tags)
├── application/
│   └── use_cases.py           # training workflow orchestration
├── infrastructure/
│   ├── data_loader.py         # dataset loading + sub-word label alignment
│   ├── model_utils.py         # tokenizer/model construction, Trainer, saving
│   └── metric_utils.py        # seqeval metric + compute_metrics
├── requirements.txt
└── README.md
```

---

## Tech stack

`Hugging Face Transformers` · `Datasets` · `evaluate` · `seqeval` · `bert-base-uncased` · `PyTorch`

---

## Author

**Parisa Kavianpour** — Applied ML Researcher
[Website](https://parisa-kavian.github.io/) · [Google Scholar](https://scholar.google.com/citations?user=Y5cXz8QAAAAJ&hl=en) · [LinkedIn](https://www.linkedin.com/in/parisa-kavianpour/) · [GitHub](https://github.com/parisa-kavian)
