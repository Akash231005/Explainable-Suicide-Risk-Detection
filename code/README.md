# SSN NeuralForge

## IEEE BigData 2026 — Explainable Suicide Risk Detection Challenge

**Team:** SSN NeuralForge

## 1. Project Objective

Given a Reddit post from r/SuicideWatch:
- **Subtask 1**: classify suicide risk level (Indicator / Ideation / Behavior / Attempt) and extract verbatim evidence phrases supporting that classification.
- **Subtask 2**: identify all applicable suicide-related factors from a 24-category taxonomy (multi-label).

## 2. Dataset Format

Two Excel files, provided separately by the competition organizers (not
included in this submission):

- `train.xlsx` — 1,635 labeled posts. Columns: `row_id`, `anon_user_id`,
  `post_id`, `post`, `suicide risk`, `evidence for suicide risk level`, `factors`.
- `leaderboard.xlsx` — 378 unlabeled posts for prediction. Columns:
  `row_id`, `anon_user_id`, `post_id`, `post`.

## 3. Folder Structure

```
SSN_NeuralForge_FINAL/
├── code/
│   ├── SSN_NeuralForge.ipynb
│   ├── README.md              (this file)
│   └── requirements.txt
├── report/
│   └── SSN_NeuralForge_Report.pdf
└── outputs/
    └── SSN NeuralForge.csv    final competition submission (378 rows)
```

## 4. Python Version

Python 3.9–3.12. Developed and tested on Python 3.12.

## 5. Required Packages

See `requirements.txt`. Core stack: `pandas`, `numpy`, `scipy`,
`scikit-learn`, `matplotlib`, `openpyxl`. Transformer sections
additionally need `torch` and `transformers` (Colab-only, see below).

## 6. Installation

```bash
pip install -r requirements.txt
```

## 7. Dataset Placement

Place `train.xlsx` and `leaderboard.xlsx` in a `data/` folder next to
the notebook (create this folder yourself — the datasets are not
included in this ZIP):

```
code/
├── SSN_NeuralForge.ipynb
└── data/
    ├── train.xlsx
    └── leaderboard.xlsx
```

## 8. Windows Execution

1. Install Python 3.9–3.12 from python.org.
2. Open a terminal (PowerShell or Command Prompt) in the `code/` folder.
3. `pip install -r requirements.txt`
4. Create the `data/` folder as described in Section 7 and copy both
   Excel files into it.
5. `jupyter notebook SSN_NeuralForge.ipynb`
6. Run all cells top to bottom. **Skip** the two Transformer sections
   (clearly marked "This section has NOT been executed... needs Colab")
   unless you have a local NVIDIA GPU with CUDA and internet access to
   `huggingface.co`.

## 9. Linux Execution

1. Ensure Python 3.9–3.12 is installed (`python3 --version`).
2. `cd` into the `code/` folder.
3. `pip install -r requirements.txt` (use a virtual environment if
   preferred: `python3 -m venv venv && source venv/bin/activate`).
4. Create the `data/` folder as described in Section 7 and copy both
   Excel files into it.
5. `jupyter notebook SSN_NeuralForge.ipynb`
6. Run all cells top to bottom, same guidance as Windows regarding the
   Transformer sections.

## 10. Google Colab Execution (full pipeline, including Transformers)

1. Upload `SSN_NeuralForge.ipynb` to Colab.
2. **Runtime → Change runtime type → T4 GPU → Save.**
3. Run the **GOOGLE COLAB SETUP** cell at the top — it installs `torch`
   and `transformers` and reports GPU availability automatically.
4. Upload `train.xlsx` and `leaderboard.xlsx` using the snippet in the
   markdown cell right before "Load Dataset" (creates a `data/` folder
   in the Colab session and lets you pick the files via a file-upload
   dialog).
5. **Runtime → Run all.**
6. Both Transformer sections (Risk: DistilBERT with a learning-rate
   sweep; Factors: multi-label DistilBERT) will train for real on the
   GPU. This is the only way to obtain a real Transformer validation
   score with this notebook.

## 11. GPU Requirement

Not required for the classical pipeline (runs on CPU in seconds to
minutes). Required only for the two Transformer sections — a free Colab
T4 GPU is sufficient; no local GPU is needed if using Colab.

## 12. How to Execute the Notebook

Run cells top to bottom ("Run All"). Every section is self-contained and
depends only on earlier cells within the same notebook, in order:
data loading → EDA → leakage check → preprocessing → train/validation
split → classical risk model → Transformer risk model (Colab-only) →
model comparison/ensemble → evidence extraction → classical factor
model → Transformer factor model (Colab-only) → Final Model Selection →
Submission Generation → Submission Validation.

## 13. Expected Output

Console output at each stage reporting real, measured validation scores
(never assumed or invented). A "Final Model Selection" printout showing
which model was actually chosen for each component and why. A generated
`outputs/SSN NeuralForge.csv` with 378 predictions. An 11-check
automated validation report confirming the submission file is correctly
formatted.

## 14. How the Final CSV Is Generated

The "Final Model Selection" section compares every model that was
**actually executed** — classical Logistic Regression vs. classical
Linear SVM (using their robust 5-seed validation means, not a single
split), and, if run, the Transformer and ensemble — and selects the
genuine best by validation score. This selection is a plain variable
(`selected_risk_source`, `selected_factor_source`) that the "Submission
Generation" section branches on directly, so the prediction code path
changes automatically based on what actually won. Predictions are then
generated on `leaderboard.xlsx` (378 unlabeled posts) using the selected
models, and saved to `outputs/SSN NeuralForge.csv`.

## 15. Subtask 1 — Risk Classification & Evidence Extraction

**Risk classification**: 4-class classification (Indicator, Ideation,
Behavior, Attempt) using TF-IDF (word + character n-grams) + Logistic
Regression as the validated classical model (Weighted F1 = 0.6468), with
a fully Colab-ready DistilBERT alternative.

**Evidence extraction**: phrase-level span extraction — posts are split
into sentences and sub-clauses, each candidate scored by a TF-IDF +
Logistic Regression classifier, and predictions above a tuned threshold
kept (preferring the shortest valid span). Every predicted span is an
exact verbatim substring of the source post — never generated or
paraphrased. Phrase F1 (local approximation of the official metric) =
0.5599.

## 16. Subtask 2 — Factor Identification

24-category multi-label classification using TF-IDF (word-only) +
One-vs-Rest Logistic Regression, with individually-tuned decision
thresholds per factor (rather than one fixed 0.5 cutoff for all 24
categories). Macro F1 = 0.4419. A fully Colab-ready multi-label
DistilBERT alternative is also implemented.

## 17. Known Limitations

- The Transformer sections are fully implemented and Colab-ready but
  were **not executed** in the environment used to build this notebook
  (no GPU, no internet access to `huggingface.co`). All reported scores
  are from the classical pipeline only; any Transformer score is
  reported as `NOT RUN` rather than fabricated.
- The factor classifier over-predicts on a subset (~15%) of long,
  vocabulary-dense posts — investigated and found to correlate with
  genuinely factor-dense posts rather than a code defect; capping this
  was tested and found to hurt validated Macro F1, so it was
  intentionally left uncapped and documented instead (see report).
