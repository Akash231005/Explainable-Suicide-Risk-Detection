# SSN NeuralForge — Explainable Suicide Risk Detection

**IEEE BigData 2026 — Explainable Suicide Risk Detection Challenge**

**Team:** SSN NeuralForge (SSN College of Engineering, Chennai, India)
**Authors:** Afras Ahamed N, Akash S, Anish M
**Contact:** akashs231005@gmail.com

> **Research prototype. Not a clinical or diagnostic tool.** It is meant to support
> trained human reviewers, never to replace them or to trigger action on an
> individual automatically. If you or someone you know is in crisis, contact local
> emergency services or a crisis line.

---

## 1. What this project does

Given a Reddit post from r/SuicideWatch, the system produces three linked outputs:

| Output               | Task      | Detail                                                                    |
| -------------------- | --------- | ------------------------------------------------------------------------- |
| **Risk level** | Subtask 1 | 4 classes: Indicator, Ideation, Behavior, Attempt                         |
| **Evidence**   | Subtask 1 | Exact phrase(s) copied from the post that support the label               |
| **Factors**    | Subtask 2 | Multi-label prediction over a 24-category suicide-related factor taxonomy |

Evidence is **never paraphrased**: every returned span is an exact substring of the
original post.

---

## 2. Two pipelines (and why both are here)

|            | Earlier pipeline ("without")                                     | Current pipeline ("with")                                                                                                |
| ---------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Notebook   | `SSN_NeuralForge.ipynb`                                        | `risk_classification_colab.ipynb`                                                                                      |
| Risk       | Word + character TF-IDF, logistic regression                     | Fine-tuned `mental/mental-roberta-base` (5 folds) blended with a TF-IDF member (weight 0.20)                           |
| Factors    | Word TF-IDF, one-vs-rest logistic regression                     | Multi-label MentalRoBERTa, 24 outputs, per-factor thresholds tuned out-of-fold                                           |
| Evidence   | Phrase-level span scorer (TF-IDF + logistic regression)          | LLM (`openai/gpt-oss-120b` via Groq) proposes spans; only exact matches in the post are kept, else a fallback sentence |
| Validation | User-grouped split, 5 seeds (risk) / 3 seeds (evidence, factors) | Nested user-grouped protocol: 20% held-out + 5-fold grouped CV, held-out scored once                                     |
| Hardware   | CPU                                                              | GPU (Colab T4)                                                                                                           |

The earlier pipeline is kept as the reproducible baseline and as a fully local,
auditable fallback.

---

## 3. Results (local validation, not official leaderboard scores)

Measured on the **same held-out set (330 posts, 32 users)**. The earlier pipeline was
re-run on the identical split so the comparison is like-for-like.

| Task     | Metric           | Without | With                                                   |
| -------- | ---------------- | ------- | ------------------------------------------------------ |
| Risk     | Accuracy         | 0.6242  | **0.7667** (95% CI 72.1–80.9%, users resampled) |
| Risk     | Weighted F1      | 0.6209  | **0.7671**                                       |
| Risk     | Macro F1         | 0.5631  | **0.7513**                                       |
| Factors  | Macro F1         | 0.3888  | **0.4811**                                       |
| Factors  | Micro F1         | 0.5465  | **0.6315**                                       |
| Evidence | Token F1 (proxy) | 0.2291  | **0.3180**                                       |

Development-set (out-of-fold, n = 1,305) risk accuracy is 75.48% (macro F1 71.36%).

**Read these numbers carefully:**

- The held-out set is small (330 posts, 32 users); the confidence interval is about ±4.4 points. Attempt has only 20 held-out posts.
- Most of the risk gain comes from the encoder. The TF-IDF blend is within noise (−0.002 out-of-fold, +0.006 held-out) and its benefit is not demonstrated.
- Evidence is scored with a token-overlap proxy, **not** the official span metric. The LLM evidence step has no Phrase F1 yet, and the proxy scores `none` references as 0 whether or not the system abstains.
- Per-factor thresholds gave the classical model no gain once thresholds were cross-fitted; an earlier +0.056 gain came from tuning on the evaluation split.
- The earlier pipeline's own numbers (risk weighted F1 0.6468, Phrase F1 0.5599, factor macro F1 0.4419) were measured on a *different* split and must not be compared with the table above.

---

## 4. Repository layout (suggested)

```
.
├── README.md
├── notebooks/
│   ├── SSN_NeuralForge.ipynb               # earlier classical pipeline
│   └── risk_classification_colab.ipynb     # current transformer pipeline
├── data/                                    # NOT included (organizer-provided)
│   ├── train.xlsx                           # 1,635 labeled posts
│   └── final_prediction_input.xlsx          # 502 unlabeled posts
├── outputs/
│   └── SSN NeuralForge.csv                  # earlier pipeline's prediction file
├── report/
│   ├── SSN_NeuralForge_IEEE_Paper_v2.pdf
│   └── SSN_NeuralForge_IEEE_Paper_v2_source.zip   # LaTeX (IEEEtran) + figures + analysis scripts
└── slides/
    └── SSN_NeuralForge_IEEE_BigData_2026.pptx
```

---

## 5. Data

Provided by the challenge organizers; **not redistributed here**.

- `train.xlsx`: 1,635 labeled posts from 153 anonymized users. Columns: `row_id`, `anon_user_id`, `post_id`, `post`, `suicide risk`, `evidence for suicide risk level`, `factors`.
- `final_prediction_input.xlsx`: 502 unlabeled posts from 48 users, disjoint from the training users.
- Class distribution: Indicator 611 (37.4%), Ideation 519 (31.7%), Behavior 391 (23.9%), Attempt 114 (7.0%).
- 583 posts (95.4% of Indicator) have the evidence annotation `none`.

> The previous README stated 378 prediction posts. That was stale; the file contains 502.

---

## 6. Running the current pipeline (Google Colab)

1. Upload `risk_classification_colab.ipynb` to Colab.
2. **Runtime → Change runtime type → T4 GPU.**
3. Upload `train.xlsx` (left sidebar → Files).
4. For the evidence step, add a **`GROQ_API_KEY`** under Colab **Secrets** and enable notebook access. Without it, the evidence step is skipped.
5. **Run all.** Per-fold results are cached (to Google Drive if you allow mounting), so after a disconnect, re-running skips finished folds.

Main settings are in the `CONFIG` cell: `RUN_FACTORS`, `RUN_EVIDENCE`, `USE_TFIDF_MEMBER`, `N_FOLDS = 5`, `MAX_LENGTH = 512` (head + tail kept), `SEEDS = [42]`. Bump `CODE_VERSION` after editing training logic so stale cached folds are ignored.

**Outputs written by the notebook:** `heldout_predictions.csv`, `heldout_analysis.csv`, `model_metrics.csv`, `factor_thresholds.csv` (copied to Drive when mounted).

**Known issue:** the configured DeBERTa-v3 backbone stopped with an FP16 gradient-unscaling error, so it is not part of the reported system.

### Reproducing the held-out split

The split uses `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` on `anon_user_id`. scikit-learn 1.8 produces a **different** split from versions 1.5–1.7. To reproduce the reported held-out set (113 / 117 / 80 / 20 posts per class), use scikit-learn ≤ 1.7.

---

## 7. Running the earlier classical pipeline (local, CPU)

- Python 3.9–3.12. Core packages: `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `openpyxl`.
- Place `train.xlsx` and the prediction file in a `data/` folder next to the notebook, then run all cells top to bottom.
- Runs on CPU in seconds to minutes. Its transformer sections are Colab-only and were not completed.
- Note: some saved cell outputs in that notebook are stale (older runs). Re-executing the cells regenerates the submission file; trust re-executed values over saved ones.

---

## 8. Evaluation protocol

- **No user leakage:** all splits are grouped by `anon_user_id`.
- A 20% held-out set (330 posts, 32 users) is set aside; the remaining 1,305 posts (121 users) go through 5-fold grouped cross-validation.
- All choices (TF-IDF blend weight, factor thresholds, class-bias test) are made on out-of-fold predictions; the held-out set is scored once.
- Confidence intervals resample **users** (2,000 bootstraps).
- Out-of-fold scores are slightly optimistic (best epoch is picked per fold), so quote held-out numbers.
- One split seed (42) is used in the current protocol.

---

## 9. Limitations and responsible use

- Small dataset and held-out set; rare factors are weak and over-predicted.
- One split seed; individual fine-tuning settings (512 tokens, layer-wise decay, label smoothing, class weights) were changed together and not ablated separately.
- Evidence uses an LLM selector; it never abstains, so it returns text even for Indicator posts whose reference is `none`.
- **Privacy:** the evidence step sends post text to a third-party hosted API. Check the data-use terms before using it on real data; a locally hosted model is preferable.
- The evidence output is *model-supported textual evidence*, not clinical validation, and need not reflect what drove the risk classifier.
- Local validation may differ from official leaderboard results.

---

## 10. Planned next steps

Multiple seeds and backbones with significance tests (including a repaired DeBERTa-v3 run); ablating each fine-tuning setting; storing evidence predictions to compute the official Phrase F1 and allowing abstention on `none` posts; a local open model for evidence; calibration and uncertainty estimation; fairness and cross-domain evaluation; human-in-the-loop evaluation with clinicians.

---

## 11. Citation

If you use this work, please cite the accompanying report:

> A. A. N, A. S, and A. M, "Explainable Suicide Risk Detection from Social-Media Text: Fine-Tuned Mental-Health Transformers, Verbatim Evidence, and Factor Identification," Team SSN NeuralForge, SSN College of Engineering, 2026.

License: *to be decided by the tea*

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
