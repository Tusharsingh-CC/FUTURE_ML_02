# Support Ticket Classification & Prioritization (NLP)

A decision-support system that reads a customer support ticket and:

1. **Categorizes** it (e.g. Billing, Technical issue, Refund, ...) with a TF-IDF + linear ML model.
2. **Prioritizes** it (Low / Medium / High / Critical) with a hybrid of an ML model and an explainable rule engine.
3. **Flags uncertain tickets** for human review and **sorts the queue** so urgent tickets are handled first.

> **Note:** `data/sample_tickets.csv` is a small *synthetic* file used only to smoke-test the pipeline.
> The results below come from the real IT Service Ticket dataset. Never quote metrics from the synthetic file.

## Why this matters for a support team

| Problem | What the system does |
|---|---|
| Tickets are not categorized properly | Model assigns a category in milliseconds, with a confidence score |
| Urgent issues get delayed | Priority engine escalates outages, security and money problems and sorts the queue |
| Agents waste time sorting | Routing is automatic; only low-confidence tickets go to a human |

## Project structure

```
src/
  __init__.py          (empty, makes `src` importable)
  load_data.py         dataset loader, auto-detects the two Kaggle schemas
  preprocess.py        text cleaning (shared by training and prediction)
  priority.py          explainable rule-based priority engine + hybrid logic
  train.py             model comparison, evaluation, plots, report, saves the model
  predict.py           triage new tickets (CLI)
  make_sample_data.py  synthetic data generator for smoke-testing only
results/               auto-generated REPORT.md, metrics.json, confusion matrices
models/                saved model (ticket_triage.joblib)
```

## Quick start

Run every command from the project root (the folder that contains `src/`), not from inside `src/`.

```bash
pip install -r requirements.txt

# 1. put a real dataset in data/ (see "Datasets" below), then:
python -m src.train --data data/customer_support_tickets.csv

# 2. triage a new ticket
python -m src.predict --text "URGENT: our site is down and customers cannot pay"

# 3. triage a whole file, sorted by urgency
python -m src.predict --input new_tickets.csv --text-col "Ticket Description" --output triaged.csv
```

The confidence threshold for human review defaults to 0.5 (`--review-threshold 0.6` to be stricter).

For any other CSV: `python -m src.train --data my.csv --text-cols Subject Body --category-col Queue [--priority-col Priority]`

## Datasets

| Dataset | Categories | Priority labels | Link |
|---|---|---|---|
| Customer Support Ticket Dataset (Kaggle) | Ticket Type (5) | Yes | kaggle.com/datasets/suraj520/customer-support-ticket-dataset |
| IT Service Ticket Classification (Kaggle) | Topic_group (8) | No (rule engine decides) | kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset |
| IT Support Tickets (Zenodo) | varies | varies | zenodo.org/records/7648117 |

Both Kaggle schemas are auto-detected. Tip: check on the first dataset whether the labels are actually
predictable from the text; if accuracy stays near chance, the labels are probably unrelated to the text,
and the IT Service dataset is the more meaningful benchmark.

## How tickets are categorized

1. **Cleaning** - lowercase; remove URLs, e-mails, template placeholders and numbers; expand contractions
   (`can't` -> `cannot`); remove stop words **except negations** (`not`, `no`, `never`, `cannot`) because
   "not working" means the opposite of "working".
2. **Features** - TF-IDF over unigrams and bigrams (sublinear term frequency, up to 50k features).
   Bigrams capture phrases like "charged twice" or "not working".
3. **Models compared** - Logistic Regression, Linear SVM (probability-calibrated), Complement Naive Bayes,
   Random Forest.
4. **Selection** - best model by **cross-validated macro-F1 on the training set only**. The held-out test
   set (20 %, stratified) is used once, for the final numbers, so the reported performance is not inflated.
5. **Output** - category + confidence. Below the confidence threshold the ticket is flagged `needs_human_review`.

## How priority is decided

**Rule engine (`priority.py`)** - a transparent point system on the raw text. Every point has a printed reason.

| Signal | Points |
|---|---|
| Hard-critical (outage, data loss/breach, hacked/fraud) | +6 (instantly Critical) |
| Urgency words (urgent, asap, blocked, deadline, losing money, ...) | +2 each, max +4 |
| Failure words (crash, error, not working, cannot, ...) | +1 each, max +2 |
| Frustration words (unacceptable, third time, ...) | +1 each, max +2 |
| Money impact (charged twice, overcharged) | +2 each, max +2 |
| Several `!` or SHOUTING | +1 each |
| Low-urgency wording (no rush, just wondering) | -2 |
| Category: technical/security/billing/refund | +1; informational categories -1 |

Negated hard signals ("we had **no** data loss", "was **not** hacked") are ignored.

Score >= 6 Critical, 4-5 High, 2-3 Medium, otherwise Low.

**Hybrid decision**
- Dataset **has** priority labels: a Logistic-Regression priority model predicts the level; the rule engine may
  only **escalate to Critical** on hard-critical signals and never downgrades a ticket (a missed outage is far
  more costly than one extra escalation).
- Dataset **has no** priority labels: the rule engine alone decides, and its reasons are shown to the agent.

## Evaluation

`results/REPORT.md` is generated automatically and contains: model comparison table, per-class
precision/recall/F1, confusion matrices, most-confused category pair, most indicative words per category,
confidence analysis, and priority results for rules-only vs ML-only vs hybrid.

Metrics: accuracy, **macro-F1** (treats all categories equally, so rare categories are not hidden by big ones),
weighted-F1, and **recall on Critical tickets** for priority, since missing a critical ticket is the costly error.

## Results (IT Service Ticket Classification, ~47k tickets, 8 categories)

| Metric | Value |
|---|---|
| Macro-F1 (held-out test) | ~0.86 (cross-validation matched, so no sign of overfitting) |
| Best models | Logistic Regression and Linear SVM (tied) |
| Easiest / hardest category | Purchase (F1 ~0.93) / Administrative rights (F1 ~0.77) |
| Most common error | other categories predicted as Hardware |
| Auto-routed (confidence >= 0.5) | ~89% of tickets, ~90% accurate |
| Flagged for human review | ~11% of tickets, ~49% accurate, so sending them to a person is justified |

Full tables and confusion matrices are in `results/REPORT.md`.

**Customer Support Ticket dataset (Kaggle, first dataset):** accuracy stayed near chance level (confidence ~0.24
across 5 classes), which suggests its labels are not related to the ticket text. It is therefore not used as a benchmark.

## Limitations and next steps

- **No priority labels in the IT dataset.** Priority is explained by the rule engine but not statistically validated.
  Next step: have a support lead check ~50 tickets and compare. Rule weights are hand-set; soft signals alone can
  reach Critical (e.g. "blocked + deadline today"), so tune them with the team's own SLA definitions.
- **~10% of auto-routed tickets are still wrong.** Auto-routing is a time saver, not a guarantee; keep the human-review path.
- **Public priority labels are noisy.** The first dataset's labels did not relate to the text.
- Tickets that mix several intents ("cancel my plan and refund me") are inherently ambiguous - a multi-label model would help.
- Next steps: transformer embeddings (DistilBERT / sentence-transformers), SHAP explanations,
  a feedback loop where agent corrections are used to retrain, and drift monitoring.
